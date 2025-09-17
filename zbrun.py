#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from nets.selectlayer import DQNAgent
#from nets.rainbow import DQNAgent

import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from nets.attention_model import set_decode_type
from utils import move_to

def train_epoch(Selectlayer, model,  baseline,  epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {},  for run {}".format(epoch, opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(dependency = opts.priority,
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    model.eval()
    set_decode_type(model, "sampling")
    Selectlayer.train()

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            Selectlayer,
            model,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'Selectlayer' : get_inner_model(Selectlayer).state_dict(),
                'model': get_inner_model(model).state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(Selectlayer, model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(Selectlayer, model, epoch)

def train_batch(
        Selectlayer,
        model,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    #cost, log_likelihood = model(x)
    cost, log_likelihood, dqn_transitions = ModelWithSelector(Selectlayer, model, x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * (log_likelihood)).mean()
    loss = reinforce_loss + bl_loss

    for transition in dqn_transitions:
        Selectlayer.bufferpush(
            transition['state'],
            transition['action'],
            transition['reward'],
            transition['next_state'],
            transition['done']
        )
    # 当经验池数据足够时，由 Selectlayer（智能体）执行优化更新
    if Selectlayer.getmemory() >= Selectlayer.getbuff_szie():
        Selectlayer.optimize(tb_logger, step)

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
        
    del cost, log_likelihood, bl_val, loss
    torch.cuda.empty_cache()


def log_values(cost, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)

def ModelWithSelector(Selectlayer, model, data): 
    """
    利用选择器网络（DQNAgent，即 Selectlayer）和 attention model（model）对输入 x 执行 m 次动作选择，
    每一步屏蔽已选点、累计 log 概率，最终将选择结果传入 attention model 得到 cost，
    并记录选择过程中的状态转移信息供 DQN 更新使用。
    """

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)

    batch_size, n, _ = x.size()
    device = x.device
    m = Selectlayer.get_select_size()

    selected_mask = torch.zeros((batch_size, n), device=device)  # 用来标记已选择的任务
    selections = []       # 存储每个步骤的选择任务
    dqn_transitions = []  # 存储状态转移信息

    # 将 x 和 selected_mask 拼接为新的 state
    state = torch.cat((x, selected_mask.unsqueeze(-1)), dim=-1)  # 拼接 x 和 selected_mask
    next_state = state.clone()  # 初始化 next_state 与 state 相同

    for i in range(m):
        # 计算当前状态下的 Q 值，并对已选任务进行屏蔽
        q_values = Selectlayer(state)
        # 将已选择任务的 Q 值设置为负无穷，避免重复选择
        q_values = q_values.masked_fill(selected_mask == 1, -float('inf'))
        # epdilon greed
        """probs = F.softmax(q_values, dim=-1)
        taken = (torch.rand(batch_size, device=device) < Selectlayer.get_epsilon()).int()
        actions = taken*torch.multinomial(probs, 1).squeeze(-1) + (1-taken)*q_values.argmax(dim=-1)"""
        actions = q_values.argmax(dim=-1)
        selections.append(actions)
        # 更新状态，标记已选择任务
        selected_mask[torch.arange(batch_size), actions] = 1  # 更新已选择的任务
        # 最后一步标记为 done，奖励为 cost（稍后更新）
        done = torch.full((batch_size,), False, device=device)
        reward = torch.zeros(batch_size, device=device)
        if i == m - 1:
            done = torch.full((batch_size,), True, device=device)
        # 更新 next_state，保持任务特征信息（state 和 selected_mask 的拼接）
        next_state = torch.cat((x, selected_mask.unsqueeze(-1)), dim=-1)  # 更新拼接的 next_state
        # 记录状态转移
        dqn_transitions.extend([{
            'state': state[b].detach().cpu(),
            'action': actions[b].item(),
            'reward': reward[b].item(),
            'next_state': next_state[b].detach().cpu(),
            'done': done[b].item()
        } for b in range(batch_size)])
        # 将更新后的 next_state 作为新的 state
        state = next_state.clone()
        del q_values

    selections = torch.stack(selections, dim=1) + 1 # (batch_size, m)
    # 将选择的任务传入注意力模型，计算 cost
    with torch.no_grad():
        cost, log_likelihood = model(data, selections)
    # 更新奖励为负的 cost
    for idx, transition in enumerate(dqn_transitions[-batch_size:]):
        transition['reward'] = -cost[idx].item()  # 奖励为 cost 的负值

    return cost, log_likelihood, dqn_transitions

def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,

        dependency=opts.priority,
        sub_encode_layers=opts.sub_encode_layers,   

        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    
    del load_data
    import gc
    gc.collect()


    Selectlayer = DQNAgent(
        feature_dim = 8*opts.graph_size ,
        action_size = opts.graph_size,
        select_size = opts.select_size,
        dqn_device = opts.device,
        logstep= opts.log_step,
        lr = 1e-5, 
        gamma = 0.99
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        Selectlayer = torch.nn.DataParallel(Selectlayer)

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(Selectlayer, model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)


    # Start the actual training loop
    val_dataset = problem.make_dataset(dependency = opts.priority,
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(Selectlayer, model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(Selectlayer, model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                Selectlayer,
                model,
                baseline,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

            torch.cuda.empty_cache()

if __name__ == "__main__":
    run(get_options())


