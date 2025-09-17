import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from nets.selectlayer import ModelWithSelector
from nets.selectlayer import EvalForRam
from utils.log_utils import log_values
from utils import move_to
torch.autograd.set_detect_anomaly(True)
#from torch.cuda.amp import autocast, GradScaler

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(Selectlayer, model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(Selectlayer, model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(Selectlayer, model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    Selectlayer.eval()
    
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _ = EvalForRam(Selectlayer, model, move_to(bat, opts.device))  
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

# import gc
# import time
# from tqdm import tqdm
# from torch.utils.data import DataLoader

# def train_epoch(Selectlayer, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
#     print(f"Start train epoch {epoch}, lr={optimizer.param_groups[0]['lr']} for run {opts.run_name}")
#     step = epoch * (opts.epoch_size // opts.batch_size)
#     start_time = time.time()

#     if not opts.no_tensorboard:
#         tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

#     # 优化1：使用生成器模式生成数据，避免一次性加载全量数据
#     def data_generator():
#         """分批生成数据，每次生成一个batch的数据量"""
#         remaining = opts.epoch_size
#         while remaining > 0:
#             # 每次生成不超过batch_size的数据（最后一批可能更少）
#             batch_size = min(opts.batch_size, remaining)
#             dataset = problem.make_dataset(
#                 dependency=opts.priority,
#                 size=opts.graph_size,
#                 num_samples=batch_size,
#                 distribution=opts.data_distribution
#             )
#             wrapped = baseline.wrap_dataset(dataset)
#             yield from wrapped  # 逐个返回样本，避免批量存储
#             remaining -= batch_size

#     # 优化2：DataLoader配置优化（num_workers=0避免进程复制，pin_memory=False减少内存）
#     training_dataloader = DataLoader(
#         list(data_generator()),  # 转换为列表（如需多进程可保留生成器，但内存更省）
#         batch_size=opts.batch_size,
#         num_workers=0,  # 禁用多进程，减少内存开销
#         pin_memory=False,  # 非GPU场景禁用，节省内存
#         shuffle=True  # 如需打乱，在生成器内或此处处理
#     )

#     # 模型训练模式
#     model.train()
#     set_decode_type(model, "sampling")
#     Selectlayer.train()

#     for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
#         # 训练批次
#         train_batch(
#             Selectlayer,
#             model,
#             optimizer,
#             baseline,
#             epoch,
#             batch_id,
#             step,
#             batch,
#             tb_logger,
#             opts
#         )

#         step += 1

#         # 优化3：及时释放当前batch内存并触发垃圾回收
#         del batch
#         gc.collect()  # 强制回收未使用的内存

#     # 优化4：释放数据集和数据加载器内存
#     del training_dataloader
#     gc.collect()

def train_epoch(Selectlayer, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(dependency = opts.priority,
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    Selectlayer.train()

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        """epsilon = Selectlayer.get_epsilon()
        t = step / ((opts.epoch_start + opts.n_epochs) * (opts.epoch_size // opts.batch_size))  
        epsilon = 0.015 + (epsilon - 0.015) * math.exp(-7 * t)
        epsilon = max(0.02, epsilon)
        Selectlayer.set_epsilon(epsilon)"""
        train_batch(
            Selectlayer,
            model,
            optimizer,
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
                'optimizer': optimizer.state_dict(),
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

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        Selectlayer,
        model,
        optimizer,
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

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

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
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
        
    del cost, log_likelihood, bl_val, loss
    torch.cuda.empty_cache()
