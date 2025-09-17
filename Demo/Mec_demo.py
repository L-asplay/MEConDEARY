import os
import torch
import argparse
from Demo.greed import congreed
from Demo.comute import EvalForDemo
from torch.utils.data import DataLoader
from utils.functions import load_problem, torch_load_cpu, load_args


def _load_mec_file(load_path, model, Selectlayer):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""
    # Load the model parameters from a saved state
    print('  [*] Loading model from {}'.format(load_path))
    load_data = torch.load(os.path.join( os.getcwd(),load_path), map_location=lambda storage, loc: storage)
    if isinstance(load_data, dict):
        load_model_state_dict = load_data.get('model', load_data)
        load_Selectlayer_state_dict = load_data.get('Selectlayer', load_data)
    else:
        load_model_state_dict = load_data.state_dict()
        assert(0), "Wrong on _load_mec_file, load_data is not dict"
    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    selec_dict = Selectlayer.state_dict()
    selec_dict.update(load_Selectlayer_state_dict)
    Selectlayer.load_state_dict(selec_dict)
    return model, Selectlayer 


def load_mec(path, epoch=None):
    if path is None :
        assert 0, "Dont def the load path"
    from nets.attention_model import AttentionModel
    #from nets.selectlayer import DQNAgent
    from nets.rainbow import DQNAgent
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path) if os.path.splitext(filename)[1] == '.pt')
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)
    args = load_args(os.path.join(path, 'args.json'))
    problem = load_problem(args['problem'])
    model = AttentionModel(args['embedding_dim'], 
        args['hidden_dim'], problem, n_encode_layers=args['n_encode_layers'],   
        dependency=args['priority'], sub_encode_layers=args['sub_encode_layers'], mask_inner=True,
        mask_logits=True, normalization=args['normalization'], tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False), shrink_size=args.get('shrink_size', None)
    )
    Selectlayer = DQNAgent(feature_dim = 8*args['graph_size'],
        action_size = args['graph_size'], select_size = args['select_size'],
        dqn_device = 'cpu', logstep= args['log_step'],
        lr = 1e-5, gamma = 0.99
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    Selectlayer.load_state_dict({**Selectlayer.state_dict(), **load_data.get('Selectlayer', {})})
    model, Selectlayer = _load_mec_file(model_filename, model, Selectlayer)
    model.eval()  # Put in eval mode
    model.set_decode_type('greedy')
    Selectlayer.eval()
    return model, Selectlayer


def demo(opts):
    # Set the random seed
    torch.manual_seed(opts.seed)
    # Figure out what's the problem
    problem = load_problem(opts.problem)
    # Load model and agent
    #model, Selectlayer = load_mec('outputs/mec_30/run_20250414T235633/', opts.num_epoch)
    model, Selectlayer = load_mec(opts.load_path, opts.num_epoch)
    # Make dataset

    dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.num_samples, dependency=model.dependency)
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))
    
    # Take the cost and pi
    from Demo.SAX import skoSA
    from Demo.GAX import skoGA
    from Demo.ATT import att
    algorithm0, algorithm1  = skoGA, att(model)
    skores, cost0, pi0 = congreed(algorithm0, batch, Selectlayer.get_select_size()) 
    attres, cost1, pi1 = congreed(algorithm1, batch, Selectlayer.get_select_size()) 
    skores = skores + 1 # 任务对齐
    attres = attres + 1

    with torch.no_grad():
        energy2, energy3 = [], []
        cost2, _, pi2 = model(batch, skores, return_pi=True)
        cost3, _, pi3 = EvalForDemo(Selectlayer, model, batch)
        energy2.append(cost2)
        energy3.append(cost3)
        for i in range(19) :
            cost2i, _, pi2i = model(batch, skores, return_pi=True)
            cost3i, _, pi3i = EvalForDemo(Selectlayer, model, batch)
            if cost2i.item() < cost2.item() :
                pi2 = pi2i
            if cost3i.item() < cost3.item() :
                pi3 = pi3i
            energy2.append(cost2i)
            energy2.append(cost3i)
        energy2 = torch.stack(energy2).mean(dim=0)
        energy3 = torch.stack(energy3).mean(dim=0)
  
    return batch, [skores, attres], [cost0, cost1, cost2, cost3], [pi0, pi1, pi2, pi3]


def demo_options(args=None):
    parser = argparse.ArgumentParser(description="The sets in demo")
    parser.add_argument('--problem', default='mec', help="The problem to solve, default 'mec'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--num_samples', type=int, default=1, help="The size of the dataset batch")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--load_path', default=None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--num_epoch', default=None, help='Number of load epoch')
    parser.add_argument('--priority', nargs='+', type=int, default=[], help='The dependency sequence for the problem')

    parser.add_argument('--result', default='plsave.npz', help="The npz saved result")
    parser.add_argument('--pic', default='plot.png', help="The pic saved result")
    parser.add_argument('--load_path1', default=None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--load_path2', default=None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--cost', default='./zz/cost.npz')
    parser.add_argument('--spend', default='./zz/spend.npz')
    parser.add_argument('--energy', default='./zz/energy.png')
    parser.add_argument('--time', default='./zz/time.png')
    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available()
    return opts
