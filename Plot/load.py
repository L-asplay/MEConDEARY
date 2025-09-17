import os
import torch

from utils.functions import load_problem, torch_load_cpu, load_args

def load_cmo(path, epoch=None) :
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
    return model_filename, args, problem

def load_file(load_path, model, selzion):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""
    # Load the model parameters from a saved state
    print('  [*] Loading {} from {}'.format(selzion, load_path))
    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)
    if isinstance(load_data, dict):
        load_model_state_dict = load_data.get(selzion, load_data)
    else:
        load_model_state_dict = load_data.state_dict()
        assert(0), "Wrong on _load_mec_file, load_data is not dict"

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)

    return model

def load_att(path, epoch=None):
    if path is None :   assert 0, "Dont def the load path"
    from nets.attention_model import AttentionModel
    model_filename, args, problem =  load_cmo(path, epoch)

    model = AttentionModel(args['embedding_dim'], 
        args['hidden_dim'], problem, n_encode_layers=args['n_encode_layers'],   
        dependency=args['priority'], sub_encode_layers=args['sub_encode_layers'], mask_inner=True,
        mask_logits=True, normalization=args['normalization'], tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False), shrink_size=args.get('shrink_size', None)
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model = load_file(model_filename, model, 'model')
    
    model.eval() 
    return model

def load_dqn(path, epoch=None):
    if path is None : assert 0, "Dont def the load path"
    from nets.selectlayer import DQNAgent
    model_filename, args, _ =  load_cmo(path, epoch)

    Selectlayer = DQNAgent(feature_dim = 8*args['graph_size'],
        action_size = args['graph_size'], select_size = args['select_size'],
        dqn_device = 'cpu', logstep= args['log_step'],
        lr = 1e-5, gamma = 0.99
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    Selectlayer.load_state_dict({**Selectlayer.state_dict(), **load_data.get('Selectlayer', {})})
    Selectlayer = load_file(model_filename, Selectlayer, 'Selectlayer')

    Selectlayer.eval()
    return Selectlayer

def load_bow(path, epoch=None):
    if path is None : assert 0, "Dont def the load path"
    from nets.rainbow import DQNAgent
    model_filename, args, _ =  load_cmo(path, epoch)

    Selectlayer = DQNAgent(feature_dim = 8*args['graph_size'],
        action_size = args['graph_size'], select_size = args['select_size'],
        dqn_device = 'cpu', logstep= args['log_step'],
        lr = 1e-5, gamma = 0.99
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    Selectlayer.load_state_dict({**Selectlayer.state_dict(), **load_data.get('Selectlayer', {})})
    Selectlayer = load_file(model_filename, Selectlayer, 'Selectlayer')

    Selectlayer.eval()
    return Selectlayer