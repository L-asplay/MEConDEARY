import torch

def att(model):
    return lambda mask, x : forw(model, mask, x)

def forw(model, local_mask, data):
    loc_indices = torch.nonzero(local_mask, as_tuple=False)
    selections = loc_indices[:,1:] + 1
    selections = selections.view(local_mask.size(0), -1)
    cost, _, pi = model(data, selections, return_pi=True)
    return cost, pi
