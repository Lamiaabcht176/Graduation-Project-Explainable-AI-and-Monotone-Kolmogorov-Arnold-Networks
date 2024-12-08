import yaml
from KAN import KAN
import torch
from utils import SYMBOLIC_LIB

import yaml
import torch

def kan_saveckpt(model, path='model'):
    """
    Save the KAN model's state and configuration to files.
    
    Parameters:
    -----------
    model: KAN
        The KAN model to save.
    path: str
        The base path for saving the checkpoint (default is 'model').
    """
    # Save configuration as a dictionary
    dic = dict(
        width=model.width,
        grid=model.grid,
        k=model.k,
        base_fun_name=model.base_fun.__class__.__name__,  # Save the name of the base function
        symbolic_enabled=model.symbolic_enabled,
        bias_trainable=model.bias_trainable,
        grid_eps=model.grid_eps,
        grid_range=model.grid_range,
        sp_trainable=model.sp_trainable,
        sb_trainable=model.sb_trainable,
        device=model.device,
    )

    # Save symbolic function names for each layer if symbolic mode is enabled
    if model.symbolic_enabled:
        for i in range(model.depth):
            dic[f'symbolic_fun.{i}'] = [
                [f for f in layer] for layer in model.symbolic_fun[i].funs_name
            ]
    
    # Save configuration to a YAML file
    with open(f'{path}_config.yml', 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), f'{path}_state')


def kan_loadckpt(path='model'):
    """
    Load a KAN model from checkpoint files.
    
    Parameters:
    -----------
    path: str
        The base path for the checkpoint files (default is 'model').
    
    Returns:
    --------
    KAN:
        The loaded KAN model.
    """
    # Load configuration
    with open(f'{path}_config.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    
    # Initialize the KAN model with the loaded configuration
    model = KAN(
        width=config['width'],
        grid=config['grid'],
        k=config['k'],
        base_fun=getattr(torch.nn, config['base_fun_name'])(),  # Dynamically load the base function
        symbolic_enabled=config['symbolic_enabled'],
        bias_trainable=config['bias_trainable'],
        grid_eps=config['grid_eps'],
        grid_range=config['grid_range'],
        sp_trainable=config['sp_trainable'],
        sb_trainable=config['sb_trainable'],
        device=config['device'],
    )
    
    # Load the model's state dictionary
    state = torch.load(f'{path}_state')
    model.load_state_dict(state)

    return model
