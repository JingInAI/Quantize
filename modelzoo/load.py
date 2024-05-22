"""
Model Loader
version: 0.0.3
update: 2024-04-19
"""
from copy import deepcopy
from utils import Register, Configs, most_similar

MODELS = Register()


def build_model(cfg_model: Configs):
    """ Build model from configs.
    Args:
        cfg_model (Configs): Configs for model
    Returns:
        torch.nn.Module: Model.
    """
    try:
        model = MODELS[cfg_model.name](**parse_parameters(cfg_model))
    except KeyError:
        raise NotImplementedError(
            f'Model {cfg_model.name} is not implemented. ' +
            f'Do you mean "{most_similar(cfg_model.name, MODELS.keys())}"?')
    
    return model


def parse_parameters(cfg_model: Configs):
    """ Parse parameters for model.
    Args:
        cfg_model (Configs): Configs for model
    Returns:
        dict: parsed parameters
    """
    params = deepcopy(cfg_model.cfg)
    if not isinstance(params, dict):
        return {}
    else:
        name = params.pop('name', '')

        if getattr(cfg_model, 'prompts', None) and 'clip' in name:
            params['prompts'] = cfg_model.prompts

        if getattr(cfg_model, 'classnames', None) and 'clip' in name:
            params['classnames'] = cfg_model.classnames

        if getattr(cfg_model, 'num_classes', None):
            params['num_classes'] = cfg_model.num_classes

        return params
