"""
Dataloader
version: 0.0.3
update: 2024-04-19
"""
from torch.utils.data import DataLoader

from utils import Configs, Register, get_logger, most_similar
from .transform import build_transform

DATASETS = Register()


def build_dataloader(cfg_dst: Configs, loader: Configs):
    """ Build dataloader.
    Args:
        cfg_dst (Configs) : dataset config
        loader (Configs) : loader config
    Returns:
        torch.utils.data.DataLoader: dataloader
    """
    try:
        dataset = DATASETS[cfg_dst.name](**parse_parameters(cfg_dst))
    except KeyError:
        raise NotImplementedError(
            f'Dataset {cfg_dst.name} is not implemented. ' + 
            f'Do you mean "{most_similar(cfg_dst.name, DATASETS.keys())}"?')
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=loader.batch_size or 1,
        num_workers=loader.num_workers or 0,
        shuffle=loader.shuffle or None,
        drop_last=loader.drop_last or False,
        pin_memory=loader.pin_memory or False,
    )
    
    return dataloader


def parse_parameters(cfg_dst: Configs) -> dict:
    """ Parse parameters for dataset.
    Args:
        cfg_dst (Configs): Configs for dataset
    Returns:
        dict: parsed parameters
    """
    params = cfg_dst.cfg
    if not isinstance(params, dict):
        return {}
    else:
        if 'name' in params:
            params.pop('name')

        if 'transform' in params:
            params['transform'] = build_transform(cfg_dst.transform)
            get_logger().info(f"Creating transform for {cfg_dst._name}:\n{params['transform']}")

        return params
