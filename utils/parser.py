#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Argument parser functions."""

import argparse
import yaml
import sys
import os
from pathlib import Path


def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        cfg (str): settings of detection in yaml format.
        local_rank (int): for distributed training.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Mono3DVG Transformer Vision2 for Monocular 3D Visual Grounding"
    )
    parser.add_argument(
        "--config", 
        dest="cfg_file", 
        help="settings of detection in yaml format", 
        required=True,
    )
    parser.add_argument(
        "--opts",
        help="See config/*.yaml for all options, use : split multiple keys, e.g. KEY1:SUBKEY1 VALUE KEY2 VALUE",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def merge_dict(old: dict, new: dict):
    """
    Merge the new dict into the old dict recursively.
    Args:
        old (dict): old dict to be updated.
        new (dict): new dict to update.
    """
    for k, v in new.items():
        if isinstance(v, dict):
            old[k] = merge_dict(old.get(k, {}), v)
        else:
            old[k] = v
    return old


def split_value_from_dict(d: dict):
    splited_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if 'value' in v and (len(v) == 1 or len(v) == 2):
                splited_dict[k] = v['value']
            else:
                splited_dict[k] = split_value_from_dict(v)
        else:
            splited_dict[k] = v
    return splited_dict


def load_yaml_config(cfg: dict, config_path: Path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
        config = split_value_from_dict(config)
    # recursively load base config, only append undefined keys, not overwrite existing keys.
    if '__base__' in config:
        base_path = config.pop('__base__')
        cfg = merge_dict(cfg, config)
        if isinstance(base_path, (list, tuple)):
            for bp in base_path:
                cfg = load_yaml_config(cfg, config_path.parent / bp)
        else:
            cfg = load_yaml_config(cfg, config_path.parent / base_path)
    return merge_dict(cfg, config)


def update_config_from_opts(cfg: dict, opts: list, separator: str=':'):
    """
    Update the config from the command line opts.
    Args:
        cfg (dict): config dict.
        opts (list): list of options.
        separator (str): separator used in the parent-child keys.
    """
    for key, value in zip(opts[0::2], opts[1::2]):
        key_list = key.split(separator)
        temp = cfg
        while len(key_list) > 1:
            subkey = key_list.pop(0)
            if subkey in temp:
                temp = temp[subkey]
            else:
                temp = {}
        subkey = key_list.pop(0)
        if subkey in temp:
            if isinstance(temp[subkey], bool):
                temp[subkey] = yaml.safe_load(value)
            elif isinstance(temp[subkey], int):
                temp[subkey] = int(value)
            elif isinstance(temp[subkey], float):
                temp[subkey] = float(value)
            else:
                temp[subkey] = value
    return cfg


def dict_to_namespace(d: dict):
    """
    Convert a dict to a namespace.
    Args:
        d (dict): input dict.
    """
    n = argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(n, k, dict_to_namespace(v))
        else:
            setattr(n, k, v)
    return n


def namespace_to_dict(n: argparse.Namespace):
    """
    Convert a namespace to a dict.
    Args:
        n (argparse.Namespace): input namespace.
    """
    d = {}
    for k, v in n.__dict__.items():
        if isinstance(v, argparse.Namespace):
            d[k] = namespace_to_dict(v)
        else:
            d[k] = v
    return d


def load_config(args, path_to_config=None) -> argparse.Namespace:
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `local_rank`, `cfg_file` and `opts`.
    """
    cfg = {}
    # Load config from cfg.
    if path_to_config is not None:
        cfg = load_yaml_config(cfg, Path(path_to_config))
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg = update_config_from_opts(cfg, args.opts)
    cfg = dict_to_namespace(cfg)
    
    if cfg.dataset.name is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return cfg
