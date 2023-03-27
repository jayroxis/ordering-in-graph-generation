
import functools
from copy import deepcopy

from timm import create_model
import timm.models.registry as registry

import torch.nn as nn


def build_model(model_name: str, *args, **kwargs):
    """
    Safely build a model from timm registry.
    """
    if "model_name" in kwargs:
        raise ValueError(f"Got multiple model_name = {kwargs}.")
    # If given model_name not in timm registry, use default name
    if model_name not in registry._model_entrypoints:
        msg = f"`model_name`: \"{model_name}\" and " + \
                " is not in `timm` model registry.\n" + \
                f"Please make sure you have \"{model_name}\" " + \
                f"registed using in timm registry " + \
                "\"timm.models.registry.register_model\"."
        raise ValueError(msg)
    if "lr" in kwargs:
        lr = float(kwargs.pop("lr"))
    else:
        lr = None
    if "weight_decay" in kwargs:
        weight_decay = float(kwargs.pop("weight_decay"))
    else:
        weight_decay = None
    model = create_model(model_name, *args, **kwargs)
    if lr is not None:
        model.lr = lr
    if weight_decay is not None:
        model.weight_decay = weight_decay
    return model



def build_partial_class(config):
    """
    Build a partial class from configs.

    Args:
        config (dict): it should have a "class" key for class name.
            Apart from that, additional class arguments are passed
            in the "params" key which is optional.
    """

    assert "class" in config, "The input config should be a key \"class\"."
    params = config.get("params", {})
    if config["class"] in registry._model_entrypoints:
        model_class = functools.partial(build_model, config["class"])
    else:
        model_class = eval(config["class"])
    partial_class = functools.partial(model_class, **params)
    return partial_class


def build_module_registry(config: dict, default_cfg: dict = {}):
    """
    Build module registry from a config dictionary.
    Note: it works for Python >= 3.8

    Args:
        config (dict): it should have the following structure:
        ```
        config = {
            "module1": {
                "class1": ModuleClass1,
                "params1": {
                    "param1": ***,
                    "param2": ***,
                    ...
                },
            },
            "module2": {
                "class2": ModuleClass2,
                "params2": {
                    "param1": ***,
                    ...
                },
            },
        }
        ```
        The `params` can be left empty of partially given.

    Returns:
        It returns the module registry which contains a dict
        of partial classes.
    """
    # module registry
    module_registry = {}
    model_cfg = deepcopy(default_cfg)
    model_cfg.update(config)
    for name, cfg in model_cfg.items():
        module_class = build_partial_class(cfg)
        module_registry[name] = module_class
    return module_registry


def get_params_group(model, lr, weight_decay=0.0, **kwargs):
    """
    ! DANGER: do not call this function in a member function 
    """
    # check whether the model has any trainable parameters
    has_grad = False
    if hasattr(model, "parameters"):
        if callable(getattr(model, "parameters")):
            for param in model.parameters():
                if param.requires_grad:
                    has_grad = True
                    break
    if not has_grad:
        return []
    
    # Get visual encoder parameters group
    if hasattr(model, "lr"):
        lr = model.lr
    if hasattr(model, "weight_decay"):
        weight_decay = model.weight_decay

    lr = float(lr)
    weight_decay = float(weight_decay)
    
    if hasattr(model, "get_params_group"):
        params_group = model.get_params_group(
            lr=lr, 
            weight_decay=weight_decay, 
            **kwargs
        )
    else:
        params_group = [{
            "params": model.parameters(), 
            "lr": lr, 
            "weight_decay": weight_decay
        }]
    return params_group