
import functools
from copy import deepcopy

from timm import create_model
import timm.models.registry as registry

import torch.nn as nn


def build_model(model_name: str, default_model: str = "", *args, **kwargs):
    """
    Safely build a model from timm registry.
    """
    # If given model_name not in timm registry, use default name
    if model_name not in registry._model_entrypoints:
        if default_model not in registry._model_entrypoints:
            msg = f"`model_name`: \"{model_name}\" and " + \
                  f"`default_model`: \"{default_model}\"" + \
                   " are both not `timm` model registry.\n" + \
                  f"Please make sure you have \"{model_name}\" " + \
                  f"and \"{default_model}\" registed using " + \
                   "\"timm.models.registry.register_model\"."
            raise ValueError(msg)
        else:
            model_name = default_model

    model_name = create_model(model_name, *args, **kwargs)
    return model_name



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
        partial_class = functools.partial(
            build_model, 
            model_name=config["class"], 
            **params
        )
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