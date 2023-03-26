
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl

from model import *
from model.misc import build_model
from utils.criterion import *

from copy import deepcopy
from matplotlib import cm



class VisionSequenceModel(pl.LightningModule):
    """
    General model framework for Vision-conditioned Sequence Generation.
    
    The `build_generators` method in the class will 
    build two models: one for forward, and one for inverse. 

    For invertable networks training, use the `InvertibleFWIModel`.
    """
    def __init__(self, config: dict):
        super(VisionSequenceModel, self).__init__()

        self.parse_config(config)
        self.build_models()
        self.build_train_metrics()
        self.build_eval_metrics()

    def parse_config(self, config: dict):
        # get model config
        model_config = config["model"]

        # Parse model config
        self.model_config = model_config
        self.name = config.get("name")
    
        # Parse training config
        self.training_config = config["training"]
        self.optimizer_config = self.training_config["optimizer"]
        self.scheduler_config = self.training_config.get("scheduler", {})

        # save the configs to the output folder
        self.save_hyperparameters()

    def build_models(self):
        # Define generator models
        self.model = build_model(
            model_name=self.model_config['class'],
            **self.model_config['params']
        )

    def build_train_metrics(self):
        # Build training metrics
        self.build_metrics(name="train_metrics")

    def build_eval_metrics(self):
        # Build evaluation metrics
        self.build_metrics(name="eval_metrics")

    def build_metrics(self, name: str):
        # Define global training metrics
        if name in self.training_config:
            metric_config = self.training_config[name]
            output_metrics = self.build_metrics_from_config(
                metric_config, 
            )
        else:
            output_metrics = {}

        # create a member variable with the same name as `name`
        setattr(self, name, output_metrics)
        
    def build_metrics_from_config(self, metric_config: dict):
        # Define metrics parsing
        metrics = {}
        for name, item in metric_config.items():
            metric = eval(item["class"])(
                **item.get("params", {})
            )
            metric = {
                "weight": float(item.get("weight", 1.0)),
                "metric": metric,
            }
            metrics[name] = metric
        return metrics

    def _make_optimizer(self, params, config):
        OPTIMIZER = eval(str(config["class"]))
        optimizer = OPTIMIZER(
            params, 
            **config.get("params", {})
        )
        return optimizer

    def _make_scheduler(self, optimizer, config):
        SCHEDULER = eval(str(config["class"]))
        scheduler = SCHEDULER(
            optimizer, 
            **config.get("params", {})
        )
        return scheduler

    def build_optimizers_and_schedulers(self, params, key=""):
        optimizers, schedulers = [], []
        # generators
        if key in self.optimizer_config:
            opt_config = self.optimizer_config[key]
        else:
            opt_config = self.optimizer_config
        if key in self.scheduler_config:
            sch_config = self.scheduler_config[key]
        else:
            sch_config = self.scheduler_config

        if opt_config != {}:  # shared optimizer
            gen_opt = self._make_optimizer(params, opt_config)
            optimizers.append(gen_opt)
            if sch_config != {}:
                gen_sch = self._make_scheduler(gen_opt, sch_config)
                schedulers.append(gen_sch)
        return optimizers, schedulers

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        # Create optimizers for generators
        optimizer_params = self.optimizer_config.get("params", {})
        lr = float(optimizer_params.get("lr", 2e-4))
        weight_decay = float(optimizer_params.get("weight_decay", 0.0))
        params = self.model.get_params_group(lr=lr, weight_decay=weight_decay)
        opt, sch = self.build_optimizers_and_schedulers(params)
        optimizers.extend(opt)
        schedulers.extend(sch)
        
        return optimizers, schedulers

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def calc_loss(self, pred, target, metrics: dict):
        # Get loss statistics
        stats = {}
        total_loss = 0
        for name, items in metrics.items():
            loss = items["metric"](pred, target)
            stats[name] = loss.item()
            weight = items.get("weight", 1.0)
            total_loss = total_loss + items["weight"] * loss
        stats["total_loss"] = total_loss.item()

        # This `loss` will be the loss used for backward
        stats["loss"] = total_loss
        return stats

    def training_step(self, batch, batch_idx):
        img, seq = batch
        pred = self(img, seq[:, :-1])
        losses = self.calc_loss(
            pred=pred, 
            target=seq, 
            metrics=self.train_metrics
        )
        return losses

    def on_train_epoch_end(self):
        # learning rate scheduler update
        self.lr_scheduler_step(epoch=self.current_epoch)
            
    def training_step_end(self, step_output):
        if step_output is not None:
            self.saved_output = step_output

        # log learning rate
        lr_schedulers = self.lr_schedulers()
        if isinstance(lr_schedulers, list) or isinstance(lr_schedulers, tuple):
            last_lr = lr_schedulers[0].get_last_lr()[0]
        else:
            last_lr = lr_schedulers.get_last_lr()[0]
        self.log('lr', last_lr, prog_bar=True)

        prefix = "train/"
        self.log_dict(
            {prefix + k: v for k, v in step_output.items()}, 
            prog_bar=False
        )
        return super().training_step_end(step_output)

    def eval_with_metrics(self, pred, target, metrics):
        # evaluate performance
        stats = {}
        for name, metric in metrics.items():
            func = metric["metric"]
            value = func(pred, target)
            stats[name] = value.item()
        return stats

    def validation_step(self, batch, batch_idx):
        img, seq = batch
        pred = self(img, seq[:, :-1])
        stats = self.calc_loss(
            pred=pred, 
            target=seq, 
            metrics=self.eval_metrics
        )
        prefix = "val/teacher_forcing/"
        self.log_dict(
            {prefix + k: v for k, v in stats.items()}, 
            prog_bar=False
        )

    def lr_scheduler_step(self, epoch):
        # Step learning rate schedulers
        lr_schedulers = self.lr_schedulers()
        if isinstance(lr_schedulers, list) or isinstance(lr_schedulers, tuple):
            for scheduler in lr_schedulers:
                if scheduler is not None:
                    scheduler.step(epoch)
        else:
            lr_schedulers.step(epoch)

    def log_image(self, name, tensor, step=None):
        # assume tensor is a torch.Tensor with shape (height, width)
        # convert to 3 channels (assuming input tensor is grayscale)
        assert tensor.ndim == 2, "Image logging only work for 2D PyTorch tensors."
        tensor = tensor.detach().cpu()
        # tensor = torch.stack([tensor, tensor, tensor], dim=0)

        # normalize to range [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # convert to numpy array and apply colormap
        img_array = tensor.squeeze().cpu().numpy()
        img_cmap = cm.get_cmap('viridis')
        img_colored_array = img_cmap(img_array)

        # log the image to Tensorboard
        if step is None:
            step = self.current_epoch
        self.logger.experiment.add_image(
            name, 
            img_colored_array, 
            dataformats="HWC",
            global_step=step,
        )
