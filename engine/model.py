
import os
import torch
from glob import glob
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from utils.model.graph_gpt import GraphGPT
from utils.criterion import UndirectedGraphLoss


class ModelModule(pl.LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model = GraphGPT(**model_config)
        self.criterion = UndirectedGraphLoss()
        self.model_config = model_config
        self.training_config = training_config

        # convert config dictionaries into member variables
        self.__dict__.update(training_config)
        self.__dict__.update(model_config)

        # load checkpoint if specified in model_config
        self.checkpoint_path = model_config.get("checkpoint")
        if self.checkpoint_path is not None:
            strict = model_config.get("strict")
            strict = True if strict is None else strict
            self.load_checkpoint(self.checkpoint_path, strict=strict)

        # correction model
        if "use_correction_model" not in model_config:
            self.use_correction_model = False

    def load_checkpoint(self, checkpoint_path, strict=True, verbose=True):
        if verbose:
            print(f"Loading from '{checkpoint_path}'.")
        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
            # PyTorch .pth or .pt file
            msg = self.model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"),
                strict=strict
            )
        elif checkpoint_path.endswith(".ckpt"):
            # PyTorch Lightning .ckpt file
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(
                state_dict,
                strict=strict
            )
        elif os.path.isdir(checkpoint_path):
            # Provided Path
            ckpt = glob(os.path.join(checkpoint_path, "*.ckpt"))
            pth = glob(os.path.join(checkpoint_path, "*.pth"))
            pt = glob(os.path.join(checkpoint_path, "*.pt"))
            if len(ckpt) > 0:
                checkpoint = torch.load(ckpt[0], map_location="cpu")
                state_dict = checkpoint["state_dict"]
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                msg = self.model.load_state_dict(
                    state_dict,
                    strict=strict
                )
            elif len(pth) > 0:
                msg = self.model.load_state_dict(
                    torch.load(pth[0], map_location="cpu"),
                    strict=strict
                )
            elif len(pt) > 0:
                msg = self.model.load_state_dict(
                    torch.load(pt[0], map_location="cpu"),
                    strict=strict
                )
            else:
                msg = f"Unsupported file extension for checkpoint path {checkpoint_path}."
        else:
            msg = f"Unsupported file extension for checkpoint {checkpoint_path}."
        if verbose:
            print(msg)  

    def forward(self, img, node_pair):
        return self.model(img, node_pair)

    def training_step(self, batch, batch_idx):
        img, node_pair = batch
        if self.use_correction_model:
            pred = self.model.iterative_forward(
                img,
                seq_len=node_pair.shape[1],
                stop_token_value=None,
                stop_threshold=None,
            )
            perturbed = node_pair[:, :-1]
            perturbed = perturbed + torch.randn_like(perturbed) * 0.01
            _, pred_tf = self(img, perturbed)
            pred = torch.cat([pred, pred_tf], dim=0)
            node_pair = torch.cat([node_pair, node_pair], dim=0)
            loss = self.criterion(pred, node_pair)
        else:
            pred = self(img, node_pair[:, :-1])
            loss = self.criterion(pred, node_pair)
        self.log('train_loss', loss)
        return loss
    
    def training_step_end(self, step_output):
        last_lr = self.lr_schedulers().get_last_lr()[0]
        self.log('lr', last_lr)
        return super().training_step_end(step_output)
    
    def training_epoch_end(self, outputs):
        reload_checkpoint = self.training_config.get("reload_checkpoint")
        if reload_checkpoint is not None:
            try:
                self.load_checkpoint(
                    reload_checkpoint, 
                    strict=False, 
                    verbose=False
                )
            except Exception as err:
                print(f"Checkpoint loading failed. Error:\n{err}")
        return super().training_epoch_end(outputs)
    
    def configure_optimizers(self):
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)
        params_group = self.model.get_params_group(
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(params_group)

        warmup_epochs = float(self.warmup_epochs)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=int(self.training_config["epochs"]),
            steps_per_epoch=int(self.training_config["steps_per_epoch"]),
            pct_start=warmup_epochs,
            anneal_strategy=self.anneal_strategy,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            verbose=False
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
