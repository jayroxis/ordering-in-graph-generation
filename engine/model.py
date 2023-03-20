import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from utils.model.graph_gpt import GraphGPT
from utils.criterion import LastTokenMatchLoss as GraphLoss


class ModelModule(pl.LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model = GraphGPT(**model_config)
        self.criterion = GraphLoss()
        self.model_config = model_config
        self.training_config = training_config

        # convert config dictionaries into member variables
        self.__dict__.update(training_config)
        self.__dict__.update(model_config)

        # load checkpoint if specified in model_config
        checkpoint_path = model_config.get("checkpoint")
        if checkpoint_path is not None:
            print(f"Loading from '{checkpoint_path}'.")
            if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
                print(self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu")
                ))
            elif checkpoint_path.endswith(".ckpt"):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                state_dict = checkpoint["state_dict"]
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                print(self.model.load_state_dict(
                    state_dict
                ))
            else:
                print("Unsupported file extension for checkpoint.")


    def forward(self, img, node_pair):
        return self.model(img, node_pair)

    def training_step(self, batch, batch_idx):
        img, node_pair = batch
        pred = self(img, node_pair[:, :-1])
        loss = self.criterion(pred, node_pair)
        self.log('train_loss', loss)
        return loss
    
    def training_step_end(self, step_output):
        last_lr = self.lr_schedulers().get_last_lr()[0]
        self.log('lr', last_lr)
        return super().training_step_end(step_output)

    def configure_optimizers(self):
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)
        params_group = self.model.get_params_group(
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(params_group)

        warmup_epochs = float(self.warmup_epochs)
        total_steps = self.training_config["total_steps"]

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_epochs,
            anneal_strategy=self.anneal_strategy,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            last_epoch=-1,
            verbose=False
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
