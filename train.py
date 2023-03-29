
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This line is very IMPORTANT for expected Cluster behavior
#       Note: if you want to use Slurm, use other environments.
from pytorch_lightning.plugins.environments import LightningEnvironment


# Load PyTorch Lightning training
from engine.data import *
from engine.arguments import *
from engine.model import *

from utils.helpers import safe_load_yaml



def main():
    # set cudnn to False since we have data of different length
    torch.backends.cudnn.benchmark=False

    # Parse Arguments
    arg_parser = ArgumentParserModule()
    args = arg_parser.parse_args()

    # Load configuration file
    configs = safe_load_yaml(args.config)
    data_config = configs["data"]
    train_config = configs["training"]

    # Create model from configs
    model = eval(configs["class"])(config=configs)
    
    # Get data module
    data = eval(data_config["class"])(config=data_config)

    # Set up logger
    save_dir = train_config.get("save_dir", "./")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tb_logger = TensorBoardLogger(save_dir=save_dir)

    # Create PyTorch Lightning Trainer
    refresh_rate = train_config["params"].get("progressbar_refresh_rate", 100)
    trainer = pl.Trainer(
        devices=args.gpu,
        logger=tb_logger,
        plugins=[LightningEnvironment()],
        callbacks=[TQDMProgressBar(refresh_rate=refresh_rate)],
        **train_config["params"]
    )

    # Train the model
    trainer.fit(
        model, 
        train_dataloaders=data.train_loader, 
        val_dataloaders=data.val_loader
    )



if __name__ == "__main__":

    main()