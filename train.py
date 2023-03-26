
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# This line is very IMPORTANT for expected Cluster behavior
#       Note: if you want to use Slurm, use other environments.
from pytorch_lightning.plugins.environments import LightningEnvironment

# Load PyTorch Lightning training
from engine.data import *
from engine.arguments import *
from engine.model import *


def main():
    # Parse Arguments
    arg_parser = ArgumentParserModule()
    args = arg_parser.parse_args()

    # Load configuration file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
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
    trainer = pl.Trainer(
        devices=args.gpu,
        logger=tb_logger,
        plugins=[LightningEnvironment()],
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