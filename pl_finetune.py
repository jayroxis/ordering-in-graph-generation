import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging

from engine.arguments import ArgumentParserModule
from engine.data import DataModule
from engine.model import ModelModule
from pytorch_lightning import loggers as pl_loggers


def main():
    # Parse Arguments
    arg_parser = ArgumentParserModule()
    args = arg_parser.parse_args()

    # Convert args.gpu to a list if it is an integer
    gpus = args.gpu
    if isinstance(gpus, int):
        gpus = [gpus]

    # Load configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Recalculate batch size 
    data_config = config['data_config']
    data_config["batch_size"] = int(
        data_config["batch_size"] // len(gpus)
    )

    # Create Data Module
    data_module = DataModule(data_config)
    data_module.setup()

    # Set up training strategy
    training_config = config['training_config']
    epochs = int(training_config["epochs"])
    total_steps = (len(
        data_module.train_dataloader()
    ) + 1) * epochs / len(gpus)
    
    total_steps = int(total_steps)
    training_config["total_steps"] = total_steps

    # Create Model Module
    model_module = ModelModule(
        model_config=config['model_config'], 
        training_config=training_config,
    )
    model_module.use_correction_model = True
    model_module.model.use_correction_model = True
    model_module.model._init_correction_model()

    # Set up logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="runs/")

    # Set up trainer
    lr = float(training_config["lr"])
    swa_lrs = training_config.get('swa_lrs', lr)
    training_strategy = training_config.get('training_strategy')
    
    # 
    trainer = pl.Trainer(
        devices=gpus,
        accelerator='gpu',
        precision=32,
        strategy=training_strategy,
        max_epochs=training_config['epochs'],
        accumulate_grad_batches=training_config.get('grad_accum', 1),
        log_every_n_steps=training_config.get('log_interval', 1),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 1),
        enable_checkpointing=training_config.get("save_checkpoint", True),
        callbacks=[StochasticWeightAveraging(swa_lrs=swa_lrs)],
        logger=tb_logger,
    )

    # Train the model
    trainer.fit(
        model=model_module, 
        train_dataloaders=data_module.train_dataloader()
    )


if __name__ == "__main__":
    main()