import os
import yaml
import argparse
from pprint import pprint
from fastprogress.fastprogress import master_bar, progress_bar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from utils.data.dataset import *
from utils.data.misc import PadSequence
from utils.data.planar_graph import PlanarGraph
from utils.model.misc import *
from utils.model.graph_gpt import GraphGPT
from utils.helpers import format_time
from utils.criterion import UndirectedGraphLoss


# Parse Arguments
parser = argparse.ArgumentParser(description='Train GraphGPT')
parser.add_argument('--config', type=str, default='configs/default.yaml',
                    help='Path to configuration file (default: config.yaml)')
parser.add_argument('--gpu', type=int, default=None,
                    help='ID of the GPU to use (default: CPU)')
args = parser.parse_args()


# Set device for PyTorch computations
if args.gpu is not None and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")


# Load configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
# print configuration
print("---------------- Configs -----------------")
print(yaml.dump(config, indent=4))

    
# Extract configuration parameters
data_config = config['data_config']
model_config = config['model_config']
training_config = config['training_config']
logging_config = config['logging_config']
checkpoint_config = config['checkpoint_config']

    
# create SummaryWriter object for logging
if logging_config['log_to_tensorboard']:
    writer = SummaryWriter()
    logging_directory = writer.get_logdir()
else:
    exp_name = os.path.basename(args.config)
    exp_name = exp_name.split(".")[0]
    logging_directory = os.path.join("logs", exp_name)


# Create Dataset and Dataloader
pad_value = data_config['pad_value']

dataset = RenderedPlanarGraphDataset(
    num_samples=data_config['num_samples'], 
    img_size=data_config['img_size'],
    num_points=data_config['num_points'],
    epsilon=data_config['epsilon'],
    tiny_angle=data_config['tiny_angle'],
)

dataloader = DataLoader(
    dataset, 
    batch_size=data_config['batch_size'], 
    shuffle=True, 
    collate_fn=PadSequence(pad_value),
    num_workers=data_config['num_workers']
)


# create models
embed_dim = model_config['embed_dim']
vis_enc_name = model_config['vis_enc_name']
gpt_output_size = model_config['gpt_output_size']
gpt_d_model = model_config['gpt_d_model']
gpt_num_layers = model_config['gpt_num_layers']

model = GraphGPT(
    vis_enc_name=vis_enc_name, 
    image_size=data_config['img_size'], 
    embed_dim=embed_dim, 
    max_freq=model_config['max_freq'], 
    gpt_output_size=gpt_output_size, 
    gpt_d_model=gpt_d_model, 
    gpt_num_layers=gpt_num_layers,
)
num_params = count_parameters(model)
model = model.to(device)

# create the expontial moving average of the model
ema_decay = training_config.get("ema_decay")
if ema_decay is not None and ema_decay > 0:
    from torch_ema import ExponentialMovingAverage
    use_ema = True
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
else:
    use_ema = False
    
# optimizers and criterions
lr = float(training_config['lr'])
weight_decay = float(training_config['weight_decay'])
params_group = model.get_params_group(
    lr=lr, 
    weight_decay=weight_decay,
)
optimizer = torch.optim.AdamW(params_group)
criterion = UndirectedGraphLoss()


# create scheduler
warmup_epochs = float(training_config['warmup_epochs'])
epochs = int(training_config['epochs'])
total_steps = len(dataloader) * epochs + 1
warmup_steps = int(total_steps * warmup_epochs)

scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, 
                       pct_start=warmup_epochs, anneal_strategy='cos', 
                       div_factor=25, final_div_factor=1000, 
                       last_epoch=-1, verbose=False)


# create master progress bar
mb = master_bar(range(epochs))

# Start Training
for epoch in mb:
    model.train()
    loss_accum = 0.0
    
    progress = progress_bar(dataloader, parent=mb)
    for i, (img, node_pair) in enumerate(progress):
        
        img, node_pair = img.to(device), node_pair.to(device)
        
        optimizer.zero_grad()
        
        # forward pass
        pad = torch.ones_like(node_pair)[:, :1] * pad_value
        node_pair = torch.cat([node_pair, pad], dim=1)
        pred = model(img, node_pair[:, :-1])
        target = node_pair

        # calculate loss
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        # accumulate loss
        loss_accum += loss.item()

        # update scheduler
        scheduler.step()
        
        # update ema model
        if use_ema:
            ema.update()

        # get last learning rate
        last_lr = scheduler.get_last_lr()[0]

        # print progress
        progress.comment = (
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {loss.item():.4f}, "
            f"Avg Loss: {loss_accum / (i + 1):.4f}, "
            f"LR: {last_lr:.6f}"
        )
        progress.update(i+1)

    # calculate average loss
    avg_loss = loss_accum / len(dataloader)
    
    # construct message
    msg = []
    msg.append("Epoch {}/{}".format(epoch+1, epochs))
    msg.append("LR: {:.6f}".format(lr))
    msg.append("Loss: {:.4f}".format(avg_loss))
    
    # join message lines
    msg = " | ".join(msg)

    # update progress bar
    mb.child.comment = msg
    
    # Log metrics
    if logging_config['log_to_file']:
        log_file = logging_config['log_file']
        log_file = os.path.join(logging_directory, log_file)
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}, "
                    f"Avg Loss: {avg_loss:.4f}\n")
    
    # log to tensorboard
    if logging_config['log_to_tensorboard']:
        writer.add_scalar("Loss/train", avg_loss, epoch+1)
    
    # Save model checkpoint
    if checkpoint_config['save_checkpoint'] and epoch % checkpoint_config['save_interval'] == 0:
        checkpoint_file = checkpoint_config['checkpoint_file'].format(epoch=epoch+1)
        checkpoint_file = os.path.join(logging_directory, checkpoint_file)
        torch.save(model.state_dict(), checkpoint_file)

# Save final checkpoint        
checkpoint_file = checkpoint_config['checkpoint_file'].format(epoch="final")
checkpoint_file = os.path.join(logging_directory, checkpoint_file)
torch.save(model.state_dict(), checkpoint_file)

# close SummaryWriter
if logging_config['log_to_tensorboard']:
    writer.close()