import os
import yaml
import argparse
from fastprogress.fastprogress import master_bar, progress_bar

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from utils.data.dataset import *
from utils.data.misc import PadSequence, shuffle_node_pair
from utils.model.misc import *
from utils.model.graph_gpt import GraphGPT
from utils.model.discriminator import ConstractiveDiscriminator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train GraphGPT')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='ID of the GPU to use (default: CPU)')
    args = parser.parse_args()
    return args


def get_dataloader(data_config):
    dataset = RenderedPlanarGraphDataset(**data_config)
    pad_value = data_config['pad_value']
    dataloader = DataLoader(
        dataset,
        collate_fn=PadSequence(pad_value),
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    return dataloader


def get_model(model_config):
    model = GraphGPT(
        **model_config
    )
    _ = count_parameters(model)
    return model


def save_checkpoint(model, logging_config, name):
    checkpoint_file = logging_config['checkpoint_file'].format(epoch=name)
    checkpoint_file = os.path.join(logging_directory, checkpoint_file)
    torch.save(model.state_dict(), checkpoint_file)
    return None


# Parse Arguments
args = parse_arguments()


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


# create SummaryWriter object for logging
if logging_config['log_to_tensorboard']:
    writer = SummaryWriter()
    logging_directory = writer.get_logdir()
else:
    exp_name = os.path.basename(args.config)
    exp_name = exp_name.split(".")[0]
    logging_directory = os.path.join("logs", exp_name)


# Create Dataloader
dataloader = get_dataloader(data_config)

# create models
model = get_model(model_config)
model = model.to(device)

# create the expontial moving average of the model
ema_decay = training_config.get("ema_decay")
if ema_decay is not None and ema_decay > 0:
    from torch_ema import ExponentialMovingAverage
    use_ema = True
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
else:
    use_ema = False

# create the discriminator
input_dim = model_config['gpt_output_size']
discriminator = ConstractiveDiscriminator(input_dim=input_dim)
discriminator = discriminator.to(device)


# optimizers and 
lr = float(training_config['lr'])
weight_decay = float(training_config['weight_decay'])
params_group = model.get_params_group(
    lr=lr, 
    weight_decay=weight_decay,
)
optimizer = torch.optim.AdamW(params_group)

disc_lr = float(training_config['disc_lr'])
disc_weight_decay = float(training_config['disc_weight_decay'])
disc_optimizer = torch.optim.AdamW(
    discriminator.parameters(),
    lr=disc_lr,
    weight_decay=disc_weight_decay,
)

# create scheduler
warmup_epochs = float(training_config['warmup_epochs'])
epochs = int(training_config['epochs'])
total_steps = len(dataloader) * epochs + 1
warmup_steps = int(total_steps * warmup_epochs)

scheduler = OneCycleLR(
    optimizer, 
    max_lr=lr, 
    total_steps=total_steps, 
    pct_start=warmup_epochs, 
    anneal_strategy=training_config['anneal_strategy'], 
    div_factor=training_config['div_factor'], 
    final_div_factor=training_config['final_div_factor'], 
    last_epoch=-1, 
    verbose=False
)


# discriminator gradient accumulation
disc_grad_accum = training_config.get('disc_grad_accum')
if disc_grad_accum is None:
    disc_grad_accum = 1

# create master progress bar
mb = master_bar(range(epochs))

# Start Training
for epoch in mb:
    model.train()
    loss_accum = 0.0
    d_loss_accum = 0.0
    
    progress = progress_bar(dataloader, parent=mb)
    for i, (img, node_pair) in enumerate(progress):
        
        img, node_pair = img.to(device), node_pair.to(device)
        
        # forward pass
        pad = torch.ones_like(node_pair)[:, :1] * data_config['pad_value']
        node_pair = torch.cat([node_pair, pad], dim=1)
        pred = model(img, node_pair[:, :-1])
        real_sequence = node_pair

        """ Discriminator """
        disc_optimizer.zero_grad()

        generated_sequence = pred.detach()

        # pass the generated vs real to the discriminator
        generated_scores = discriminator(
            seq_1=generated_sequence,
            seq_2=real_sequence
        )

        # Shuffle the real sequences to reorder the tokens
        shuffled_real_sequence = shuffle_node_pair(real_sequence)

        # pass the real sequences to the discriminator
        real_scores = discriminator(
            seq_1=shuffled_real_sequence,
            seq_2=real_sequence
        )

        # compute discriminator loss
        d_loss = torch.mean(generated_scores) - torch.mean(real_scores)

        # optimize discriminator
        d_loss_accum += d_loss.item()
        d_loss.backward()
        
        if i % min(disc_grad_accum, epoch + 1) == 0:
            disc_optimizer.step()

        """ Generator """
        optimizer.zero_grad()

        # pass the generated sequences to the discriminator again
        generated_sequence = pred
        generated_scores = discriminator(
            seq_1=generated_sequence,
            seq_2=real_sequence
        )

        # compute generator loss
        g_loss = -torch.mean(generated_scores)

        # optimize generator
        g_loss.backward()
        optimizer.step()

        # update scheduler
        scheduler.step()
        
        # update ema model
        if use_ema:
            ema.update()

        """ Statistics """

        # accumulate loss
        loss_accum += g_loss.item()

        # get last learning rate
        last_lr = scheduler.get_last_lr()[0]

        # print progress
        progress.comment = (
            f"Epoch {epoch+1}/{epochs}, "
            f"Gen Loss: {g_loss.item():.4f}, "
            f"Disc Loss: {d_loss.item():.4f}, "
            f"Avg Gen Loss: {loss_accum / (i + 1):.4f}, "
            f"Avg Disc Loss: {d_loss_accum / (i + 1):.4f}, "
            f"LR: {last_lr:.6f}"
        )
        progress.update(i+1)

    # calculate average loss
    avg_loss = loss_accum / len(dataloader)
    avg_d_loss = d_loss_accum / len(dataloader)

    # construct message
    msg = []
    msg.append("Epoch {}/{}".format(epoch+1, epochs))
    msg.append("LR: {:.6f}".format(lr))
    msg.append("Avg Gen Loss: {:.4f}".format(avg_loss))
    msg.append("Avg Disc Loss: {:.4f}".format(avg_d_loss))
    
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
                    f"Avg Gen Loss: {avg_loss:.4f}, "
                    f"Avg Disc Loss: {d_loss.item():.4f}\n")
    
    # log to tensorboard
    if logging_config['log_to_tensorboard']:
        writer.add_scalar("Loss/train_gen", avg_loss, epoch+1)
        writer.add_scalar("Loss/train_disc", d_loss.item(), epoch+1)
    
    # Save model checkpoint
    if logging_config['save_checkpoint'] and epoch % logging_config['save_interval'] == 0:
        save_checkpoint(model, logging_config, name=epoch+1)

# Save final checkpoint
if logging_config['save_checkpoint']:        
    save_checkpoint(model, logging_config, name="final")

# close SummaryWriter
if logging_config['log_to_tensorboard']:
    writer.close()