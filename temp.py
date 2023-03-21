import argparse
parser = argparse.ArgumentParser(description='FCN Training')
parser.add_argument('-d', '--device', default='cuda', help='device')
parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
parser.add_argument('-fs', '--file-size', default=None, type=str, help='number of samples in each npy file')

# Path related
parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
parser.add_argument('-t', '--train-anno', default='flatfault_b_train_invnet.txt', help='name of train anno')
parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')
parser.add_argument('-pd', '--plot_directory', type=str, default='visualisation', help='directory to save intermediate model results')

# Model related
parser.add_argument('-m', '--model', type=str, help='inverse model name')
parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
# Training related
parser.add_argument('-pi', '--plot_interval', default=10, type=int, help='Training results save frequency')
parser.add_argument('--num_images', default=10, type=int, help='plotting 10 random images')
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4 , type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')   
parser.add_argument('-eb', '--epoch_block', type=int, default=40, help='epochs in a saved block')
parser.add_argument('-nb', '--num_block', type=int, default=3, help='number of saved block')
parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--k', default=1, type=float, help='k in log transformation')
parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')

# Loss related
parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
parser.add_argument('-p1v', '--lambda_p1v', type=float, default=0.1)
parser.add_argument('-p2v', '--lambda_p2v', type=float, default=0.1)
parser.add_argument('-reg', '--lambda_reg', type=float, default=0.1)

# Distributed training related
parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

# Tensorboard related
parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

args = parser.parse_args()



class ArgumentParser:
    def __init__(self):
        # Call member functions to initialize member variables
        self.init_device_args()
        self.init_dataset_args()
        self.init_file_size_args()
        self.init_path_args()
        self.init_output_args()
        self.init_log_args()
        self.init_save_name_args()
        self.init_suffix_args()
        self.init_plot_directory_args()
        self.init_model_args()
        self.init_training_args()
        self.init_loss_args()
        self.init_distributed_training_args()
        self.init_tensorboard_args()

    def init_device_args(self):
        self.device = 'cuda'

    def init_dataset_args(self):
        self.dataset = 'flatfault-b'

    def init_file_size_args(self):
        self.file_size = None

    def init_path_args(self):
        self.anno_path = 'split_files'
        self.train_anno = 'flatfault_b_train_invnet.txt'
        self.val_anno = 'flatfault_b_val_invnet.txt'

    def init_output_args(self):
        self.output_path = 'Invnet_models'

    def init_log_args(self):
        self.log_path = 'Invnet_models'

    def init_save_name_args(self):
        self.save_name = 'fcn_l1loss_ffb'

    def init_suffix_args(self):
        self.suffix = None

    def init_plot_directory_args(self):
        self.plot_directory = 'visualisation'

    def init_model_args(self):
        self.model = None
        self.up_mode = None
        self.sample_spatial = 1.0
        self.sample_temporal = 1

    def init_training_args(self):
        self.plot_interval = 10
        self.num_images = 10
        self.batch_size = 256
        self.lr = 0.0001
        self.lr_milestones = []
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_gamma = 0.1
        self.lr_warmup_epochs = 0
        self.epoch_block = 40
        self.num_block = 3
        self.workers = 16
        self.k = 1
        self.print_freq = 50
        self.resume = None
        self.start_epoch = 0

    def init_loss_args(self):
        self.lambda_g1v = 1.0
        self.lambda_g2v = 1.0
        self.lambda_p1v = 0.1
        self.lambda_p2v = 0.1
        self.lambda_reg = 0.1

    def init_distributed_training_args(self):
        self.sync_bn = False
        self.world_size = 1
        self.dist_url = 'env://'

    def init_tensorboard_args(self):
        self.tensorboard = False

