import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=1500, type=int)
parser.add_argument('--save_ckpt_freq', default=100, type=int)
# Model parameters
parser.add_argument('--rel_pos_bias', default=True, action='store_true')
parser.add_argument('--abs_pos_emb', default=False, action='store_true')
parser.add_argument('--num_mask_patches', default=75, type=int, help='number of the visual tokens/patches need be masked')
parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
parser.add_argument('--second_input_size', default=112, type=int, help='images input size for discrete vae')

parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
parser.add_argument('--patch_size', type=int, default=16)
# Optimizer parameters
parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the weight decay. We use a cosine schedule for WD. (Set the same value with args.weight_decay to keep weight decay no change)""")
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 5e-4)')
parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=100, metavar='N', help='epochs to warmup LR, if scheduler supports')
# Augmentation parameters
parser.add_argument('--train_interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--second_interpolation', type=str, default='lanczos', help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

# Dataset parameters
parser.add_argument('--data_path', default='/data/huxin/xjtuhx/projects/oneyear/D2VDemo/datasets_dir/images/', type=str, help='dataset path')
parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
parser.add_argument('--output_dir', default='/data/huxin/xjtuhx/projects/oneyear/D2VDemo/output_dir/', help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default='/data/huxin/xjtuhx/projects/oneyear/D2VDemo/log_dir/', help='path where to tensorboard log')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
parser.set_defaults(pin_mem=True)
# distributed training parameters
parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=True, type=bool)


opt = parser.parse_args()