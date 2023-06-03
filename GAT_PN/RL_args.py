import os
import time
import argparse
import torch

def get_arguments(args = None):
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of GAT-PN model trained with RL")
    
    # Data
    parser.add_argument('--graph_size', type=int, default=20, help="Number of nodes in TSP graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=12800, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', type=int, default=3, help='Number of process block iterations in Encoder')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--vector_context', default=0)
    parser.add_argument('--graph_encoding', default='GAT')

    # Training
    parser.add_argument('--training_steps', type=int, default=2500, help='Training steps per epoch')
    parser.add_argument('--lr_model', type=float, default=1e-3, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-3, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.96, help='Learning rate decay per epoch')
    parser.add_argument('--lr_decay_steps', type=int, default=2500)
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default=None,
                        help="Baseline to use: 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    # Other
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    args_rl = parser.parse_args(args)
    args_rl.use_cuda = 0# torch.cuda.is_available() and not args_rl.no_cuda
    args_rl.device = torch.device("cuda:0" if args_rl.use_cuda else "cpu")
    args_rl.run_name = f"{args_rl.run_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    args_rl.save_dir = os.path.join(args_rl.output_dir, 
                                    f"_{args_rl.graph_size}_{args_rl.batch_size}",
                                    args_rl.run_name)
    assert args_rl.baseline in [None, 'critic', 'exponential'], "Invalid baseline {}".format(args_rl.baseline)
    assert args_rl.epoch_size % args_rl.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return args_rl




