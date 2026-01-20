import argparse
import sys

def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Generate the MEAGraph Autoencoder model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file name',
        required=True,
        type=str
    )
    parser.add_argument(
    '--rate',
    dest='rate',
    help='self-defined pooling rate for edge_reduction in Encoder layers',
    default=-1,
    type=float
    )
    parser.add_argument(
    '--fixed_rate_l',
    dest='fixed_rate_l',
    help='Lower rate for edge selection of build_graph function in utils.py',
    default=0.9-1e-5,
    type=float
    )
    parser.add_argument(
    '--fixed_rate_r',
    dest='fixed_rate_r',
    help='Upper rate for edge selection of build_graph function in utils.py',
    default=1+1e-5,
    type=float
    )
    parser.add_argument(
    '--device',
    dest='device',
    help='device',
    default=None,
    type=str
    )
    parser.add_argument(
    '--train_val_ratio',
    dest='train_val_ratio',
    help='ratio of training data to test data for the potetnial fitting using inference.py',
    default=0.8,
    type=float
    )
    parser.add_argument(
    '--group_name',
    dest='group_name',
    help='names of structure type',
    default=None,
    type=str
    )
    parser.add_argument(
    'opts',
    help='See MEAG_VAE/config.py for all options',
    default=None,
    nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()