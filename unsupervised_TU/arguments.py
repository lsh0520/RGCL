import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='RD-GCL')
    parser.add_argument('--DS', dest='DS', type=str, default='MUTAG', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='hidden dimension')

    parser.add_argument('--aug', type=str, default='drop_ra')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--pooling', type=str, default='all')
    parser.add_argument('--log', type=str, default='full')

    return parser.parse_args()

