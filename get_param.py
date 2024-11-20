import argparse

def str2bool(v):
    """
    'boolean type variable' for add_argument
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

def params():
    """
    return parameters for training / testing / plotting of models
    :return: parameter-Namespace
    """
    parser = argparse.ArgumentParser(description='train / test a pytorch model for shallow water simulation')

    # Training parameters
    parser.add_argument('--net', default="WaterSurfaceUNet", type=str, help='network to train')
    parser.add_argument('--n_epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--n_batches_per_epoch', default=5000, type=int, help='number of batches per epoch (default: 5000)')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size (default: 100)')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size of network (default: 64)')
    parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
    parser.add_argument('--average_sequence_length', default=5000, type=int, help='average sequence length in dataset (default: 5000)')

    # Loss function weights
    parser.add_argument('--loss_mass', default=1.0, type=float, help='mass conservation loss weight')
    parser.add_argument('--loss_momentum', default=1.0, type=float, help='momentum conservation loss weight')
    parser.add_argument('--loss_energy', default=0.0, type=float, help='energy conservation loss weight (default: 0.0)')
    parser.add_argument('--loss_positive_depth', default=10.0, type=float, help='positive depth constraint weight')
    parser.add_argument('--loss_cfl', default=1.0, type=float, help='CFL condition loss weight')
    parser.add_argument('--loss_multiplier', default=1.0, type=float, help='multiply loss / gradients (default: 1.0)')
    parser.add_argument('--loss_boundary', default=1.0, type=float, help='boundary condition loss weight (default: 1.0)')

    # Physical parameters
    parser.add_argument('--H0', default=100.0, type=float, help='mean water depth (default: 100.0 m)')
    parser.add_argument('--g', default=9.81, type=float, help='gravitational acceleration (default: 9.81 m/s²)')
    parser.add_argument('--dt', default=1.0, type=float, help='timestep of simulation (default: 1.0 s)')

    # Domain parameters
    parser.add_argument('--width', default=300, type=int, help='domain width (default: 300)')
    parser.add_argument('--height', default=100, type=int, help='domain height (default: 100)')
    parser.add_argument('--dx', default=1.0, type=float, help='spatial step size (default: 1.0 m)')
    
    # Boundary condition parameters
    parser.add_argument('--boundary_type', default='periodic', type=str, choices=['periodic', 'rigid', 'outflow', 'reflecting'],
                       help='type of boundary conditions (default: rigid)')
    
    # Optimization parameters
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer (default: 0.0001)')
    parser.add_argument('--clip_grad_norm', default=1.0, type=float, help='gradient norm clipping (default: 1.0)')

    # System parameters
    parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
    parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training')
    parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')

    # Load parameters
    parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
    parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
    parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: False)')
    parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (default: False)')

    # parse parameters
    params = parser.parse_args()
    
    return params

def get_hyperparam(params):
    return f"net {params.net}; hs {params.hidden_size}; H0 {params.H0}; g {params.g}; dt {params.dt};"