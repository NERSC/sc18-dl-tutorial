"""
Main training script for the Deep Learning at Scale Keras examples.
"""

# System
import argparse
import logging

# Externals
import horovod.keras as hvd
import yaml

# Locals
from data import get_datasets
from models import get_model
from utils.device import configure_session
from utils.optimizers import get_optimizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def config_logging(verbose=False):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def init_workers(distributed=False):
    rank, n_ranks = 0, 1
    if distributed:
        hvd.init()
        rank, n_ranks = hvd.rank(), hvd.size()
    return rank, n_ranks

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
    return config

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    config_logging(args.verbose)
    logging.info('Initializing')
    rank, n_ranks = init_workers(args.distributed)
    if args.show_config:
        logging.info('Command line config: %s' % args)
    logging.info('Rank %i out of %i', rank, n_ranks)

    # Load configuration
    config = load_config(args.config)
    if rank == 0:
        logging.info('Configuration: %s' % config)
    train_config = config['training']

    # Configure session
    configure_session()

    # Load the data
    # TODO: move to data generators
    (x_train, y_train), (x_valid, y_valid) = get_datasets(**config['data'])
    logging.info('Training shape: %s, %s', x_train.shape, y_train.shape)
    logging.info('Validation shape: %s, %s', x_valid.shape, y_valid.shape)

    # Build the model
    model = get_model(**config['model'])
    # Configure optimizer
    opt = get_optimizer(n_ranks=n_ranks, distributed=args.distributed,
                        **config['optimizer'])
    # Compile the model
    model.compile(loss=train_config['loss'], optimizer=opt,
                  metrics=train_config['metrics'])
    if rank == 0:
        model.summary()

    # Prepare the training callbacks
    callbacks = []
    if args.distributed:
        callbacks += [
            # Broadcast initial variable states from rank 0 to all processes.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        ]

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=config['batch_size'],
                        epochs=config['n_epochs'],
                        validation_data=(x_valid, y_valid),
                        shuffle=True, verbose=2)

    # Drop to IPython interactive shell
    if args.interactive and (rank == 0):
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
