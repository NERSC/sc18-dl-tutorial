"""
Main training script for the Deep Learning at Scale Keras examples.
"""

# System
import argparse
import logging

# Externals
import yaml

# Locals
from data import get_datasets
from models import get_model
from utils.device import configure_session

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

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    config_logging(args.verbose)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f)
    #if not args.distributed or (dist.get_rank() == 0):
    logging.info('Configuration: %s' % config)

    # Configure session
    configure_session()

    # Load the data
    # TODO: move to data generators
    (x_train, y_train), (x_valid, y_valid) = get_datasets(**config['data'])
    logging.info('Training shape: %s, %s', x_train.shape, y_train.shape)
    logging.info('Validation shape: %s, %s', x_valid.shape, y_valid.shape)

    # Build the model
    model = get_model(**config['model'])
    model.summary()

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=config['batch_size'],
                        epochs=config['n_epochs'],
                        validation_data=(x_valid, y_valid),
                        shuffle=True, verbose=2)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
