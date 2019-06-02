"""Contains the training options user can specify in command line."""
import argparse


def parse_options(args):
    """Parse training options user can specify in command line.

    Parameters
    ----------
    args : list of str
        Arguments to be parsed by this method. These include:

        * --retrain (boolean): If included, actor and critic neural network
          parameters are reset.
        * --test (boolean): If included, agent only uses greedy policy without
          noise. No changes are made to policy and neural networks. If not
          included, periods of training are by default interleaved with periods
          of testing to evaluate progress.
        * show (boolean): If included, training will be visualized.
        * --train_only (boolean): If included, agent will be solely in training
          mode and will not interleave periods of training and testing.
        * --verbose (boolean): If included, summary of each transition will be
          printed.
        * --layers (integer): number of levels in agent hierarchy
        * --time_scale (integer): max sequence length in which each policy will
          specialize

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--layers',
        type=int,
        default=1,
        help='number of levels in agent hierarchy'
    )

    parser.add_argument(
        '--time_scale',
        type=int,
        default=10,
        help='max sequence length in which each policy will specialize'
    )

    flags, _ = parser.parse_known_args(args)

    return flags
