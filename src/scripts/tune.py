import argparse
from src.training import ModelTuner


def main():
    parser = argparse.ArgumentParser(description='Autotune LSTM model')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials for optimization')
    parser.add_argument('--timeout', type=int, default=3600 * 8,
                        help='Timeout in seconds')
    args = parser.parse_args()

    tuner = ModelTuner(
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    tuner.tune()

if __name__ == '__main__':
    main()
