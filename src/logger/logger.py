import logging
import sys


def setup_logger(write_to_file=False):
    # Check if the root logger is already configured
    if not logging.root.handlers:
        handlers = [logging.StreamHandler(sys.stdout)]
        if write_to_file:
            handlers.append(logging.FileHandler("local_artifacts/logs.log", mode="w"))

        # Configure the root logger only if it's not already configured
        logging.basicConfig(
            format="%(asctime)s: %(levelname)s: %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )

    # Create and return a logger instance
    return logging.getLogger(__name__)
