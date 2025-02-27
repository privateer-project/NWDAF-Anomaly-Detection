import logging
import sys

def setup_logger():
    lgr = logging.getLogger('privateer')
    lgr.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)

    lgr.addHandler(console_handler)
    lgr.addHandler(file_handler)
    return lgr

logger = setup_logger()
