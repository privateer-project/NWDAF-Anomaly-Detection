import logging
import zipfile

import requests
from tqdm import tqdm

from privateer_ad.config import PathConfig


class Downloader:
    def __init__(self):
        self.paths = PathConfig()

    def download(self):
        logging.info(f'Downloading from {self.paths.data_url} ...')
        response = requests.get(self.paths.data_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        logging.info(f'Saving to {self.paths.zip_file}...')
        self.paths.zip_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths.zip_file, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    size = f.write(chunk)
                    pbar.update(size)

    def extract(self):
        self.paths.raw_dir.mkdir(exist_ok=True)
        # Extract the zip file
        logging.info(f'Extracting to {self.paths.raw_dir}...')
        with zipfile.ZipFile(self.paths.zip_file, 'r') as zip_ref:
            files = zip_ref.namelist()
            for file in tqdm(files, desc='Extracting'):
                zip_ref.extract(file, self.paths.raw_dir)
        logging.info('Download and extract completed!')
        logging.warning(f'Removing {self.paths.zip_file} ...')
        self.paths.zip_file.unlink(missing_ok=True)

    def download_extract(self):
        self.download()
        self.extract()

if __name__ == '__main__':
    downloader = Downloader()
    downloader.download_extract()
