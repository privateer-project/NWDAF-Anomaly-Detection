import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from privateer_ad.config import PathsConf
from privateer_ad import logger

@dataclass
class DownloadConfig:
   zip_name: str = 'nwdaf-data.zip'
   url: str = os.environ.get('DATA_URL', 'https://zenodo.org/api/records/13900057/files-archive')
   extraction_dir: Path = Path(os.environ.get('RAW_DIR', PathsConf.raw))
   raw_dataset: Path = Path(os.environ.get('RAW_DATASET', PathsConf.raw_dataset))

class Downloader:
    def __init__(self, config: DownloadConfig):
        self.url: str = config.url
        self.extraction_dir: Path = config.extraction_dir
        self.zip_path: Path = self.extraction_dir.joinpath(config.zip_name)

    def download(self):
        logger.info(f'Downloading from {self.url} ...')
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f'Saving to {self.zip_path}...')
        os.makedirs(os.path.dirname(os.path.abspath(self.zip_path)), exist_ok=True)
        with open(self.zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    size = f.write(chunk)
                    pbar.update(size)

    def extract(self):
        os.makedirs(self.extraction_dir, exist_ok=True)
        # Extract the zip file
        logger.info(f'Extracting to {self.extraction_dir}...')
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            for file in tqdm(files, desc='Extracting'):
                zip_ref.extract(file, self.extraction_dir)
        logger.info('Download and extract completed!')

    def remove_zip(self):
        logger.warning(f'Removing {self.zip_path} ...')
        os.remove(self.zip_path)

    def download_extract(self):
        self.download()
        self.extract()
        self.remove_zip()

if __name__ == '__main__':
    downloader = Downloader(DownloadConfig(zip_name='nwdaf.zip'))
    downloader.download_extract()
