import logging
import zipfile

import requests
from tqdm import tqdm

from privateer_ad.config import PathConfig


class Downloader:
    """
    Handles downloading and extracting dataset archives for the PRIVATEER project.
    """
    def __init__(self):
        self.paths = PathConfig()

    def download(self):
        """
        Download the dataset archive from the configured remote URL.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request returns an error status
            requests.exceptions.RequestException: For network-related errors
            OSError: If local file operations fail (disk space, permissions, etc.)
        """

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
        """
        Extract the downloaded archive to the configured raw data directory.

        Processes the downloaded zip file and extracts all contents to the raw data
        directory. The extraction maintains the original directory structure from
        the archive while providing progress feedback for each file being extracted.

        Raises:
            zipfile.BadZipFile: If the downloaded file is corrupted or not a valid zip
            OSError: If extraction fails due to filesystem issues
        """
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
        """
        Convenience method that performs both download and extraction in sequence.

        This method combines the download and extract operations into a single call,
        which is the most common usage pattern. It ensures that both operations
        complete successfully before returning, providing a simple interface for
        the complete dataset preparation workflow.
        """
        self.download()
        self.extract()

if __name__ == '__main__':
    downloader = Downloader()
    downloader.download_extract()
