import os
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.WARN, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLANCK_URLS = [
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_353_2048_R3.01_full.fits"
]


class Downloader:
    """General-purpose file downloader using wget."""

    def __init__(self, destination: str | Path):
        """
        Args:
            destination: Directory where downloaded files will be saved.
        """
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=True)

    def download(self, urls: list[str]) -> None:
        """Download a list of files from the given URLs.

        Args:
            urls: List of URLs to download.
        """
        for url in tqdm(urls, desc="Downloading files", unit="file"):
            self._download_file(url)

    def _download_file(self, url: str) -> None:
        """Download a single file, skipping it if already present.

        Args:
            url: URL of the file to download.
        """
        filename = self.destination / Path(url).name

        if filename.exists():
            logger.info("Skipping (already exists): %s", filename)
            return

        logger.info("Downloading: %s → %s", url, self.destination)
        result = subprocess.run(
            ["wget", url, "-P", str(self.destination), "--progress=bar:force"],
            text=True
        )

        if result.returncode != 0:
            logger.error("Failed to download %s", url)
        else:
            logger.info("Saved: %s", filename)

dataFolder = Path(__file__).parent

if __name__ == "__main__":
    
    Downloader(dataFolder).download(PLANCK_URLS)