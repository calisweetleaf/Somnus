import os
import hashlib
import requests
import concurrent.futures
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import logging
from urllib.parse import urlparse
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DownloadSystem")

class DownloadError(Exception):
    """Custom exception for download-related errors."""
    pass

class FileDownloader:
    def __init__(self):
        self.session = requests.Session()

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")

    def _verify_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity using hash"""
        return self.verify_file_hash(str(file_path), expected_hash)

    def verify_file_hash(self, file_path: str, expected_hash: str, hash_type: str = 'sha256') -> bool:
        """Verify downloaded file integrity using hash."""
        hash_func = getattr(hashlib, hash_type)()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest() == expected_hash

    def download_file(
            self,
            url: str,
            save_path: str,
            expected_hash: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: int = 30,
            chunk_size: int = 8192
        ) -> Tuple[bool, str]:
            """Enhanced secure download implementation."""
            try:
                self._validate_url(url)
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                response = self.session.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=timeout
                )
                response.raise_for_status()
                
                file_size = int(response.headers.get('content-length', 0))
                mode = 'ab' if save_path.exists() and 'Range' in headers else 'wb'
                
                with save_path.open(mode) as f, self._create_progress_bar(
                    file_size, save_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            size = f.write(chunk)
                            pbar.update(size)
                
                if expected_hash and not self._verify_file_integrity(save_path, expected_hash):
                    save_path.unlink()
                    raise DownloadError("File integrity check failed")
                    
                return True, f"Successfully downloaded {url}"
                
            except Exception as e:
                logger.error(f"Download failed: {str(e)}")
                return False, str(e)

    def download_with_resume(self, url: str, save_path: str, **kwargs) -> Tuple[bool, str]:
        """Support download resume capability."""
        file_path = Path(save_path)
        headers = kwargs.get('headers', {})
        
        if file_path.exists():
            file_size = file_path.stat().st_size
            headers['Range'] = f'bytes={file_size}-'
            
        return self.download_file(url, save_path, headers=headers, **kwargs)

    def parallel_download(self, urls: list, save_dir: str, max_workers: int = 4) -> Dict[str, Tuple[bool, str]]:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for url in urls:
                save_path = Path(save_dir) / Path(urlparse(url).path).name
                futures.append(executor.submit(self.download_file, url, str(save_path)))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results[url] = result
        return results

    def _create_progress_bar(self, total: int, desc: str) -> tqdm:
        """Create enhanced progress bar with ETA and speed."""
        return tqdm(
            total=total,
            desc=desc,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

# Example usage:
if __name__ == "__main__":
    downloader = FileDownloader()
    success, message = downloader.download_file(
        url="https://example.com/file.zip",
        save_path="downloads/file.zip",
        expected_hash="123...abc",  # SHA-256 hash
        timeout=60
    )
    print(message)
