import os
import hashlib
import logging
from typing import Optional, Tuple, Union, Dict
from pathlib import Path
import mmap
from contextlib import contextmanager
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FileSystem")

class FileSystemError(Exception):
    """Base exception for file operations."""
    pass

class FileHandler:
    def __init__(self):
        self.read_cache: Dict[str, Tuple[float, str]] = {}
        self.hash_cache: Dict[str, Tuple[float, str]] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Configure handler-specific logging."""
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler("file_operations.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def read_file(
            self, 
            path: Union[str, Path], 
            encoding: str = 'utf-8',
            verify_hash: bool = False,
            chunk_size: int = 8192
        ) -> Tuple[bool, str]:
        """
        Securely read file with error handling and optional hash verification.
        Returns (success, content/error_message)
        """
        try:
            # Add file size check before opening
            if not os.path.exists(path):
                return False, f"File not found: {path}"
            
            if os.path.getsize(path) == 0:
                return False, "File is empty"

            path = Path(path).resolve()
            self._validate_path(path)
            
            if not path.exists():
                return False, f"File not found: {path}"

            # Check file size before reading
            file_size = path.stat().size
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return False, "File too large for direct reading"

            content = ""
            hash_obj = hashlib.sha256() if verify_hash else None
            
            with open(path, 'r', encoding=encoding, errors='replace') as f:
                while chunk := f.read(chunk_size):
                    content += chunk
                    if hash_obj:
                        hash_obj.update(chunk.encode(encoding))

            if verify_hash:
                self.hash_cache[str(path)] = (os.path.getmtime(path), hash_obj.hexdigest())

            return True, content

        except UnicodeDecodeError:
            return False, f"File encoding error. Try different encoding for: {path}"
        except PermissionError:
            return False, f"Permission denied: {path}"
        except Exception as e:
            self.logger.error(f"Error reading {path}: {str(e)}")
            return False, str(e)

    def write_file(
            self, 
            path: Union[str, Path], 
            content: str,
            backup: bool = True,
            create_dirs: bool = True
        ) -> Tuple[bool, str]:
        """
        Securely write file with backup and directory creation options.
        Returns (success, message)
        """
        try:
            # Add content validation
            if not content:
                return False, "Empty content provided"

            # Add disk space check
            if not self._check_disk_space(path, len(content)):
                return False, "Insufficient disk space"

            path = Path(path).resolve()
            self._validate_path(path)
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                shutil.copy2(path, backup_path)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"Successfully wrote {len(content)} characters to {path}"
            
        except PermissionError:
            return False, f"Permission denied: {path}"
        except Exception as e:
            self.logger.error(f"Error writing {path}: {str(e)}")
            return False, str(e)

    def _check_disk_space(self, path: Union[str, Path], content_size: int) -> bool:
        """Check if there's enough disk space for writing"""
        try:
            path = Path(path)
            free_space = shutil.disk_usage(path.parent).free
            return free_space > content_size * 2  # 2x safety factor
        except:
            return True  # Default to True if check fails

    @contextmanager
    def mmap_file(self, path: Union[str, Path], access: str = 'r') -> mmap.mmap:
        """Memory-map file for efficient reading of large files."""
        path = Path(path).resolve()
        self._validate_path(path)
        
        try:
            with open(path, access + 'b') as f:
                with mmap.mmap(
                    f.fileno(),
                    0,
                    access=mmap.ACCESS_READ if access == 'r' else mmap.ACCESS_WRITE
                ) as mm:
                    yield mm
        except Exception as e:
            self.logger.error(f"Memory mapping error for {path}: {str(e)}")
            raise FileSystemError(f"Failed to memory-map file: {str(e)}")

    def verify_file_integrity(self, path: Union[str, Path], expected_hash: str) -> bool:
        """Verify file integrity using SHA-256."""
        path = Path(path).resolve()
        
        if not path.exists():
            return False

        try:
            hash_obj = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest() == expected_hash
        except Exception as e:
            self.logger.error(f"Hash verification failed for {path}: {str(e)}")
            return False

    def _validate_path(self, path: Path) -> None:
        """Validate path for potential security issues."""
        try:
            resolved = path.resolve()
            if not str(resolved).startswith(str(Path.cwd())):
                raise FileSystemError("Path outside current working directory")
            if any(part.startswith('.') for part in resolved.parts):
                raise FileSystemError("Hidden directories not allowed")
        except Exception as e:
            raise FileSystemError(f"Path validation failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    handler = FileHandler()
    
    # Read file example
    success, content = handler.read_file("example.txt", verify_hash=True)
    if success:
        print(f"Content: {content}")
    else:
        print(f"Error: {content}")
    
    # Write file example
    success, message = handler.write_file("output.txt", "Hello, World!", backup=True)
    print(message)
    
    # Memory-mapped reading example
    with handler.mmap_file("large_file.txt") as mm:
        print(f"First line: {mm.readline().decode()}")
