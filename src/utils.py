import hashlib
from pathlib import Path
import cv2
import imagehash
from PIL import Image
import yaml

# Config
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

class FileUtils:
    """Filesystem & hashing utilities"""

    @staticmethod
    def is_image(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTS

    @staticmethod
    def md5(path: Path) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def phash(path: Path):
        try:
            return imagehash.phash(Image.open(path))
        except Exception as e:
            print(f"Warning: Could not hash {path}: {e}")
            return None

class ImageUtils:
    """Image-level quality metrics"""

    @staticmethod
    def load_gray(path: Path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def blur_score(img):
        # Variance of Laplacian (higher = sharper)
        return cv2.Laplacian(img, cv2.CV_64F).var()

    @staticmethod
    def contrast(img):
        return img.std()
    
class ConfigUtils:
    @staticmethod
    def load_yaml(path: Path):
        with open(path, "r") as f:
            return yaml.safe_load(f)