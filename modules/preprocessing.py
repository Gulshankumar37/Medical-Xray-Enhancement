"""Read an upload once, keep its original pixels, and prepare a grayscale copy."""

from dataclasses import dataclass
from io import BytesIO
from struct import error as StructError
from zlib import error as ZlibError

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_FILE_MB = 32
MAX_PIXELS = 16_000_000
MAX_SIDE = 16_384


class ImageInputError(ValueError):
    """An upload problem that can be explained directly to the user."""


@dataclass(frozen=True)
class SourceImage:
    original: np.ndarray  # Grayscale, RGB, or RGBA, in display order
    gray: np.ndarray
    file_format: str

    @property
    def bit_depth(self) -> int:
        return self.gray.itemsize * 8


def check_grayscale(image: np.ndarray) -> None:
    """The processing modules work with nonempty 8- or 16-bit grayscale arrays."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or not image.size:
        raise ValueError("A nonempty, two-dimensional grayscale image is required.")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Use unsigned 8-bit or 16-bit pixels.")


def load_image(data: bytes) -> SourceImage:
    """Validate the actual file contents before allocating the full image."""
    if not data:
        raise ImageInputError("The file is empty. Choose a PNG or JPEG image.")
    if len(data) > MAX_FILE_MB * 1024**2:
        raise ImageInputError(f"Choose a file of {MAX_FILE_MB} MB or less.")

    try:
        with Image.open(BytesIO(data), formats=("PNG", "JPEG")) as header:
            file_format = header.format
            width, height = header.size
            if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
                raise ImageInputError(
                    "This image is too large. Use at most 16 megapixels "
                    f"with each side at most {MAX_SIDE:,} pixels."
                )
            if getattr(header, "n_frames", 1) != 1:
                raise ImageInputError("Choose a single image. Animated PNG files are unsupported.")

            if file_format == "JPEG":
                # Pillow checks the complete JPEG as it decodes and honors camera rotation.
                upright = ImageOps.exif_transpose(header)
                original = np.array(upright if upright.mode == "L" else upright.convert("RGB"))
            else:
                # Verify PNG chunks, then decode with OpenCV to retain 16-bit color and grayscale.
                header.verify()
                original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if original is None or original.shape[:2] != (height, width):
                    raise ImageInputError("The PNG could not be decoded. Choose an intact image.")
                if original.ndim == 3:
                    conversion = cv2.COLOR_BGRA2RGBA if original.shape[2] == 4 else cv2.COLOR_BGR2RGB
                    original = cv2.cvtColor(original, conversion)

        gray = original
        if original.ndim == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            if original.shape[2] == 4:
                # Match the black backdrop used to display transparent pixels.
                peak = np.iinfo(original.dtype).max
                gray = cv2.multiply(gray, original[:, :, 3], scale=1.0 / peak)
        check_grayscale(gray)
    except ImageInputError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageInputError("This image is too large. Choose an image of 16 megapixels or less.") from exc
    except (OSError, ValueError, SyntaxError, EOFError, StructError, ZlibError, cv2.error) as exc:
        raise ImageInputError("The file could not be read. Choose an intact PNG or JPEG image.") from exc

    original.setflags(write=False)
    gray.setflags(write=False)
    return SourceImage(original, gray, file_format)


def make_preview(image: np.ndarray, longest_side: int = 960) -> np.ndarray:
    """Make a screen-sized preview. Both panels use the same full-range mapping."""
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=255.0 / 65535)
    if image.ndim == 3 and image.shape[2] == 4:
        # Composite before resizing so hidden colors cannot bleed into visible pixels.
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2mRGBA)[:, :, :3]
    height, width = image.shape[:2]
    scale = min(1.0, longest_side / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def encode_png(image: np.ndarray) -> bytes:
    """Encode a full-resolution grayscale result losslessly at its native bit depth."""
    check_grayscale(image)
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise ValueError("The enhanced image could not be encoded as PNG.")
    return encoded.tobytes()
