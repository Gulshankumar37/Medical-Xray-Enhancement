"""Run from the project directory with python -m unittest discover -s tests -v."""

from io import BytesIO
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from modules import contrast, filtering, histogram, sharpening
from modules.preprocessing import ImageInputError, encode_png, load_image, make_preview


ALGORITHMS = (
    histogram.global_equalization, histogram.clahe,
    contrast.contrast_stretching, contrast.gamma_correction,
    filtering.gaussian_filter, filtering.median_filter, filtering.bilateral_filter,
    sharpening.laplacian_sharpening, sharpening.unsharp_masking,
)


class EnhancementTests(unittest.TestCase):
    def test_every_algorithm_preserves_source_shape_and_bit_depth(self):
        rng = np.random.default_rng(42)
        for dtype in (np.uint8, np.uint16):
            for shape in ((31, 47), (1, 1), (1, 17), (19, 1)):
                image = rng.integers(0, int(np.iinfo(dtype).max) + 1, shape, dtype=dtype)
                reference = image.copy()
                for operation in ALGORITHMS:
                    with self.subTest(method=operation.__name__, dtype=dtype, shape=shape):
                        result = operation(image)
                        self.assertEqual(result.shape, shape)
                        self.assertEqual(result.dtype, dtype)
                        np.testing.assert_array_equal(image, reference)
                        np.testing.assert_array_equal(load_image(encode_png(result)).gray, result)

    def test_equalization_uses_the_full_native_range(self):
        for dtype in (np.uint8, np.uint16):
            image = np.array([[10, 10, 20, 30]], dtype=dtype)
            peak = int(np.iinfo(dtype).max)
            expected = [[0, 0, round(peak / 2), peak]]
            np.testing.assert_array_equal(histogram.global_equalization(image), expected)
            flat = np.full((5, 7), 99, dtype=dtype)
            np.testing.assert_array_equal(histogram.global_equalization(flat), flat)
            counts = histogram.intensity_histogram(np.array([[0, 0, peak]], dtype=dtype))
            self.assertEqual((counts.size, counts[0], counts[-1], counts.sum()), (256, 2, 1, 3))

    def test_clahe_responds_to_the_selected_grid(self):
        image = np.tile(np.arange(80, 144, dtype=np.uint8), (64, 1))
        fine = histogram.clahe(image, clip_limit=3, grid_size=8)
        coarse = histogram.clahe(image, clip_limit=3, grid_size=2)
        self.assertFalse(np.array_equal(fine, coarse))
        self.assertGreater(np.ptp(fine), 0)

    def test_percentiles_clip_and_a_flat_image_is_stable(self):
        image = np.array([[0, 10, 20, 30, 40]], dtype=np.uint8)
        np.testing.assert_array_equal(
            contrast.contrast_stretching(image, 25, 75), [[0, 0, 128, 255, 255]]
        )
        flat = np.full((7, 9), 1531, dtype=np.uint16)
        np.testing.assert_array_equal(contrast.contrast_stretching(flat), flat)

    def test_gamma_brightens_and_darkens_midtones(self):
        for dtype in (np.uint8, np.uint16):
            peak = int(np.iinfo(dtype).max)
            image = np.array([[0, peak // 2, peak]], dtype=dtype)
            np.testing.assert_array_equal(contrast.gamma_correction(image, 1), image)
            self.assertGreater(contrast.gamma_correction(image, 0.5)[0, 1], image[0, 1])
            self.assertLess(contrast.gamma_correction(image, 2)[0, 1], image[0, 1])

    def test_narrow_16_bit_stretch_clips_outliers_to_the_correct_endpoint(self):
        for low in (1, 32000, 65000):
            image = np.array([[0, low, low, low + 1, low + 1, 65535]], dtype=np.uint16)
            result = contrast.contrast_stretching(image, 25, 75)
            np.testing.assert_array_equal(result, [[0, 0, 0, 65535, 65535, 65535]])

    def test_filters_smooth_an_impulse_and_respect_an_intensity_edge(self):
        for dtype in (np.uint8, np.uint16):
            peak = int(np.iinfo(dtype).max)
            image = np.zeros((15, 15), dtype=dtype)
            image[7, 7] = peak
            self.assertEqual(filtering.median_filter(image)[7, 7], 0)
            blurred = filtering.gaussian_filter(image)
            self.assertTrue(0 < blurred[7, 6] < blurred[7, 7] < peak)
            self.assertEqual(filtering.bilateral_filter(image, sigma_color=3)[7, 7], peak)

    def test_sharpening_uses_both_sides_of_an_edge(self):
        image = np.full((11, 11), 100, dtype=np.uint8)
        image[:, 5:] = 140
        for operation in (sharpening.laplacian_sharpening, sharpening.unsharp_masking):
            np.testing.assert_array_equal(operation(image, strength=0), image)
            result = operation(image, strength=1)
            self.assertLess(result[5, 4], 100)
            self.assertGreater(result[5, 5], 140)

    def test_invalid_parameters_report_a_value_error(self):
        image = np.zeros((9, 9), np.uint16)
        cases = (
            (histogram.clahe, {"grid_size": 0}),
            (contrast.contrast_stretching, {"low_percentile": 99, "high_percentile": 99}),
            (contrast.gamma_correction, {"gamma": 0}),
            (filtering.gaussian_filter, {"kernel_size": 4}),
            (filtering.median_filter, {"kernel_size": 7}),
            (filtering.bilateral_filter, {"sigma_color": -1}),
            (sharpening.laplacian_sharpening, {"strength": -1}),
            (sharpening.unsharp_masking, {"sigma": 0}),
        )
        for operation, params in cases:
            with self.subTest(method=operation.__name__), self.assertRaises(ValueError):
                operation(image, **params)


class FileHandlingTests(unittest.TestCase):
    def test_color_order_transparency_and_unmodified_source(self):
        for dtype in (np.uint8, np.uint16):
            peak = int(np.iinfo(dtype).max)
            rgba = np.array([[[peak, 0, 0, peak], [0, peak, 0, 0]]], dtype=dtype)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            source = load_image(cv2.imencode(".png", bgra)[1].tobytes())
            np.testing.assert_array_equal(source.original, rgba)
            self.assertFalse(source.original.flags.writeable)
            self.assertFalse(source.gray.flags.writeable)
            self.assertEqual(source.gray[0, 1], 0)
            self.assertAlmostEqual(int(source.gray[0, 0]), round(0.299 * peak), delta=2)
            np.testing.assert_array_equal(make_preview(source.original), [[[255, 0, 0], [0, 0, 0]]])
            np.testing.assert_array_equal(make_preview(source.original, longest_side=1), [[[128, 0, 0]]])

    def test_jpeg_rotation_is_shared_by_both_views(self):
        data = BytesIO()
        exif = Image.Exif()
        exif[274] = 6  # Rotate the displayed image 90 degrees clockwise.
        Image.new("RGB", (11, 7), "navy").save(data, "JPEG", exif=exif)
        source = load_image(data.getvalue())
        self.assertEqual(source.file_format, "JPEG")
        self.assertEqual(source.original.shape, (11, 7, 3))
        self.assertEqual(source.gray.shape, (11, 7))

    def test_only_the_screen_preview_is_reduced(self):
        image = np.tile(np.array([0, 257, 32768, 65535], np.uint16), (2, 1))
        np.testing.assert_array_equal(make_preview(image)[0], [0, 1, 128, 255])
        self.assertEqual(make_preview(image, longest_side=2).shape, (1, 2))
        decoded = cv2.imdecode(np.frombuffer(encode_png(image), np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, image)

    def test_invalid_truncated_animated_and_oversized_uploads(self):
        jpeg, gif, animated = BytesIO(), BytesIO(), BytesIO()
        image = Image.new("L", (20, 16), 80)
        image.save(jpeg, "JPEG")
        image.save(gif, "GIF")
        image.save(animated, "PNG", save_all=True, append_images=[image.point(lambda x: x + 1)])
        png = encode_png(np.full((20, 16), 80, np.uint8))
        for data in (b"", b"invalid file", gif.getvalue(), jpeg.getvalue()[:-8], png[:45], animated.getvalue()):
            with self.subTest(length=len(data)), self.assertRaises(ImageInputError):
                load_image(data)
        with patch("modules.preprocessing.MAX_PIXELS", 100), self.assertRaisesRegex(ImageInputError, "large"):
            load_image(png)
        with patch("modules.preprocessing.MAX_FILE_MB", 0), self.assertRaisesRegex(ImageInputError, "MB"):
            load_image(png)


if __name__ == "__main__":
    unittest.main()
