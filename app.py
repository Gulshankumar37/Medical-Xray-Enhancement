"""Run the local dashboard with: streamlit run app.py"""

from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from modules import contrast, filtering, histogram, sharpening
from modules.preprocessing import (
    MAX_FILE_MB, ImageInputError, SourceImage, encode_png, load_image, make_preview,
)


ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "dataset" / "chest-pa.jpg"

# A single menu connects each visible technique to its reusable OpenCV function.
TECHNIQUES = {
    "Histogram equalization": {
        "CLAHE": histogram.clahe,
        "Global histogram equalization": histogram.global_equalization,
    },
    "Contrast enhancement": {
        "Contrast stretching": contrast.contrast_stretching,
        "Gamma correction": contrast.gamma_correction,
    },
    "Image filtering": {
        "Gaussian filter": filtering.gaussian_filter,
        "Median filter": filtering.median_filter,
        "Bilateral filter": filtering.bilateral_filter,
    },
    "Image sharpening": {
        "Laplacian sharpening": sharpening.laplacian_sharpening,
        "Unsharp masking": sharpening.unsharp_masking,
    },
}

DESCRIPTIONS = {
    "CLAHE": "Adjusts contrast in small regions. The clip limit controls how much local detail is amplified.",
    "Global histogram equalization": "Redistributes brightness levels using the histogram of the entire image.",
    "Contrast stretching": "Maps the lower percentile to black and the upper percentile to white.",
    "Gamma correction": "Values below 1 brighten midtones. Values above 1 darken them.",
    "Gaussian filter": "Smooths nearby pixels with a weighted average. A larger sigma spreads the blur.",
    "Median filter": "Replaces each pixel with its neighborhood median to reduce isolated speckles.",
    "Bilateral filter": "Smooths pixels that are close in both position and brightness to retain edges.",
    "Laplacian sharpening": "Uses local changes in brightness to emphasize edges.",
    "Unsharp masking": "Adds fine detail back to the image by subtracting a blurred copy.",
}


def parameter_controls(technique: str, bit_depth: int) -> dict:
    """Return the arguments for just the selected operation."""
    if technique == "CLAHE":
        return {
            "clip_limit": st.slider("Clip limit", 0.5, 8.0, 2.0, 0.1),
            "grid_size": st.slider("Tiles per axis", 2, 16, 8),
        }
    if technique == "Contrast stretching":
        low, high = st.slider("Percentile range", 0, 100, (1, 99), format="%d%%")
        return {"low_percentile": low, "high_percentile": high}
    if technique == "Gamma correction":
        return {"gamma": st.slider("Gamma", 0.2, 3.0, 0.8, 0.05)}
    if technique in ("Gaussian filter", "Median filter"):
        largest = 5 if technique == "Median filter" and bit_depth == 16 else 15
        default = 5 if technique == "Gaussian filter" else 3
        size = st.slider("Kernel size", 3, largest, default, 2, key=f"kernel_{technique}_{bit_depth}")
        params = {"kernel_size": size}
        if technique == "Gaussian filter":
            params["sigma"] = st.slider("Gaussian sigma (px)", 0.2, 5.0, 1.2, 0.1)
        return params
    if technique == "Bilateral filter":
        return {
            "diameter": st.slider("Neighborhood diameter (px)", 3, 15, 7, 2),
            "sigma_color": st.slider(
                "Intensity sigma", 1, 150, 35,
                help="Measured on a 0–255 scale and adjusted for 16-bit images.",
            ),
            "sigma_space": st.slider("Spatial sigma (px)", 1, 30, 5),
        }
    if technique == "Laplacian sharpening":
        return {"strength": st.slider("Sharpening strength", 0.0, 2.0, 0.4, 0.05)}
    if technique == "Unsharp masking":
        return {
            "strength": st.slider("Detail strength", 0.0, 3.0, 1.0, 0.1),
            "sigma": st.slider("Blur sigma (px)", 0.3, 5.0, 1.5, 0.1),
        }
    return {}


def clear_source() -> None:
    """Discard the previous session image when the upload changes or is removed."""
    st.session_state.pop("decoded_image", None)


def source_controls() -> tuple[SourceImage | None, str]:
    st.html('<h2 class="panel-title">Image source</h2>')
    uploaded = st.file_uploader(
        "Upload an X-ray image", type=["png", "jpg", "jpeg"], key="upload",
        max_upload_size=MAX_FILE_MB, on_change=clear_source, label_visibility="collapsed",
    )
    filename = Path(uploaded.name).name if uploaded is not None else SAMPLE.name
    try:
        # Cache only the current source within this browser session.
        if "decoded_image" not in st.session_state:
            data = uploaded.getvalue() if uploaded is not None else SAMPLE.read_bytes()
            st.session_state.decoded_image = load_image(data)
        source = st.session_state.decoded_image
    except ImageInputError as exc:
        st.error(str(exc))
        return None, filename
    except OSError:
        st.error("The bundled sample could not be opened. Upload a PNG or JPEG to continue.")
        return None, filename
    except MemoryError:
        st.error("There is too little memory to open this image. Choose a smaller file.")
        return None, filename

    height, width = source.gray.shape
    origin = "Bundled sample" if uploaded is None else "Uploaded image"
    st.html(
        f'<div class="source-detail"><span class="source-kind">{origin}</span>'
        f'<strong>{escape(filename)}</strong>'
        f'<span class="source-meta">{width:,} × {height:,} px &nbsp;·&nbsp; '
        f'{source.bit_depth}-bit {source.file_format}</span></div>'
    )
    return source, filename


def image_panel(title: str, image: np.ndarray, caption: str, key: str) -> None:
    mode = "Grayscale"
    if image.ndim == 3:
        mode = "RGBA" if image.shape[2] == 4 else "RGB"
    with st.container(key=key, gap=None):
        st.html(
            f'<div class="film-heading"><h3>{title}</h3><span>{mode}</span></div>'
        )
        st.image(make_preview(image), width="stretch", output_format="PNG")
        st.html(f'<div class="film-caption">{escape(caption)}</div>')


def draw_histograms(original: np.ndarray, result: np.ndarray) -> None:
    """Overlay full-resolution pixel counts on a common scale."""
    st.html(
        '<div class="chart-heading"><h2 class="panel-title">Intensity histogram</h2>'
        '<div class="chart-legend"><span><i class="original-line"></i>Original</span>'
        '<span><i class="enhanced-line"></i>Enhanced</span></div></div>'
    )
    peak = int(np.iinfo(original.dtype).max)
    # Half-level outer edges keep the black and white endpoint bins fully visible.
    edges = np.linspace(-0.5, peak + 0.5, 257)
    figure = Figure(figsize=(7.6, 2.05), facecolor="white")
    axis = figure.subplots()
    for pixels, color, style in ((original, "#586e88", "--"), (result, "#087d76", "-")):
        percent = histogram.intensity_histogram(pixels) * (100.0 / pixels.size)
        axis.stairs(percent, edges, fill=True, color=color, alpha=0.09, linewidth=0)
        axis.stairs(percent, edges, color=color, linestyle=style, linewidth=1.35)

    axis.set(xlim=(edges[0], edges[-1]), ylim=(0, None), xlabel="Gray level", ylabel="Pixels (%)")
    axis.set_xticks([0, (peak + 1) // 4, (peak + 1) // 2, 3 * (peak + 1) // 4, peak])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e1e8e8", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#bac8c9")
    axis.tick_params(colors="#465c63", labelsize=9, length=0, pad=5)
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set(color="#465c63", size=10)
    figure.subplots_adjust(left=0.1, right=0.975, bottom=0.26, top=0.96)
    st.pyplot(figure, width="stretch")


def show_result(source: SourceImage, filename: str, category: str, technique: str, params: dict) -> None:
    operation = TECHNIQUES[category][technique]
    try:
        with st.spinner("Enhancing image…"):
            result = operation(source.gray.copy(), **params)
            png = encode_png(result)
    except (ValueError, cv2.error, MemoryError) as exc:
        if isinstance(exc, ValueError):
            st.error(str(exc))
        else:
            st.error("The image could not be processed. Reduce the image size or adjust the parameters.")
        image_panel("Original", source.original, filename, "original_view")
        return

    with st.container(border=True, key="comparison_card"):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.html('<h2 class="panel-title">Image comparison</h2>', width="content")
            st.download_button(
                "Download PNG", data=png, mime="image/png", icon=":material/download:",
                file_name=f"{Path(filename).stem}_{operation.__name__}.png",
                type="primary", on_click="ignore",
                help="Save the enhanced image at its full resolution and bit depth.",
            )
        before, after = st.columns(2, gap="small")
        with before:
            image_panel("Original", source.original, filename, "original_view")
        with after:
            image_panel("Enhanced", result, technique, "enhanced_view")

    with st.container(border=True, key="histogram_card"):
        try:
            draw_histograms(source.gray, result)
        except (cv2.error, MemoryError, OSError, ValueError):
            st.error("The histogram could not be drawn. The enhanced PNG is available above.")


def main() -> None:
    st.set_page_config(page_title="Medical X-Ray Image Enhancement System", page_icon="🩻", layout="wide")
    st.html(ROOT / "assets" / "workspace.css")
    st.html(
        '<header class="masthead"><div class="brand"><span class="brand-mark" aria-hidden="true">'
        '<i></i></span><span>Radiograph <b>Studio</b></span></div>'
        '<span class="local-status"><i></i>Local processing</span></header>'
    )
    st.html(
        '<div class="workspace-intro"><h1>Medical X-ray enhancement</h1>'
        '<p>Adjust contrast, reduce noise, and compare fine detail in your radiographs.</p></div>'
    )
    with st.container(key="workspace_layout"):
        controls, workspace = st.columns([1, 2.55], gap="medium")
        with controls:
            with st.container(border=True, key="source_card"):
                source, filename = source_controls()
            if source is not None:
                with st.container(border=True, key="settings_card"):
                    st.html('<h2 class="panel-title">Enhancement settings</h2>')
                    category = st.selectbox("Category", list(TECHNIQUES))
                    technique = st.selectbox("Technique", list(TECHNIQUES[category]), key=f"method_{category}")
                    st.caption(DESCRIPTIONS[technique])
                    params = parameter_controls(technique, source.bit_depth)
        with workspace:
            if source is None:
                st.info("Choose a valid image to start the comparison.")
            else:
                show_result(source, filename, category, technique, params)


if __name__ == "__main__":
    main()
