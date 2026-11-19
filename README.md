# Medical X-Ray Image Enhancement System

Radiograph Studio is a local Streamlit dashboard for exploring traditional X-ray image processing. Open the bundled chest radiograph or upload a PNG or JPEG, choose an enhancement, and compare the result with the original. Every adjustment starts from the same source pixels, so changing techniques never compounds earlier edits.

## Installation

Use Python 3.11 or newer. The application has been tested with 64-bit Python 3.14 on Windows. Open a terminal in the `Medical-Xray-Enhancement` directory.

### Windows PowerShell

Create the environment with the Python 3.14 launcher.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the dashboard

Launch Streamlit from the activated environment.

```bash
streamlit run app.py
```

Open the local address printed in the terminal, normally [http://127.0.0.1:8501](http://127.0.0.1:8501). Press **Ctrl+C** in the terminal to stop the server. Once the packages are installed, the dashboard, sample, enhancement operations, and downloads work offline.

If PowerShell restricts virtual-environment activation, these commands use the environment's executables directly.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

## Using the workspace

1. Start with the bundled radiograph, or choose **Upload** in **Image source** and select a PNG, JPG, or JPEG file.
2. Choose a **Category** and **Technique**. The settings panel shows the parameters that apply to that technique. The comparison updates when a control changes.
3. Inspect the original and enhanced panels. The histogram overlays their grayscale distributions on shared axes. The dashed line represents the original and the solid teal line represents the result.
4. Choose **Download PNG** to save the enhanced grayscale image at its full resolution and original bit depth. The filename includes the selected operation.
5. Remove the uploaded file to return to the bundled sample.

## Enhancement techniques

| Category | Technique | Operation and controls |
| --- | --- | --- |
| Histogram equalization | Global histogram equalization | Remaps intensities using the cumulative histogram of the whole image. A constant image retains its value. |
| Histogram equalization | CLAHE | Contrast Limited Adaptive Histogram Equalization adjusts local tiles and blends the boundaries. **Clip limit** controls local contrast amplification. **Tiles per axis** sets the number of regions. |
| Contrast enhancement | Contrast stretching | Maps the selected lower and upper percentiles to black and white, clipping values beyond them. Choose **0–100** for min–max stretching. A window containing a single intensity leaves the image unchanged. |
| Contrast enhancement | Gamma correction | Applies `output = peak × (input / peak) ** gamma`. A gamma below 1 brightens midtones, 1 preserves them, and a gamma above 1 darkens them. |
| Image filtering | Gaussian filter | Blurs with a Gaussian-weighted neighborhood. **Kernel size** sets its width and **sigma** sets the spread in pixels. |
| Image filtering | Median filter | Replaces each pixel with its neighborhood median to suppress isolated bright or dark speckles. Choose an odd **kernel size**. OpenCV supports 3 or 5 for 16-bit data. |
| Image filtering | Bilateral filter | Averages neighbors according to distance and brightness similarity. Adjust the neighborhood diameter, intensity sigma, and spatial sigma. The intensity control uses a 0–255 scale for both bit depths. |
| Image sharpening | Laplacian sharpening | Subtracts a signed, four-neighbor Laplacian. **Sharpening strength** controls the edge boost. |
| Image sharpening | Unsharp masking | Adds a scaled difference between the original and a Gaussian-blurred copy. **Blur sigma** sets the detail scale and **detail strength** sets the amount added back. |

## Image handling

- File validation checks the decoded format, byte size, and dimensions. Each upload can contain one image up to **32 MiB**, **16 million pixels**, and **16,384 pixels per side**. Invalid or damaged files produce an on-screen message.
- The original pixels remain read-only in the current session. Color uploads retain a color reference panel. OpenCV converts a separate working image to grayscale. JPEG rotations are applied to both views, and transparent PNG regions are composited on black for display and enhancement.
- PNG inputs retain 8-bit or 16-bit precision throughout enhancement. Previews fit within 960 pixels on their longest side and use the same linear mapping in both panels. A 16-bit level of 0 displays as black and 65,535 displays as white.
- Both histograms use all full-resolution pixels and 256 equal-width bins. The horizontal axis shows native gray levels, and the vertical axis shows the percentage of pixels in each bin.
- Exports use lossless PNG encoding. The dashboard keeps the active source in session memory, and downloads go to the location chosen by your browser.

## Technologies

Python runs the application. OpenCV provides the enhancement operations, histogram counts, and PNG encoder. NumPy maintains pixel arrays and intensity curves. Pillow validates input files and decodes JPEGs. Matplotlib draws the histogram and Streamlit provides the interactive dashboard. The headless OpenCV package works in a web application without a separate desktop viewer.

## Project layout

```text
Medical-Xray-Enhancement/
├── app.py                      Streamlit dashboard and parameter controls
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py        Validation, decoding, previews, and PNG export
│   ├── histogram.py            Global equalization, CLAHE, and histogram counts
│   ├── contrast.py             Percentile stretching and gamma correction
│   ├── filtering.py            Gaussian, median, and bilateral filters
│   └── sharpening.py           Laplacian sharpening and unsharp masking
├── dataset/chest-pa.jpg        Bundled offline sample
├── assets/workspace.css        Responsive dashboard styles
├── .streamlit/config.toml      Local server, upload limit, and theme
├── tests/
│   ├── test_processing.py      Algorithms, validation, bit depth, and PNG round trips
│   └── test_dashboard.py       Menu selection and recovery from invalid parameters
├── requirements.txt
├── README.md
└── .gitignore
```

## Tests

Run the test suite from the project directory with the virtual environment active.

```bash
python -m unittest discover -s tests -v
```

The tests exercise all nine techniques, both integer bit depths, tiny and constant images, extreme contrast windows, color and transparency, damaged uploads, and lossless export. Streamlit's app tests exercise each menu option and check that correcting invalid settings restores the download.

## Sample credit

The bundled `dataset/chest-pa.jpg` is Mikael Häggström's 2017 posteroanterior chest radiograph, distributed under CC0 1.0. [Original image and attribution](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg) · [CC0 dedication](https://creativecommons.org/publicdomain/zero/1.0/)
