# Mini Project: Parallel Image Processing

## Goal
Convert a colour image into a grayscale image and compare serial processing with vectorized processing.

## Why SIMD?
Every pixel uses the same grayscale calculation. A vectorized operation applies the same formula to many pixel values together. NumPy performs these array operations using optimized low-level routines; this is a practical demonstration of data-parallel processing and SIMD-style work. It is not a guarantee that every NumPy operation maps to a particular CPU SIMD instruction.

## Formula
Gray = 0.299 × Red + 0.587 × Green + 0.114 × Blue

## Requirements
- Python 3.9 or newer
- NumPy
- Pillow

Install dependencies:
```bash
pip install numpy pillow
```

## Run
From this folder:
```bash
python parallel_image_processing.py
```
Or provide your own image:
```bash
python parallel_image_processing.py path/to/your_image.jpg
```

## Project files
- `parallel_image_processing.py` — complete program
- `input/sample_colour_image.png` — sample input
- `outputs/grayscale_serial.png` — serial result
- `outputs/grayscale_vectorized.png` — vectorized result
- `outputs/before_after_comparison.png` — original and grayscale side by side

## Measurement
The program measures both methods on the same image and reports the speed-up:
`speed-up = serial execution time / vectorized execution time`

Actual speed-up depends on image size, processor, Python/NumPy version, memory bandwidth and overhead. For a fair comparison, use the same device and image and repeat the measurements.

## Sample run in this build
Image size: 1200 × 800 pixels
Serial run: 1.380480 seconds
Vectorized run: 0.040187 seconds
Speed-up for this run: 34.35×
Pixel outputs identical: True

Times vary between computers and runs. The values above are illustrative measurements from the environment where this sample was built.


## Host it online (Streamlit Community Cloud)

1. Create a GitHub account at https://github.com if you do not already have one.
2. Extract this ZIP file on your computer.
3. Create a new **public** GitHub repository, for example `parallel-image-processing`.
4. Upload the contents of this project folder to the repository root. Make sure `app.py`, `requirements.txt`, and the `input` folder are at the top level.
5. Visit https://share.streamlit.io and sign in with GitHub.
6. Select **Create app**, choose your repository, branch (usually `main`), and the entrypoint file `app.py`.
7. Click **Deploy**. Streamlit will build the app and provide a shareable `*.streamlit.app` URL.

The hosted web interface is `app.py`; the command-line demonstration remains available as `parallel_image_processing.py`.
