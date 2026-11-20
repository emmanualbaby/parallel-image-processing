"""
Mini Project: Parallel Image Processing
Convert a colour image to grayscale using:
1) Serial pixel-by-pixel processing
2) Vectorized processing with NumPy (SIMD-style data parallelism)

Install:
    pip install pillow numpy

Run:
    python parallel_image_processing.py input/sample_colour_image.png

The script saves grayscale_serial.png, grayscale_vectorized.png,
and a before_after_comparison.png file in an outputs folder.
"""
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

def serial_grayscale(rgb):
    """Convert pixels one at a time using the given grayscale formula."""
    height, width, _ = rgb.shape
    output = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            red, green, blue = map(int, rgb[y, x])
            gray = 0.299 * red + 0.587 * green + 0.114 * blue
            output[y, x] = int(round(gray))
    return output

def vectorized_grayscale(rgb):
    """Apply the same formula to whole arrays using NumPy vector operations."""
    red = rgb[..., 0].astype(np.float64)
    green = rgb[..., 1].astype(np.float64)
    blue = rgb[..., 2].astype(np.float64)
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.rint(gray).clip(0, 255).astype(np.uint8)

def main():
    project_dir = Path(__file__).resolve().parent
    default_image = project_dir / "input" / "sample_colour_image.png"
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width, _ = rgb.shape

    start = time.perf_counter()
    serial = serial_grayscale(rgb)
    serial_seconds = time.perf_counter() - start

    start = time.perf_counter()
    vectorized = vectorized_grayscale(rgb)
    vector_seconds = time.perf_counter() - start

    same = np.array_equal(serial, vectorized)
    speedup = serial_seconds / vector_seconds if vector_seconds else float("inf")

    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    Image.fromarray(serial, mode="L").save(output_dir / "grayscale_serial.png")
    Image.fromarray(vectorized, mode="L").save(output_dir / "grayscale_vectorized.png")

    original = Image.open(image_path).convert("RGB")
    gray_preview = Image.fromarray(vectorized, mode="L").convert("RGB")
    preview = Image.new("RGB", (original.width * 2, original.height + 45), "white")
    preview.paste(original, (0, 45))
    preview.paste(gray_preview, (original.width, 45))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(preview)
    draw.text((12, 14), "Original colour image", fill="black")
    draw.text((original.width + 12, 14), "Grayscale output", fill="black")
    preview.save(output_dir / "before_after_comparison.png")

    print(f"Image size: {width} x {height} pixels")
    print(f"Serial time: {serial_seconds:.6f} seconds")
    print(f"Vectorized time: {vector_seconds:.6f} seconds")
    print(f"Speed-up: {speedup:.2f}x")
    print(f"Outputs identical: {same}")
    print(f"Saved results in: {output_dir}")

if __name__ == "__main__":
    main()
