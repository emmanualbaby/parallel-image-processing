import time
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Parallel Image Processing", page_icon="🖼️", layout="wide")

st.title("🖼️ Parallel Image Processing")
st.caption("Mini project: compare serial grayscale conversion with vectorized (SIMD-style) processing.")

st.markdown("""
**Objective:** Convert a colour image into grayscale using two methods and compare their execution times.

**Grayscale formula:**  
`Gray = 0.299 × Red + 0.587 × Green + 0.114 × Blue`
""")

def _image_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

def serial_grayscale(rgb):
    height, width, _ = rgb.shape
    output = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            red, green, blue = map(int, rgb[y, x])
            gray = 0.299 * red + 0.587 * green + 0.114 * blue
            output[y, x] = int(round(gray))
    return output

def vectorized_grayscale(rgb):
    red = rgb[..., 0].astype(np.float64)
    green = rgb[..., 1].astype(np.float64)
    blue = rgb[..., 2].astype(np.float64)
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.rint(gray).clip(0, 255).astype(np.uint8)

uploaded = st.file_uploader("Upload a colour image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")
else:
    sample_path = Path(__file__).parent / "input" / "sample_colour_image.png"
    original = Image.open(sample_path).convert("RGB")
    st.info("Showing the included sample image. Upload your own image above to test it.")

# Keep browser work reasonable on very large images while clearly stating resized dimensions.
max_side = 1600
if max(original.size) > max_side:
    scale = max_side / max(original.size)
    resized = (int(original.width * scale), int(original.height * scale))
    original = original.resize(resized)
    st.warning(f"Image resized to {resized[0]} × {resized[1]} pixels for responsive processing.")

rgb = np.asarray(original, dtype=np.uint8)
height, width, _ = rgb.shape

if st.button("Convert image and compare", type="primary"):
    with st.spinner("Running serial and vectorized conversions..."):
        start = time.perf_counter()
        serial_result = serial_grayscale(rgb)
        serial_seconds = time.perf_counter() - start

        start = time.perf_counter()
        vector_result = vectorized_grayscale(rgb)
        vector_seconds = time.perf_counter() - start

    same = bool(np.array_equal(serial_result, vector_result))
    speedup = serial_seconds / vector_seconds if vector_seconds > 0 else float("inf")

    st.session_state["results"] = {
        "serial": serial_result,
        "vector": vector_result,
        "serial_seconds": serial_seconds,
        "vector_seconds": vector_seconds,
        "same": same,
        "speedup": speedup,
        "width": width,
        "height": height,
        "original": original.copy()
    }

if "results" in st.session_state:
    result = st.session_state["results"]
    left, right = st.columns(2)
    with left:
        st.subheader("Original colour image")
        st.image(result["original"], use_container_width=True)
    with right:
        st.subheader("Grayscale output")
        st.image(result["vector"], clamp=True, use_container_width=True)

    st.subheader("Performance comparison")
    m1, m2, m3 = st.columns(3)
    m1.metric("Serial time", f'{result["serial_seconds"]:.6f} s')
    m2.metric("Vectorized time", f'{result["vector_seconds"]:.6f} s')
    m3.metric("Speed-up", f'{result["speedup"]:.2f}×')

    st.write(f'**Image size:** {result["width"]} × {result["height"]} pixels')
    st.write(f'**Both methods produce identical pixels:** {"Yes" if result["same"] else "No"}')

    st.dataframe(
        {
            "Method": ["Serial pixel-by-pixel", "Vectorized NumPy"],
            "Execution time (seconds)": [
                result["serial_seconds"], result["vector_seconds"]
            ],
        },
        use_container_width=True,
        hide_index=True
    )

    serial_img = Image.fromarray(result["serial"], mode="L")
    vector_img = Image.fromarray(result["vector"], mode="L")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download serial grayscale",
            data=_image_bytes(serial_img),
            file_name="grayscale_serial.png",
            mime="image/png"
        )
    with c2:
        st.download_button(
            "Download vectorized grayscale",
            data=_image_bytes(vector_img),
            file_name="grayscale_vectorized.png",
            mime="image/png"
        )

    st.caption("Note: NumPy vectorization demonstrates data-parallel/SIMD-style processing. Actual speed-up depends on image size, CPU, memory bandwidth, software and overhead; it does not guarantee a specific hardware SIMD instruction.")

else:
    st.subheader("Input image")
    st.image(original, caption=f"{width} × {height} pixels", use_container_width=True)
    st.write("Click **Convert image and compare** to generate the grayscale output and timing results.")

st.divider()
st.caption("MCA Mini Project · Parallel Processing · Flynn’s Classification")
