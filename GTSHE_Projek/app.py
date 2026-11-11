import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from python.preprocessing import preprocess_image
from python.he import histogram_equalization
from python.gtshe import gtshe
from python.evaluation import evaluate_image


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="GTSHE Image Enhancement",
    page_icon="🖼️",
    layout="wide"
)


# JUDUL APLIKASI
st.title("GTSHE Image Enhancement")

st.write(
    "Implementasi dan Evaluasi Metode Global Two-Stage "
    "Histogram Equalization untuk Peningkatan Kualitas "
    "Citra Grayscale Berkontras Rendah."
)

st.divider()


# LOKASI DATASET
BASE_DIR = Path(__file__).parent

DATASET_DIR = {
    "scikit-image": BASE_DIR / "dataset" / "scikit_image",
    "USC-SIPI": BASE_DIR / "dataset" / "USC_SIPI",
    "Kylberg": BASE_DIR / "dataset" / "Kylberg"
}

SUPPORTED_FORMATS = [
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
]


# FUNGSI MEMBACA DATASET
def get_dataset_images(folder):

    if not folder.exists():
        return []

    images = []

    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            images.append(file)

    return sorted(images)


# FUNGSI HISTOGRAM
def show_histogram(image, title):

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.hist(
        image.ravel(),
        bins=256,
        range=(0, 256)
    )

    ax.set_title(title)
    ax.set_xlabel("Intensitas Keabuan")
    ax.set_ylabel("Jumlah Piksel")
    ax.set_xlim(0, 255)

    st.pyplot(fig)
    plt.close(fig)


# PILIH SUMBER GAMBAR
st.subheader("Pilih Sumber Gambar")

source = st.radio(
    "Sumber gambar:",
    ["Upload Gambar", "Gunakan Dataset"],
    horizontal=True
)


selected_image = None
image_name = None


# PILIHAN UPLOAD GAMBAR
if source == "Upload Gambar":

    uploaded_file = st.file_uploader(
        "Unggah gambar dari perangkat",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded_file is not None:

        selected_image = Image.open(uploaded_file)
        image_name = uploaded_file.name


# PILIHAN DATASET
elif source == "Gunakan Dataset":

    dataset_name = st.selectbox(
        "Pilih Dataset",
        list(DATASET_DIR.keys())
    )

    dataset_images = get_dataset_images(
        DATASET_DIR[dataset_name]
    )

    if dataset_images:

        selected_file = st.selectbox(
            "Pilih Gambar",
            dataset_images,
            format_func=lambda x: x.relative_to(
                DATASET_DIR[dataset_name]
            ).as_posix()
        )

        selected_image = Image.open(selected_file)
        image_name = selected_file.name

    else:

        st.warning(
            f"Belum ada gambar di folder dataset {dataset_name}. "
            "Silakan masukkan gambar terlebih dahulu."
        )


# PROSES CITRA
if selected_image is not None:

    try:

        original = preprocess_image(selected_image)

        result_he = histogram_equalization(original)

        result_gtshe = gtshe(original)

        st.success("Gambar berhasil diproses.")

        st.caption(
            f"Nama gambar: {image_name} | "
            f"Ukuran: {original.shape[1]} × {original.shape[0]} piksel"
        )

        st.divider()


        # HASIL PENGOLAHAN
        st.subheader("Hasil Pengolahan Citra")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### Citra Asli")

            st.image(
                original,
                use_container_width=True
            )

        with col2:

            st.markdown("### Hasil HE")

            st.image(
                result_he,
                use_container_width=True
            )

        with col3:

            st.markdown("### Hasil GTSHE")

            st.image(
                result_gtshe,
                use_container_width=True
            )


        st.divider()


        # HISTOGRAM
        st.subheader("Perbandingan Histogram")

        hist_col1, hist_col2, hist_col3 = st.columns(3)

        with hist_col1:
            show_histogram(original, "Citra Asli")

        with hist_col2:
            show_histogram(result_he, "Histogram Equalization")

        with hist_col3:
            show_histogram(result_gtshe, "GTSHE")


        st.divider()


        # EVALUASI
        st.subheader("Evaluasi Kualitas Citra")

        evaluation_original = evaluate_image(
            original,
            original
        )

        evaluation_he = evaluate_image(
            original,
            result_he
        )

        evaluation_gtshe = evaluate_image(
            original,
            result_gtshe
        )

        evaluation_table = pd.DataFrame(
            [
                evaluation_original,
                evaluation_he,
                evaluation_gtshe
            ],
            index=[
                "Citra Asli",
                "HE",
                "GTSHE"
            ]
        )

        st.dataframe(
            evaluation_table.round(4),
            use_container_width=True
        )

        st.caption(
            "PSNR dihitung terhadap citra asli. "
            "Nilai PSNR yang tinggi menunjukkan kemiripan "
            "yang lebih besar dengan citra asli, bukan "
            "otomatis peningkatan kualitas visual."
        )

    except Exception as e:

        st.error(f"Terjadi kesalahan: {e}")


else:

    st.info(
        "Silakan pilih sumber gambar dan tentukan gambar "
        "yang ingin diproses."
    )