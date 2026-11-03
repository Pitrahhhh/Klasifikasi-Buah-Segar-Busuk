import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Deteksi Kesegaran Buah",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Deteksi Buah Segar vs Busuk")
st.write("Unggah foto buah untuk mengecek kondisi kesegarannya (**Segar** atau **Busuk**).")

# 1. Masukkan urutan label asli sesuai dengan folder/dataset pelatihan model Keras kamu
CLASS_NAMES = [
    'Apel Busuk', 
    'Apel Segar', 
    'Jeruk Busuk', 
    'Jeruk Segar', 
    'Pisang Busuk', 
    'Pisang Segar'
]

# Ambang batas keyakinan (Confidence Threshold)
# Jika keyakinan model di bawah 60%, dianggap buah tidak dikenali / di luar dataset
CONFIDENCE_THRESHOLD = 60.0  

# 2. Cache pembacaan model Keras agar efisien
@st.cache_resource
def load_keras_model():
    model_path = 'my_model.keras'
    
    # Jika file model belum ada di server Streamlit, download dari Google Drive
    if not os.path.exists(model_path):
        file_id = '1-vtNHLO1uFLpDK4AWalGP5IFLC_SeBBA'  # GANTI DENGAN FILE_ID GOOGLE DRIVE KAMU
        url = f'https://drive.google.com/uc?id={file_id}'
        
        with st.spinner("Mengunduh model dari cloud (hanya saat pertama kali)..."):
            gdown.download(url, model_path, quiet=False)
            
    return tf.keras.models.load_model(model_path)

# 3. Form Upload Gambar
uploaded_file = st.file_uploader("Unggah gambar buah (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar yang Diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    # Preprocessing Gambar
    # Sesuaikan ukuran target dengan resolusi input model Keras kamu (misal 224x224)
    target_size = (224, 224) 
    image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # Konversi ke array dan normalisasi piksel (skala [0, 1])
    img_array = np.asarray(image_resized)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    
    # Tambahkan dimensi batch: (1, 224, 224, 3)
    data = np.expand_dims(normalized_image_array, axis=0)

    # Tombol Analisis Prediksi
    if st.button("🔍 Analisis Kesegaran", type="primary"):
        with st.spinner("Menganalisis kondisi buah..."):
            predictions = model.predict(data)
            
            # Ambil indeks kelas dengan probabilitas tertinggi
            class_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            raw_label = CLASS_NAMES[class_index]

            # -------------------------------------------------------------
            # OPS B LOGIC: Simplifikasi label mentah menjadi 2 Kategori
            # -------------------------------------------------------------
            if "Segar" in raw_label:
                simple_label = "Buah Segar"
            else:
                simple_label = "Buah Busuk"

            st.divider()
            
            # Pengecekan apakah gambar terdeteksi dengan yakin atau di luar dataset
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Buah tidak terdeteksi dengan jelas atau gambar di luar dataset pelatihan model.")
                st.info("Saran: Unggah gambar jenis buah yang sesuai dengan pencahayaan yang terang.")
            else:
                # Tampilkan hasil simplifikasi (Hanya Segar atau Busuk)
                if simple_label == "Buah Segar":
                    st.success(f"### Hasil Prediksi: **{simple_label}** 🟢")
                else:
                    st.error(f"### Hasil Prediksi: **{simple_label}** 🔴")
                
                st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")