import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
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
   'freshapples',
 'freshbanana',
 'freshoranges',
 'rottenapples',
 'rottenbanana',
 'rottenoranges'
]

# Ambang batas keyakinan (Confidence Threshold diturunkan ke 30.0 agar lebih toleran)
CONFIDENCE_THRESHOLD = 30.0  

# 2. Cache pembacaan model Keras agar efisien
@st.cache_resource
def load_keras_model():
    model_path = 'model_terbaik.keras'
    
    # Jika file model belum ada di server Streamlit, download dari Google Drive
    if not os.path.exists(model_path):
        file_id = '1-vtNHLO1uFLpDK4AWalGP5IFLC_SeBBA'  
        url = f'https://drive.google.com/uc?id={file_id}'
        
        with st.spinner("Mengunduh model dari cloud (hanya saat pertama kali)..."):
            gdown.download(url, model_path, quiet=False)
            
    return tf.keras.models.load_model(model_path)

# Memuat model Keras
model = None
try:
    with st.spinner("Memuat model klasifikasi..."):
        model = load_keras_model()
except Exception as e:
    st.error(f"Gagal memuat file model! Pastikan akses file Google Drive sudah publik. Detail Error: {e}")

# 3. Form Upload Gambar
uploaded_file = st.file_uploader("Unggah gambar buah (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Pastikan model sudah ter-load sebelum melakukan klasifikasi
    if model is None:
        st.error("Model belum siap. Mohon periksa kembali koneksi atau ketersediaan file model!")
    else:
        # Tampilkan Gambar yang Diunggah
        image = Image.open(uploaded_file).convert("RGB") # Memastikan format 3 kanal RGB
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        
        # Preprocessing Gambar: Resize standar tanpa memotong gambar
        target_size = (224, 224) 
        image_resized = image.resize(target_size)
        
        # Konversi ke array
        img_array = np.asarray(image_resized)
        
        # Tambahkan dimensi batch: (1, 224, 224, 3)
        data = np.expand_dims(img_array.astype(np.float32), axis=0)

        # Tombol Analisis Prediksi
        if st.button("🔍 Analisis Kesegaran", type="primary"):
            with st.spinner("Menganalisis kondisi buah..."):
                predictions = model.predict(data)
                st.write("Input shape:", model.input_shape, "| Output shape:", model.output_shape)

                # Cek tiap varian normalisasi
                raw = np.asarray(image_resized).astype(np.float32)
                variants = {
                    "raw 0-255": raw,
                    "0-1": raw / 255.0,
                    "-1 sampai 1": raw / 127.5 - 1.0,
                }
                for name, arr in variants.items():
                    p = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                    st.write(name, "| sum:", round(float(p.sum()), 3), "| top:", CLASS_NAMES[int(np.argmax(p))], round(float(p.max()), 3))
                
                # Ambil indeks kelas dengan probabilitas tertinggi
                class_index = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                raw_label = CLASS_NAMES[class_index]

                # Simplifikasi label mentah menjadi 2 Kategori
                if raw_label.startswith("fresh"):
                    simple_label = "Buah Segar"
                else:
                    simple_label = "Buah Busuk"

                st.divider()
                
                # Pengecekan ambang batas keyakinan
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"⚠️ Buah tidak terdeteksi dengan jelas atau di luar dataset (Confidence: {confidence:.2f}%).")
                    st.info("Saran: Unggah gambar jenis buah yang sesuai dengan pencahayaan yang terang.")
                else:
                    # Tampilkan hasil simplifikasi (Hanya Segar atau Busuk)
                    if simple_label == "Buah Segar":
                        st.success(f"### Hasil Prediksi: **{simple_label}** 🟢")
                    else:
                        st.error(f"### Hasil Prediksi: **{simple_label}** 🔴")
                    
                    st.caption(f"Kelas Asli: **{raw_label}**")
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")
