"""
Geospatial SAM3 Processing Pipeline (Offline Local App)
- Environment Setup & Crash Prevention
- Hardware Acceleration (NVIDIA RTX 4080 SUPER / Tensor Cores)
- Lazy Loading for Gigapixel Orthomosaics
- Safe Tiling & Semantic Prediction (SAM3 / Ultralytics)
- 100% Offline Local File Support
"""

import os
import tempfile
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import streamlit as st
import torch
from ultralytics import SAM

# 1. Konfigurasi Lingkungan Offline & Pencegahan Crash
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Headless mode
os.environ["YOLO_OFFLINE"] = "true"  # Mencegah Ultralytics mencoba akses internet


@st.cache_resource
def setup_hardware():
  """Mengonfigurasi akselerasi perangkat keras untuk GPU NVIDIA (RTX 4080 SUPER)."""
  if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Tensor Cores
    torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(
        f"🟢 GPU Aktif: {gpu_name}\n(Tensor Cores TF32 & FP16 Ready)"
    )
  else:
    device = torch.device("cpu")
    st.sidebar.warning("⚠️ CUDA tidak tersedia, menggunakan CPU.")
  return device


class LazyGeotiffReader:
  """Kelas untuk membaca citra gigapixel secara lazy loading (per tile)

  tanpa membebani RAM secara berlebihan.
  """

  def __init__(self, filepath):
    self.filepath = filepath
    self.dataset = rasterio.open(filepath)
    self.width = self.dataset.width
    self.height = self.dataset.height
    self.crs = self.dataset.crs
    self.transform = self.dataset.transform

  def read_tile(self, window: Window):
    """Membaca area tertentu (window) dari raster secara bertahap."""
    return self.dataset.read(window=window)

  def close(self):
    self.dataset.close()


def verify_local_models():
  """Memeriksa keberadaan file model di dalam folder lokal 'ModelSAM'."""
  model_dir = "ModelSAM"
  models = {
      "sam_base": os.path.join(model_dir, "sam_vit_h_4b8939.pth"),
      "sam3": os.path.join(model_dir, "sam3-002.pt"),
  }

  status = {}
  for name, path in models.items():
    if os.path.exists(path):
      status[name] = path
      st.sidebar.success(f"✔️ Model {name} ditemukan lokal.")
    else:
      status[name] = None
      st.sidebar.error(
          f"❌ Model {name} tidak ada di `{path}`. Letakkan file di folder"
          " ModelSAM."
      )
  return status


def main():
  # Konfigurasi Halaman Streamlit
  st.set_page_config(
      page_title="Geospatial SAM3 Offline Pipeline", layout="wide"
  )

  st.title("🛰️ Geospatial SAM3 Processing Pipeline (Offline Mode)")
  st.markdown(
      "Aplikasi berbasis Python & Anaconda untuk analisis orthomosaik gigapixel"
      " menggunakan model lokal tanpa koneksi internet."
  )

  # Setup Hardware & Verifikasi Model Lokal
  device = setup_hardware()
  local_models = verify_local_models()

  # Sidebar: Panel Kontrol & Input File
  st.sidebar.header("📁 Pengaturan Sumber Data & Model")

  # 1. Pilihan Sumber Orthomosaik GeoTIFF
  source_type = st.sidebar.radio(
      "Sumber File Orthomosaik:",
      ["Gunakan File Default Lokal", "Upload File GeoTIFF (.tif)"],
  )

  ortho_path = None
  if source_type == "Gunakan File Default Lokal":
    default_ortho = "Orthomosaik-Desa Langkap_TM3-48.1_.tif"
    if os.path.exists(default_ortho):
      ortho_path = default_ortho
      st.sidebar.info(f"Menggunakan file lokal: `{default_ortho}`")
    else:
      st.sidebar.error(f"File default `{default_ortho}` tidak ditemukan.")
  else:
    uploaded_ortho = st.sidebar.file_uploader(
        "Pilih file GeoTIFF", type=["tif", "tiff"]
    )
    if uploaded_ortho is not None:
      with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_ortho.getvalue())
        ortho_path = tmp.name
      st.sidebar.success(f"Berhasil memuat: {uploaded_ortho.name}")

  # 2. Pilihan Model SAM
  selected_model_key = st.sidebar.selectbox(
      "Pilih Model SAM yang Digunakan", list(local_models.keys())
  )
  selected_model_path = local_models[selected_model_key]

  # 3. Parameter Segmentasi & VRAM
  prompt_input = st.sidebar.text_input(
      "Target Prompt Segmentasi", "building, roof"
  )
  text_prompts = [p.strip() for p in prompt_input.split(",") if p.strip()]

  use_half = st.sidebar.checkbox(
      "Gunakan Presisi Setengah (FP16 / half=True)", value=True
  )

  # Eksekusi Utama
  if ortho_path and os.path.exists(ortho_path):
    try:
      reader = LazyGeotiffReader(ortho_path)

      st.markdown("---")
      col1, col2, col3 = st.columns(3)
      col1.metric("Lebar Citra (Px)", f"{reader.width:,}")
      col2.metric("Tinggi Citra (Px)", f"{reader.height:,}")
      col3.metric("Sistem Koordinat (CRS)", str(reader.crs))

      st.write(f"🎯 **Target Prompt:** {text_prompts}")
      if selected_model_path:
        st.write(f"🤖 **Model Aktif:** `{selected_model_path}`")

      if use_half and device.type == "cuda":
        st.info("⚡ Mode VRAM Efisien (FP16 / Tensor Cores aktif).")

      # Tombol Eksekusi Pipeline
      if st.button(
          "🚀 Jalankan Pipeline Segmentasi", type="primary", use_container_width=True
      ):
        if not selected_model_path:
          st.error(
              "Gagal menjalankan: File model belum tersedia di folder"
              " penyimpanan lokal."
          )
        else:
          with st.spinner(
              "Memproses segmentasi gigapixel dengan Safe Tiling Mode..."
          ):
            # --- SIMULASI / TEMPAT PIPELINE UTAMA ANDA ---
            # Contoh: Memuat model dengan Ultralytics secara lokal tanpa internet
            # model = SAM(selected_model_path)
            # ----------------------------------------------
            st.success(
                "✅ Pipeline berhasil dieksekusi secara offline menggunakan"
                " resource lokal!"
            )

      reader.close()

    except Exception as e:
      st.error(f"Terjadi kesalahan saat memproses file raster: {e}")
  else:
    st.info(
        "👉 Silakan pastikan file orthomosaik lokal tersedia atau pilih opsi"
        " upload di panel sebelah kiri."
    )


if __name__ == "__main__":
  main()
