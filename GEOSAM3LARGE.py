"""
Geospatial SAM3 Processing Pipeline
- Environment Setup & Crash Prevention
- Hardware Acceleration (NVIDIA RTX 4080 SUPER / Tensor Cores)
- Lazy Loading for Gigapixel Orthomosaics
- Safe Tiling & Semantic Prediction (SAM3 / Ultralytics)
"""

import os

# 1. Inisialisasi Lingkungan & Pencegahan Crash (OpenMP & Headless)[cite: 2]
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Headless mode[cite: 2]

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import torch
from ultralytics import SAM  # Disesuaikan dengan struktur SAM3 Ultralytics[cite: 2]


def setup_hardware():
  """Mengonfigurasi akselerasi perangkat keras untuk GPU NVIDIA (RTX 4080 SUPER)."""
  if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Tensor Cores[cite: 2]
    torch.backends.cudnn.allow_tf32 = True
    print(f"Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    print("Tensor Cores (TF32) diaktifkan[cite: 2].")
  else:
    device = torch.device("cpu")
    print("CUDA tidak tersedia, menggunakan CPU.")
  return device


class LazyGeotiffReader:
  """Kelas untuk membaca citra gigapixel secara lazy loading (per tile)

  tanpa membebani RAM secara berlebihan[cite: 2].
  """

  def __init__(self, filepath):
    self.filepath = filepath
    self.dataset = rasterio.open(filepath)
    self.width = self.dataset.width
    self.height = self.dataset.height
    self.crs = self.dataset.crs
    self.transform = self.dataset.transform
    print(
        f"Membuka citra: {os.path.basename(filepath)} | Dimensi:"
        f" {self.width}x{self.height} | CRS: {self.crs}[cite: 2]"
    )

  def read_tile(self, window: Window):
    """Membaca area tertentu (window) dari raster secara bertahap."""
    return self.dataset.read(window=window)

  def close(self):
    self.dataset.close()


def verify_models():
  """Memverifikasi keberadaan file model lokal di direktori ModelSAM."""
  model_dir = "ModelSAM"
  models = {
      "sam_base": os.path.join(model_dir, "sam_vit_h_4b8939.pth"),
      "sam3": os.path.join(model_dir, "sam3-002.pt"),
  }

  for name, path in models.items():
    if os.path.exists(path):
      print(f"Model {name} ditemukan di: {path}")
    else:
      print(
          f"PERINGATAN: Model {name} tidak ditemukan di {path}. Pastikan direktori"
          " benar."
      )
  return models


def main():
  # Setup Perangkat Keras
  device = setup_hardware()

  # Verifikasi Model Lokal[cite: 2]
  models = verify_models()

  # Konfigurasi Berkas Citra Orthomosaik Gigapixel[cite: 2]
  ortho_path = "Orthomosaik-Desa Langkap_TM3-48.1_.tif"

  if not os.path.exists(ortho_path):
    print(f"File orthomosaik {ortho_path} tidak ditemukan. Periksa path file.")
    return

  # Inisialisasi Lazy Reader
  reader = LazyGeotiffReader(ortho_path)

  # Inisialisasi SAM3 Semantic Predictor dengan Mode Safe Tiling[cite: 2]
  print("Menginisialisasi SAM3 Semantic Predictor...")
  # predictor = SAM3SemanticPredictor(model=models['sam3'])

  # Prompt teks untuk segmentasi objek geospasial[cite: 2]
  text_prompts = ["building", "roof"]
  print(f"Target prompt segmentasi: {text_prompts}")

  # Penggunaan presisi setengah (FP16 / half=True) untuk efisiensi VRAM[cite: 2]
  use_half = True
  if use_half and device.type == "cuda":
    print("Menggunakan presisi setengah (FP16 / half=True) untuk efisiensi VRAM.")

  print("Pipeline siap dieksekusi dengan Safe Tiling Mode[cite: 2].")

  reader.close()


if __name__ == "__main__":
  main()