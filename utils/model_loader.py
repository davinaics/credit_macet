"""Helper untuk memuat model dan melakukan inferensi prediksi status kredit."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"

# Mapping label saat training: LANCAR = 0, MACET = 1
LABELS = {0: "LANCAR", 1: "MACET"}

# Kolom kategorikal yang di-encode dengan pd.get_dummies saat training
CATEGORICAL_COLS = [
    "jenis_kelamin",
    "jenis_pinjaman",
    "bi_sektor_ekonomi",
    "bi_golongan_debitur",
    "bi_gol_penjamin",
    "col",
]


def load_artifacts():
    """Muat model dan daftar kolom fitur hasil training."""
    for path in (MODEL_PATH, COLUMNS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"File '{path.name}' tidak ditemukan di {path.parent}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        columns = list(pickle.load(f))
    return model, columns


def build_features(raw: dict, columns: list[str]) -> pd.DataFrame:
    """Ubah input mentah user menjadi DataFrame berfitur persis seperti saat training.

    Catatan: drop_first TIDAK dipakai di sini (pada 1 baris data semua dummy
    akan terhapus). Kategori dasar (baseline) otomatis hilang saat reindex
    karena kolomnya tidak ada di `columns`.
    """
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    # Samakan kolom & urutan dengan training; kolom yang tidak ada diisi 0
    df = df.reindex(columns=columns, fill_value=0)
    return df.astype(float)


def predict(model, columns: list[str], raw: dict) -> dict:
    """Jalankan prediksi dan kembalikan label, confidence, serta probabilitas tiap kelas."""
    X = build_features(raw, columns)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    prob_by_class = {LABELS[int(c)]: float(p) for c, p in zip(model.classes_, proba)}
    label = LABELS[pred]
    return {"label": label, "confidence": prob_by_class[label], "probabilities": prob_by_class}
