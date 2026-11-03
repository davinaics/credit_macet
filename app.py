"""Aplikasi Prediksi Status Kredit Nasabah (Lancar / Macet) - Streamlit."""
import pandas as pd
import streamlit as st

from utils.model_loader import load_artifacts, predict

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Status Kredit", page_icon="💳", layout="wide")

PRIMARY, BG, DANGER = "#2E7D32", "#F1F8E9", "#C62828"

CUSTOM_CSS = f"""
<style>
.stApp {{ background-color: {BG}; }}
section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #DCEDC8; }}
h1, h2, h3 {{ color: {PRIMARY}; }}
.hero {{ background: linear-gradient(135deg, {PRIMARY}, #66BB6A); color: #fff; padding: 1.6rem 2rem;
        border-radius: 20px; box-shadow: 0 6px 18px rgba(46,125,50,.25); margin-bottom: 1.2rem; }}
.hero h1 {{ color: #fff; margin: 0; font-size: 1.9rem; }}
.hero p {{ margin: .3rem 0 0; opacity: .92; }}
.section-title {{ color: {PRIMARY}; font-weight: 700; font-size: 1.05rem; margin: 0 0 .4rem; }}
div[data-testid="stForm"] {{ background: #fff; border: none; border-radius: 18px; padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,.07); }}
.card {{ background: #fff; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,.07); }}
.result-card {{ background: #fff; border-radius: 20px; padding: 1.8rem; text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,.09); }}
.result-label {{ font-size: 2.6rem; font-weight: 800; letter-spacing: 2px; margin: .2rem 0; }}
.result-sub {{ color: #607D8B; font-size: .9rem; text-transform: uppercase; letter-spacing: 1px; }}
.badge {{ display: inline-block; padding: .35rem 1rem; border-radius: 999px; color: #fff; font-weight: 700; }}
.bar-bg {{ background: #ECEFF1; border-radius: 999px; height: 12px; overflow: hidden; margin-top: .4rem; }}
.bar-fg {{ height: 12px; border-radius: 999px; }}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
        background-color: {PRIMARY}; color: #fff; border: none; border-radius: 12px;
        font-weight: 600; padding: .6rem 1.2rem; }}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #1B5E20; color: #fff; }}
table.summary {{ width: 100%; border-collapse: collapse; }}
table.summary td {{ padding: .45rem .6rem; border-bottom: 1px solid #EEE; }}
table.summary td:first-child {{ color: #607D8B; width: 55%; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load model (di-cache supaya tidak dimuat ulang tiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def get_artifacts():
    return load_artifacts()


def rupiah(x: float) -> str:
    """Format angka ke Rupiah, contoh: Rp 5.000.000"""
    return "Rp " + f"{x:,.0f}".replace(",", ".")


# ----------------------------------------------------------------------------
# Halaman: Prediksi
# ----------------------------------------------------------------------------
def render_result(result: dict, summary: dict) -> None:
    is_lancar = result["label"] == "LANCAR"
    color = PRIMARY if is_lancar else DANGER
    icon = "✅" if is_lancar else "⚠️"
    conf = result["confidence"] * 100
    desc = ("Nasabah diprediksi mampu memenuhi kewajiban pembayaran dengan baik."
            if is_lancar else
            "Nasabah diprediksi berisiko gagal bayar. Disarankan analisis lebih lanjut.")

    st.markdown(
        f"""
        <div class="result-card" style="border-top: 8px solid {color};">
            <div class="result-sub">Hasil Prediksi Status Kredit</div>
            <div class="result-label" style="color:{color};">{icon} {result['label']}</div>
            <span class="badge" style="background:{color};">Confidence Score: {conf:.1f}%</span>
            <p style="color:#455A64; margin-top:1rem;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2 = st.columns(2)
    for col, name, clr in ((c1, "LANCAR", PRIMARY), (c2, "MACET", DANGER)):
        p = result["probabilities"].get(name, 0.0) * 100
        col.markdown(
            f"""<div class="card"><b style="color:{clr};">Probabilitas {name}</b>
            <div style="font-size:1.6rem;font-weight:700;color:{clr};">{p:.1f}%</div>
            <div class="bar-bg"><div class="bar-fg" style="width:{p}%;background:{clr};"></div></div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("📋 Ringkasan Data yang Diinput"):
        rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in summary.items())
        st.markdown(f'<table class="summary">{rows}</table>', unsafe_allow_html=True)


def page_prediksi() -> None:
    st.markdown(
        """<div class="hero"><h1>💳 Aplikasi Prediksi Status Kredit Nasabah (Lancar / Macet)</h1>
        <p>Masukkan data nasabah untuk memprediksi status kredit menggunakan model Decision Tree.</p></div>""",
        unsafe_allow_html=True,
    )

    try:
        model, columns = get_artifacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    with st.form("form_prediksi"):
        st.markdown('<p class="section-title">👤 Data Nasabah</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        jk_label = c1.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        umur = c2.number_input("Umur (tahun)", min_value=18, max_value=70, value=35, step=1)

        st.markdown('<p class="section-title">💰 Data Pinjaman</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        jml_pinjaman = c1.number_input("Jumlah Pinjaman / Plafon (Rp)", min_value=1_000_000,
                                        max_value=500_000_000, value=5_000_000, step=1_000_000)
        jkw = c2.number_input("Jangka Waktu / JKW (bulan)", min_value=1, max_value=60, value=12, step=1)
        jenis_pinjaman = c3.selectbox("Jenis Pinjaman", [301, 302, 303, 304, 305])

        c1, c2, c3 = st.columns(3)
        angsuran = c1.number_input("Angsuran per Bulan (Rp)", min_value=0, max_value=100_000_000,
                                    value=450_000, step=10_000)
        saldo = c2.number_input("Saldo Nominatif (Rp)", min_value=0, max_value=500_000_000,
                                value=3_000_000, step=100_000)
        col_val = c3.selectbox("Kolektibilitas (COL)", [1, 2],
                                format_func=lambda x: {1: "1 - Lancar", 2: "2 - Dalam Perhatian Khusus"}[x])

        st.markdown('<p class="section-title">📉 Tunggakan</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        tung_pokok = c1.number_input("Tunggakan Pokok (Rp)", min_value=0, max_value=500_000_000,
                                    value=0, step=100_000)
        tung_bunga = c2.number_input("Tunggakan Bunga (Rp)", min_value=0, max_value=100_000_000,
                                    value=0, step=10_000)

        st.markdown('<p class="section-title">🏦 Klasifikasi Bank Indonesia</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        sektor = c1.selectbox("Sektor Ekonomi BI", [6000.0, 8000.0, 9990.0],
                                format_func=lambda x: f"{x:.0f}")
        # 0 = kategori dasar (baseline) yang di-drop saat training (drop_first=True)
        gol_debitur = c2.selectbox("Golongan Debitur BI", [0, 874, 876],
                                    format_func=lambda x: "Lainnya" if x == 0 else str(x))
        gol_penjamin = c3.selectbox("Golongan Penjamin BI", [0, 800, 835, 874, 875],
                                    format_func=lambda x: "Tanpa penjamin / lainnya (0)" if x == 0 else str(x))

        submitted = st.form_submit_button("🔍 Prediksi Status Kredit")

    if not submitted:
        st.info("Lengkapi form di atas lalu klik **Prediksi Status Kredit**.")
        return

    # Nilai harus sama persis dengan format saat training
    raw = {
        "umur": float(umur),
        "jml_pinjaman": float(jml_pinjaman),
        "jkw": float(jkw),
        "jml_angsuran_per_bulan": float(angsuran),
        "saldo_nominatif": float(saldo),
        "tunggakan_pokok": float(tung_pokok),
        "tunggakan_bunga": float(tung_bunga),
        "jenis_kelamin": "L" if jk_label == "Laki-laki" else "P",
        "jenis_pinjaman": int(jenis_pinjaman),
        "bi_sektor_ekonomi": float(sektor),
        "bi_golongan_debitur": int(gol_debitur),
        "bi_gol_penjamin": int(gol_penjamin),
        "col": int(col_val),
    }

    try:
        result = predict(model, columns, raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Terjadi kesalahan saat prediksi: {exc}")
        return

    summary = {
        "Jenis Kelamin": jk_label, "Umur": f"{umur} tahun",
        "Jumlah Pinjaman": rupiah(jml_pinjaman), "Jangka Waktu": f"{jkw} bulan",
        "Jenis Pinjaman": jenis_pinjaman, "Angsuran per Bulan": rupiah(angsuran),
        "Saldo Nominatif": rupiah(saldo), "Kolektibilitas (COL)": col_val,
        "Tunggakan Pokok": rupiah(tung_pokok), "Tunggakan Bunga": rupiah(tung_bunga),
        "Sektor Ekonomi BI": f"{sektor:.0f}",
        "Golongan Debitur BI": "Lainnya" if gol_debitur == 0 else gol_debitur,
        "Golongan Penjamin BI": "Tanpa penjamin / lainnya" if gol_penjamin == 0 else gol_penjamin,
    }
    st.markdown("### Hasil Prediksi")
    render_result(result, summary)


# ----------------------------------------------------------------------------
# Halaman: Tentang Dataset & Model
# ----------------------------------------------------------------------------
def page_tentang() -> None:
    st.markdown(
        """<div class="hero"><h1>📚 Tentang Dataset & Model</h1>
        <p>Informasi singkat mengenai data dan model yang digunakan.</p></div>""",
        unsafe_allow_html=True,
    )
    try:
        model, columns = get_artifacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    c1, c2 = st.columns(2)
    c1.markdown(
        """<div class="card"><p class="section-title">Dataset</p>
        <ul><li>Data kredit nasabah (±766 baris) dengan target <b>status kredit</b>: LANCAR / MACET</li>
        <li>Pembersihan: standardisasi jenis kelamin, penanganan umur &amp; JKW tidak wajar,
        saldo negatif, serta imputasi median / modus</li>
        <li>Encoding kategorikal: <code>pd.get_dummies(drop_first=True)</code></li></ul></div>""",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""<div class="card"><p class="section-title">Model</p>
        <ul><li>Algoritma: <b>{type(model).__name__}</b></li>
        <li>Jumlah fitur: <b>{len(columns)}</b></li>
        <li>Label: LANCAR = 0, MACET = 1</li>
        <li>Pembagian data: 80% training, 20% testing (stratified)</li></ul></div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        st.markdown("#### Tingkat Kepentingan Fitur")
        df_imp = pd.DataFrame({"Fitur": columns, "Importance": imp}).sort_values(
            "Importance", ascending=False).set_index("Fitur")
        st.bar_chart(df_imp, color=PRIMARY)

    st.caption("⚠️ Prediksi bersifat pendukung keputusan dan bukan pengganti analisis kredit oleh petugas.")


# ----------------------------------------------------------------------------
# Navigasi sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:{PRIMARY};'>💳 Credit Scoring</h2>", unsafe_allow_html=True)
    menu = st.radio("Menu Navigasi", ["Prediksi Status Kredit", "Tentang Dataset & Model"])
    st.divider()
    st.caption("Model: Decision Tree Classifier")

if menu == "Prediksi Status Kredit":
    page_prediksi()
else:
    page_tentang()
