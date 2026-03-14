import streamlit as st
import requests
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

# --- 1. AYARLAR VE GÜVENLİK (SECRETS) ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Hata: Secrets panelinde 'GROQ_API_KEY' tanımlı değil!")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Algoritmik Düşünme Atölyesi", layout="wide")

# --- 2. AKADEMİK PROTOKOLLER (TAM METİN - EKSİKSİZ) ---
BASAMAK_TALIMATLARI = {
"1. Ayrıştırma": """Amaç: Öğrencinin problemi alt bileşenlerine ayırmasını sağlamak.
Zorunlu davranışlar:
- Verilenleri tek tek fark ettir.
- İstenen bilgiyi açıkça söylet.
- Problemi en az iki alt probleme ayırtmasını iste.
- Sayısal ilişkileri tanımlat.
Yasaklar: Çözüm yolu önermek, işlem yaptırmak, ipucu vermek.""",

"2. Soyutlama": """Amaç: Problemin matematiksel yapısını ortaya çıkarmak.
Zorunlu davranışlar:
- Bu problemin hangi tür probleme benzediğini sordur.
- Önceki bilgileri hatırlat.
- Gereksiz bilgileri ayırt ettir.
- Örüntü veya genel yapı düşündür.
Yasaklar: Kavramı (EBOB/EKOK vb.) doğrudan söylemek, strateji belirtmek.""",

"3. Algoritma Tasarımı": """Amaç: Öğrencinin kendi çözüm planını oluşturmasını sağlamak.
Zorunlu davranışlar:
- Adım adım plan oluşturmasını iste.
- Neden bu yöntemi seçtiğini sorgula.
- Başka bir yöntem mümkün mü diye düşündür.
Yasaklar: İşlem adımlarını doğrudan vermek, hesap sonucu söylemek.""",

"4. Hata Ayıklama": """Amaç: Öğrencinin çözümünü kontrol etmesini sağlamak.
Zorunlu davranışlar:
- Sonucun problem koşullarını sağlayıp sağlamadığını sorgulat.
- Alternatif doğrulama yolu düşündür.
- Mantıksal tutarlılık kontrolü yaptır.
Yasaklar: Sonucu doğru/yanlış diye kesin belirtmek, doğru cevabı ima etmek."""
}

METABILISSEL_SORULAR = {
    "1. Ayrıştırma": "Bu problemi parçalara ayırırken en çok hangi bilgi dikkatini çekti?",
    "2. Soyutlama": "Bu soruda özellikle dikkat etmemiz gereken noktalar nelerdir?",
    "3. Algoritma Tasarımı": "Çözüm adımlarını planlarken nasıl bir yol izledin?",
    "4. Hata Ayıklama": "Bulduğun sonucun mantıklı olduğundan nasıl emin oldun?"
}

# --- 3. VERİ SİSTEMİ ---
def log_kaydet(data):
    dosya = "tez_verileri_final.csv"
    df = pd.DataFrame([data])
    df.to_csv(dosya, mode='a', index=False, header=not os.path.isfile(dosya), encoding="utf-8-sig")

# --- 4. SESSION STATE (BELLEK) ---
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {s: [] for s in BASAMAK_TALIMATLARI.keys()}
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "current_step" not in st.session_state:
    st.session_state.current_step = "1. Ayrıştırma"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("👨‍🏫 Araştırma Paneli")
    mode = st.selectbox("Giriş Türü:", ["Öğrenci Girişi", "Öğretmen (Admin)"])
    
    if mode == "Öğretmen (Admin)":
        sifre = st.text_input("Şifre:", type="password")
        if sifre == "tez2024":
            st.success("Admin Paneli Aktif")
            if os.path.isfile("tez_verileri_final.csv"):
                try:
                    df_csv = pd.read_csv("tez_verileri_final.csv", on_bad_lines='skip')
                    st.dataframe(df_csv.tail(15))
                    csv_indir = df_csv.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 Verileri İndir", csv_indir, "tez_verileri.csv", "text/csv")
                except: st.error("Veri okuma hatası!")
        st.stop()
        
    student_id = st.text_input("Öğrenci No:", placeholder="Örn: 123")
    if not student_id:
        st.warning("Lütfen giriş yapın.")
        st.stop()
        
    st.divider()
    steps = list(BASAMAK_TALIMATLARI.keys())
    sel = st.radio("Aşamalar:", steps, index=steps.index(st.session_state.current_step))
    if sel != st.session_state.current_step:
        st.session_state.current_step = sel
        st.rerun()

# --- 6. ANA EKRAN ---
st.title("🎯 Algoritmik Düşünme Atölyesi")
st.write(f"### Mevcut Basamak: {st.session_state.current_step}")

if st.session_state.uploaded_file_data is None:
    up = st.file_uploader("Soru Fotoğrafı Yükle", type=["png", "jpg", "jpeg"])
    if up:
        st.session_state.uploaded_file_data = up
        st.rerun()
