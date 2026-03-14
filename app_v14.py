import streamlit as st
import requests
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

# --- 1. AYARLAR VE GÜVENLİK ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Hata: Secrets panelinde GROQ_API_KEY bulunamadı!")
    st.stop()

GROQ_URL = "https://api.api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Algoritmik Düşünme Atölyesi", layout="wide")

# --- 2. AKADEMİK PROTOKOLLER (TAM METİN - EKSİKSİZ) ---
BASAMAK_TALIMATLARI = {
"1. Ayrıştırma": """Amaç: Öğrencinin problemi alt bileşenlerine ayırmasını sağlamak.
Zorunlu davranışlar:
- Verilenleri tek tek fark ettir (Örn: "Sence bu soruda bize hangi bilgiler verilmiş?").
- İstenen bilgiyi açıkça söylet.
- Problemi en az iki alt probleme ayırtmasını iste.
- Sayısal ilişkileri tanımlat.
Yasak:
- Çözüm yolu önermek. - İşlem yaptırmak. - Sonuca yaklaşan ipucu vermek.""",

"2. Soyutlama": """Amaç: Problemin matematiksel yapısını ortaya çıkarmak.
Zorunlu davranışlar:
- Bu problemin hangi tür probleme (EBOB, EKOK, Üslü Sayı vb.) benzediğini sordur.
- Önceki bilgileri hatırlat.
- Gereksiz bilgileri ayırt ettir.
- Eğer uygunsa örüntü veya genel yapı düşündür.
Yasak:
- EKOK/EBOB gibi kavramı doğrudan söylemek. - Stratejiyi açıkça belirtmek.""",

"3. Algoritma Tasarımı": """Amaç: Öğrencinin kendi çözüm planını oluşturmasını sağlamak.
Zorunlu davranışlar:
- Adım adım plan oluşturmasını iste.
- Neden bu yöntemi seçtiğini sorgula.
- Başka bir yöntem mümkün mü diye düşündür.
- Çözümün genellenebilirliğini sorgula.
Yasak:
- İşlem adımlarını doğrudan vermek. - Hesap sonucu söylemek.""",

"4. Hata Ayıklama": """Amaç: Öğrencinin çözümünü kontrol etmesini sağlamak.
Zorunlu davranışlar:
- Sonucun problem koşullarını sağlayıp sağlamadığını sorgulat.
- Alternatif doğrulama yolu düşündür.
- Mantıksal tutarlılık kontrolü yaptır (Örn: "Bulduğun bu sonuç sorudaki mantığa uyuyor mu?").
Yasak:
- Sonucu doğru veya yanlış diye kesin olarak belirtmek. - Doğru cevabı ima etmek."""
}

METABILISSEL_SORULAR = {
    "1. Ayrıştırma": "Bu problemi parçalara ayırırken en çok hangi bilgi dikkatini çekti?",
    "2. Soyutlama": "Bu soruda özellikle dikkat etmemiz gereken noktalar nelerdir? Gereksiz olduğunu düşündüğün yerler var mı?",
    "3. Algoritma Tasarımı": "Çözüm adımlarını planlarken nasıl bir yol izledin?",
    "4. Hata Ayıklama": "Bulduğun sonucun mantıklı olduğundan nasıl emin oldun? Farklı bir strateji kullanmayı düşünsen ne yapardın?"
}

# --- 3. VERİ SİSTEMİ ---
def log_kaydet(data):
    dosya = "tez_verileri_final.csv"
    df = pd.DataFrame([data])
    df.to_csv(dosya, mode='a', index=False, header=not os.path.isfile(dosya), encoding="utf-8-sig")

# --- 4. SESSION STATE ---
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
                    df_csv = pd.read_csv("tez_verileri_final.csv", sep=None, engine='python', on_bad_lines='skip')
                    st.dataframe(df_csv.tail(20))
                    st.download_button("📥 Verileri İndir", df_csv.to_csv(index=False).encode('utf-8-sig'), "tez_verileri.csv", "text/csv")
                except: st.error("Veri okuma hatası!")
        st.stop()
        
    student_id = st.text_input("Öğrenci No:", placeholder="Örn: 8A-123")
    if not student_id:
        st.warning("Giriş yapın.")
        st.stop()
        
    st.divider()
    steps = list(BASAMAK_TALIMATLARI.keys())
    sel = st.radio("Aşamalar:", steps, index=steps.index(st.session_state.current_step))
    if sel != st.session_state.current_step:
        st.session_state.current_step = sel
        st.rerun()

# --- 6. ANA EKRAN ---
st.title("🎯 Algoritmik Düşünme İstasyonu")
st.write(f"### Mevcut Basamak: {st.session_state.current_step}")

if st.session_state.uploaded_file_data is None:
    up = st.file_uploader("
