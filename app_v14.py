import streamlit as st
import requests
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

# --- 1. AYARLAR & SECRETS ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Hata: Secrets panelinde GROQ_API_KEY bulunamadı!")
    st.stop()

st.set_page_config(page_title="LGS Strateji Atölyesi", layout="wide")

# --- 2. TAM PROTOKOLLER (KISALTILMADI) ---
BASAMAK_TALIMATLARI = {
"1. Ayrıştırma": "Amaç: Problemi alt bileşenlerine ayırmak. Zorunlu: Verilenleri fark ettir, isteneni söylet. Yasak: Çözüm yolu, işlem, ipucu.",
"2. Soyutlama": "Amaç: Matematiksel yapıyı ortaya çıkarmak. Zorunlu: Benzer problemleri hatırlat, gereksiz bilgiyi ayırt ettir. Yasak: Kavramı (EBOB/EKOK vb.) söylemek.",
"3. Algoritma Tasarımı": "Amaç: Kendi çözüm planını oluşturmak. Zorunlu: Adım adım plan, neden bu yöntem? Yasak: İşlem adımlarını vermek.",
"4. Hata Ayıklama": "Amaç: Çözümü kontrol etmek. Zorunlu: Koşulları sorgulat, mantık kontrolü. Yasak: Doğru/Yanlış demek."
}

# --- 3. VERİ SİSTEMİ ---
def log_kaydet(data):
    dosya = "tez_verileri_8sinif_final.csv"
    df = pd.DataFrame([data])
    if not os.path.isfile(dosya):
        df.to_csv(dosya, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(dosya, mode='a', index=False, header=False, encoding="utf-8-sig")

# --- 4. SESSION STATE (BELLEK YÖNETİMİ) ---
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = {step: None for step in BASAMAK_TALIMATLARI.keys()}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {step: [] for step in BASAMAK_TALIMATLARI.keys()}
if "current_step" not in st.session_state:
    st.session_state.current_step = "1. Ayrıştırma"
if "img_data" not in st.session_state:
    st.session_state.img_data = None

# --- 5. YAN PANEL ---
with st.sidebar:
    st.title("👨‍🏫 Panel")
    mode = st.selectbox("Giriş:", ["Öğrenci", "Öğretmen"])
    if mode == "Öğretmen":
        if st.text_input("Şifre:", type="password") == "tez2024":
            if os.path.isfile("tez_verileri_8sinif_final.csv"):
                df = pd.read_csv("tez_verileri_8sinif_final.csv")
                st.download_button("Veriyi İndir", df.to_csv(index=False), "tez.csv")
                st.dataframe(df)
        st.stop()
    
    student_id = st.text_input("Öğrenci No:")
    if not student_id: st.stop()
    
    st.divider()
    steps = list(BASAMAK_TALIMATLARI.keys())
    selected = st.radio("Adım Seç:", steps, index=steps.index(st.session_state.current_step))
    if selected != st.session_state.current_step:
        st.session_state.current_step = selected
        st.rerun()

# --- 6. ANA EKRAN ---
st.title("🎯 LGS Strateji İstasyonu")

# Fotoğraf Bölümü
if st.session_state.img_data is None:
    up = st.file_uploader("Soru Yükle", type=["jpg", "png"])
    if up: st.session_state.img_data = up; st.rerun()
else:
    st.image(st.session_state.img_data, width=500)
    if st.button("❌ Soruyu Değiştir"): 
        st.session_state.img_data = None; st.rerun()

st.divider()
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader(f"🖌️ Çizim: {st.session_state.current_step}")
    
    # ÇİZİM SORUNUNU ÇÖZEN KRİTİK BİLEŞEN
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        drawing_mode="freedraw",
        # İŞTE BURASI: Çizim verisini session_state'den zorla okutuyoruz
        initial_drawing=st.session_state.canvas_data[st.session_state.current_step],
        update_streamlit=True, # Her harekette veriyi Streamlit'e gönder
        key=f"canvas_fix_{st.session_state.current_step}", # Basamağa özel sabit anahtar
    )
    
    # Çizimi anlık olarak belleğe kaydet (Sessizce)
    if canvas_result.json_data is not None:
        st.session_state.canvas_data[st.session_state.current_step] = canvas_result.json_data

with col2:
    st.subheader("💬 Sohbet")
    current_chat = st.session_state.chat_history[st.session_state.current_step]
    
    chat_container = st.container(height=350)
    for m in current_chat:
        chat_container.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Buraya yaz..."):
        # Kullanıcı mesajı
        current_chat.append({"role": "user", "content": prompt})
        log_kaydet({"tarih": datetime.now(), "id": student_id, "adim": st.session_state.current_step, "rol": "user", "icerik": prompt})
        
        # AI Yanıtı
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile", 
                      "messages": [{"role": "system", "content": BASAMAK_TALIMATLARI[st.session_state.current_step]}] + current_chat})
            ans = r.json()['choices'][0]['message']['content']
            current_chat.append({"role": "assistant", "content": ans})
            log_kaydet({"tarih": datetime.now(), "id": student_id, "adim": st.session_state.current_step, "rol": "assistant", "icerik": ans})
            st.session_state.chat_history[st.session_state.current_step] = current_chat
            st.rerun()
        except: st.error("Bot hatası.")
