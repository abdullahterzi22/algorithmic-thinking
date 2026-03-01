import streamlit as st
import requests
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

# --- 1. GÜVENLİ AYARLAR ---
try:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    else:
        st.error("Hata: GROQ_API_KEY Secrets panelinde bulunamadı!")
        st.stop()
except Exception as e:
    st.error(f"Secrets erişim hatası: {e}")
    st.stop()

st.set_page_config(page_title="LGS Matematik Strateji Atölyesi", layout="wide")

# --- 2. AKADEMİK PROTOKOLLER (TAM METİN - HİÇ KISALTILMADI) ---
BASAMAK_TALIMATLARI = {
"1. Ayrıştırma": """
Amaç: Öğrencinin problemi alt bileşenlerine ayırmasını sağlamak.
Zorunlu davranışlar:
- Verilenleri tek tek fark ettir.
- İstenen bilgiyi açıkça söylet.
- Problemi en az iki alt probleme ayırtmasını iste.
- Sayısal ilişkileri tanımlat.
Yasak:
- Çözüm yolu önermek.
- İşlem yaptırmak.
- Sonuca yaklaşan ipucu vermek.
""",
"2. Soyutlama": """
Amaç: Problemin matematiksel yapısını ortaya çıkarmak.
Zorunlu davranışlar:
- Bu problemin hangi tür probleme benzediğini sordur.
- Önceki bilgileri hatırlat.
- Gereksiz bilgileri ayırt ettir.
- Eğer uygunsa örüntü veya genel yapı düşündür.
Yasak:
- EKOK/EBOB gibi kavramı doğrudan söylemek.
- Stratejiyi açıkça belirtmek.
""",
"3. Algoritma Tasarımı": """
Amaç: Öğrencinin kendi çözüm planını oluşturmasını sağlamak.
Zorunlu davranışlar:
- Adım adım plan oluşturmasını iste.
- Neden bu yöntemi seçtiğini sorgula.
- Başka bir yöntem mümkün mü diye düşündür.
- Çözümün genellenebilirliğini sorgula.
Yasak:
- İşlem adımlarını vermek.
- Hesap sonucu söylemek.
""",
"4. Hata Ayıklama": """
Amaç: Öğrencinin çözümünü kontrol etmesini sağlamak.
Zorunlu davranışlar:
- Sonucun problem koşullarını sağlayıp sağlamadığını sorgulat.
- Alternatif doğrulama yolu düşündür.
- Mantıksal tutarlılık kontrolü yaptır.
Yasak:
- Sonucu doğru/yanlış diye belirtmek.
- Doğru cevabı ima etmek.
"""
}

# --- 3. VERİ SİSTEMİ ---
def log_kaydet(data):
    dosya = "tez_verileri_8sinif_final.csv"
    df = pd.DataFrame([data])
    if not os.path.isfile(dosya):
        df.to_csv(dosya, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(dosya, mode='a', index=False, header=False, encoding="utf-8-sig")

# --- 4. SESSION STATE (HAFIZA KİLİDİ) ---
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "canvas_storage" not in st.session_state:
    st.session_state.canvas_storage = {step: None for step in BASAMAK_TALIMATLARI.keys()}
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {step: [] for step in BASAMAK_TALIMATLARI.keys()}
if "current_step" not in st.session_state:
    st.session_state.current_step = "1. Ayrıştırma"
if "step_start_time" not in st.session_state:
    st.session_state.step_start_time = time.time()

# --- 5. SOL PANEL (SIDEBAR & ÖĞRETMEN ANALİZİ) ---
with st.sidebar:
    st.title("👨‍🏫 Araştırma Paneli")
    mode = st.selectbox("Giriş Türü:", ["Öğrenci Girişi", "Öğretmen (Admin)"])
    
    if mode == "Öğretmen (Admin)":
        if st.text_input("Şifre:", type="password") != "tez2024": st.stop()
        if os.path.isfile("tez_verileri_8sinif_final.csv"):
            df_csv = pd.read_csv("tez_verileri_8sinif_final.csv")
            st.subheader("📊 Basamak Dağılımı")
            st.bar_chart(df_csv["basamak"].value_counts())
            st.divider()
            st.download_button("📥 Tüm Verileri İndir (CSV)", df_csv.to_csv(index=False), "tez_verileri.csv")
            st.dataframe(df_csv)
        st.stop()
    
    student_id = st.text_input("Öğrenci No:", placeholder="Örn: 8A-12")
    if not student_id: st.warning("Numaranızı girin."); st.stop()
    
    st.divider()
    step_list = list(BASAMAK_TALIMATLARI.keys())
    step_index = step_list.index(st.session_state.current_step)
    selected_radio = st.radio("Aşamayı Seçin:", step_list, index=step_index)
    
    if selected_radio != st.session_state.current_step:
        st.session_state.current_step = selected_radio
        st.session_state.step_start_time = time.time()
        st.rerun()

# --- 6. ANA EKRAN ---
st.title("🎯 LGS Matematik Strateji İstasyonu")

if st.session_state.uploaded_file_data is None:
    up_file = st.file_uploader("Soru Fotoğrafını Yükle", type=["png", "jpg", "jpeg"])
    if up_file:
        st.session_state.uploaded_file_data = up_file
        st.rerun()
else:
    st.image(st.session_state.uploaded_file_data, use_container_width=True)
    if st.button("❌ Yeni Soru Yükle"):
        st.session_state.uploaded_file_data = None
        st.rerun()

st.divider()
col_draw, col_chat = st.columns([1.1, 0.9], gap="large")

with col_draw:
    st.markdown(f"#### 🖌️ Karalama Alanı ({st.session_state.current_step})")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", stroke_width=3, stroke_color="#000000",
        background_color="#ffffff", height=450, drawing_mode="freedraw",
        initial_drawing=st.session_state.canvas_storage[st.session_state.current_step],
        key=f"canvas_lgs_analitik_{st.session_state.current_step.replace(' ', '_')}",
    )
    if canvas_result.json_data is not None:
        st.session_state.canvas_storage[st.session_state.current_step] = canvas_result.json_data

    # --- MADDE 1: Üstbilişsel Güven Ölçeği ---
    st.divider()
    conf_col, btn_col = st.columns([3, 1])
    with conf_col:
        confidence = st.select_slider("Bu adımdaki çözümünden ne kadar eminsin?", 
                                     options=["Kararsızım", "Az Eminim", "Orta", "Eminim", "Çok Eminim"])
    with btn_col:
        if st.button("Kaydet"):
            log_kaydet({
                "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ogrenci_id": student_id,
                "basamak": st.session_state.current_step,
                "sure_sn": round(time.time() - st.session_state.step_start_time, 2),
                "rol": "Metacognition",
                "icerik": f"Güven Düzeyi: {confidence}"
            })
            st.toast("Düşüncen kaydedildi!")

with col_chat:
    st.markdown("### 💬 Rehber Bot")
    step_chat = st.session_state.chat_storage[st.session_state.current_step]
    
    chat_box = st.container(height=400)
    with chat_box:
        for m in step_chat:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    # --- MADDE 2: Hareketsizlik Uyarısı (3 Dakika) ---
    current_duration = time.time() - st.session_state.step_start_time
    if current_duration > 180 and len(step_chat) < 2:
        st.info("💡 Rehber: Biraz takılmış gibisin. Soruda verilenleri karalama alanına not etmeyi denedin mi?")

    if prompt := st.chat_input("Düşünceni buraya yaz..."):
        duration = round(time.time() - st.session_state.step_start_time, 2)
        
        user_msg = {"role": "user", "content": prompt}
        step_chat.append(user_msg)
        
        log_kaydet({
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ogrenci_id": student_id,
            "basamak": st.session_state.current_step,
            "sure_sn": duration,
            "rol": "user",
            "icerik": prompt
        })
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": f"Sen bir matematik rehberisin. ASLA cevap verme, işlem yapma. ŞU KURALLARA UY: {BASAMAK_TALIMATLARI[st.session_state.current_step]}"}
                    ] + step_chat,
                    "temperature": 0.1
                }
            )
            ans = response.json()['choices'][0]['message']['content']
            
            assistant_msg = {"role": "assistant", "content": ans}
            step_chat.append(assistant_msg)
            
            log_kaydet({
                "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ogrenci_id": student_id,
                "basamak": st.session_state.current_step,
                "sure_sn": duration,
                "rol": "assistant",
                "icerik": ans
            })
            
            st.session_state.chat_storage[st.session_state.current_step] = step_chat
            st.rerun()
        except: st.error("Bağlantı sorunu oluştu.")

# --- MADDE 3: Final Strateji Kartı ---
st.write("---")
if st.button("🏁 Çözümü Bitir ve Özetini Al"):
    st.balloons()
    st.markdown("### 📜 Stratejik Düşünce Kartın")
    user_inputs = [m["content"] for m in step_chat if m["role"] == "user"]
    if user_inputs:
        st.info(f"Bravo! Bu soruda şu adımları izledin: {', '.join(user_inputs[:3])}...")
    else:
        st.info("Harika bir başlangıç yaptın, bir sonraki soruda görüşürüz!")
