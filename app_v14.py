import streamlit as st
import requests
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

# --- 1. AYARLAR VE GÜVENLİK ---
# API anahtarı artık Secrets panelinden çekiliyor
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Hata: API anahtarı (GROQ_API_KEY) Secrets panelinde bulunamadı!")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Algoritmik Düşünme Atölyesi", layout="wide")

# --- 2. AKADEMİK PROTOKOLLER ---
BASAMAK_TALIMATLARI = {
"1. Ayrıştırma": """Amaç: Öğrencinin problemi alt bileşenlerine ayırmasını sağlamak.
Zorunlu davranışlar:
- Verilenleri tek tek fark ettir (Örn: "Sence bu soruda bize hangi bilgiler verilmiş?").
- İstenen bilgiyi açıkça söylet.
- Problemi en az iki alt probleme ayırtmasını iste.
- Sayısal ilişkileri tanımlat.
Yasak:
- Çözüm yolu önermek.
- İşlem yaptırmak.
- Sonuca yaklaşan ipucu vermek.""",

"2. Soyutlama": """Amaç: Problemin matematiksel yapısını ortaya çıkarmak.
Zorunlu davranışlar:
- Bu problemin hangi tür probleme (EBOB, EKOK, Üslü Sayı vb.) benzediğini sordur.
- Önceki bilgileri hatırlat.
- Gereksiz bilgileri ayırt ettir.
- Eğer uygunsa örüntü veya genel yapı düşündür.
Yasak:
- EKOK/EBOB gibi kavramı doğrudan söylemek.
- Stratejiyi açıkça belirtmek.""",

"3. Algoritma Tasarımı": """Amaç: Öğrencinin kendi çözüm planını oluşturmasını sağlamak.
Zorunlu davranışlar:
- Adım adım plan oluşturmasını iste (Örn: "Önce ne yapacaksın, sonra hangi adımı izleyeceksin?").
- Neden bu yöntemi seçtiğini sorgula.
- Başka bir yöntem mümkün mü diye düşündür.
- Çözümün genellenebilirliğini sorgula.
- Akış şeması oluşturması için destekle.
Yasak:
- İşlem adımlarını doğrudan vermek.
- Hesap sonucu söylemek.""",

"4. Hata Ayıklama": """Amaç: Öğrencinin çözümünü kontrol etmesini sağlamak.
Zorunlu davranışlar:
- Sonucun problem koşullarını sağlayıp sağlamadığını sorgulat.
- Alternatif doğrulama yolu düşündür.
- Mantıksal tutarlılık kontrolü yaptır (Örn: "Bulduğun bu sonuç sorudaki mantığa uyuyor mu?").
Yasak:
- Sonucu doğru veya yanlış diye kesin olarak belirtmek.
- Doğru cevabı ima etmek."""
}

METABILISSEL_SORULAR = {
    "1. Ayrıştırma": "Bu problemi parçalara ayırırken en çok hangi bilgi dikkatini çekti?",
    "2. Soyutlama": "Bu soruda özellikle dikkat etmemiz gereken noktalar nelerdir? Gereksiz olduğunu düşündüğün yerler var mı?",
    "3. Algoritma Tasarımı": "Çözüm adımlarını planlarken nasıl bir yol izledin?",
    "4. Hata Ayıklama": "Bulduğun sonucun mantıklı olduğundan nasıl emin oldun? Farklı bir strateji kullanmayı düşünsen ne yapardın?"
}

SYSTEM_PROMPT = """

Rolün: Ortaokul matematik öğretmeni ve rehberisin.

KESİN KURALLAR:
- ASLA doğrudan çözüm verme.
- ASLA işlem sonucu söyleme.
- ASLA formülü doğrudan yazma.
- Öğrenci yerine çözme.

GÖREVİN:
- Öğrenciyi adım adım düşündürmek
- Soru sorarak yönlendirmek
- Öğrencinin hatasını fark ettirmek
- Metabilişsel düşünmeyi desteklemek

GÖRSEL İŞLEME TALİMATI:
- Önce görseldeki matematik problemini analiz et
- Verilenleri ve isteneni zihinsel olarak ayır
- Emin olmadığın kısımları belirt
- Gerekirse öğrenciden netleştirme iste

PEDAGOJİK YAKLAŞIM:
- Sadece soru sor
- Kısa ve yönlendirici ol
- Aynı anda tek beceriye odaklan
- Öğrencinin seviyesine uygun ilerle

ASLA YAPMA:
- "Cevap şudur"
- "Sonuç budur"
- Uzun açıklama
- Tek seferde çözüm

Görseldeki matematik problemini dikkatlice analiz et.

Adımlar:
1. Görseldeki metni ve sayıları belirle
2. Problem türünü tahmin et
3. Eksik veya belirsiz kısımları belirt

Eğer görsel net değilse öğrenciden açıklama iste.

AMAÇ:
Öğrenci çözümü KENDİ bulmalı.
"""

# --- 3. VERİ SİSTEMİ ---
def log_kaydet(data):
    dosya = "tez_verileri_final.csv"
    df = pd.DataFrame([data])
    df.to_csv(dosya, mode='a', index=False, header=not os.path.isfile(dosya), encoding="utf-8-sig")

# --- 4. SESSION STATE ---
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {s: [] for s in BASAMAK_TALIMATLARI.keys()}
if "current_step" not in st.session_state:
    st.session_state.current_step = "1. Ayrıştırma"
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = {} # Her basamağın çizimini burada tutacağız

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("👨‍🏫 Araştırma Paneli")
    mode = st.selectbox("Giriş Türü:", ["Öğrenci Girişi", "Öğretmen (Admin)"])
    
    if mode == "Öğretmen (Admin)":
        sifre = st.text_input("Şifre:", type="password")
        if sifre == "tez2024":
            st.success("Admin Paneli Aktif")
            dosya_yolu = "tez_verileri_final.csv"
            if os.path.isfile(dosya_yolu):
                try:
                    df_csv = pd.read_csv(dosya_yolu, sep=None, engine='python', on_bad_lines='skip')
                    st.write("### 📊 Veri Kayıtları")
                    st.dataframe(df_csv.tail(20))
                    csv_data = df_csv.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 Tüm Verileri İndir (CSV)", csv_data, "tez_data.csv", "text/csv")
                except Exception as e:
                    st.error(f"Dosya hatası: {e}")
            else:
                st.info("Henüz veri kaydı yok.")
        st.stop()
        
    student_id = st.text_input("Öğrenci No:", placeholder="Örn: Hakan")
    if not student_id:
        st.warning("Devam etmek için giriş yapın.")
        st.stop()
        
    st.divider()
    st.write("🖌️ **Akış Şeması Araçları**")
    tool_map = {
        "Dikdörtgen (İşlem)": "rect",
        "Elips (Başla/Bitir)": "circle",
        "Ok/Çizgi": "line",
        "Serbest Çizim": "freedraw",
        "Düzenle/Taşı": "transform",
	"Çokgen (Paralelkenar ve Baklava Çizimi)": "polygon"
    }
    secilen_etiket = st.selectbox("Araç Seçin:", list(tool_map.keys()))
    drawing_mode = tool_map[secilen_etiket]
    
    # Rengin şeffaflığını sabitliyoruz (Hatayı önlemek için Hex kullanıyoruz)
    stroke_color = st.color_picker("Çizgi Rengi:", "#000000")
    fill_color = st.color_picker("Kutu Rengi:", "#EEEEEE")

    st.divider()
    step_list = list(BASAMAK_TALIMATLARI.keys())
    cur_idx = step_list.index(st.session_state.current_step)
    sel = st.radio("Aşamayı Seçin:", step_list, index=cur_idx)
    
    if sel != st.session_state.current_step:
        st.session_state.current_step = sel
        st.rerun()

# --- 6. ANA EKRAN ---
st.title("🎯 Algoritmik Problem Çözme Rehberi")
st.write(f"### Mevcut Basamak: {st.session_state.current_step}")

# Fotoğraf Bölümü
if st.session_state.uploaded_file_data is None:
    up = st.file_uploader("Soru Fotoğrafı Yükle", type=["png", "jpg", "jpeg"])
    if up: 
        st.session_state.uploaded_file_data = up
        st.rerun()
else:
    st.image(st.session_state.uploaded_file_data, width=450)
    if st.button("❌ Soruyu Değiştir"): 
        st.session_state.uploaded_file_data = None
        st.rerun()

st.divider()
col1, col2 = st.columns([1.3, 1], gap="large")

with col1:
    st.write("🖼️ **Tasarım ve Planlama Alanı**")
    st.caption("Sol menüden 'rect' ile kutu çizebilir, 'line' ile bağlayabilir, 'transform' ile kutuları taşıyabilirsiniz.")
    
    canvas_result = st_canvas(
    fill_color=fill_color,        # Sidebar'daki renk seçiciye bağlı
    stroke_color=stroke_color,    # Sidebar'daki renk seçiciye bağlı
    stroke_width=3,
    background_color="#ffffff",
    height=450,
    drawing_mode=drawing_mode,
    update_streamlit=True,
    # Hem hafıza hem düzen için en kritik satır:
    key=f"canvas_v36_{st.session_state.current_step.replace(' ', '_')}"
)
    
    if st.button("🖼️ Tasarımı Kaydet"):
        if canvas_result.json_data:
            log_kaydet({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": student_id, "basamak": st.session_state.current_step, "tip": "Cizim", "icerik": str(canvas_result.json_data)})
            st.success("Tasarım veritabanına kaydedildi!")

    st.write("---")
    st.info(f"🧠 **Öz-Yansıtma:** {METABILISSEL_SORULAR[st.session_state.current_step]}")
    m_cevap = st.text_area("Düşünceni buraya yaz...", key=f"meta_area_{st.session_state.current_step[0]}")
    if st.button("💾 Düşüncemi Kaydet"):
        log_kaydet({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": student_id, "basamak": st.session_state.current_step, "tip": "Metabiliş", "icerik": m_cevap})
        st.success("Kaydedildi!")

    st.write("---")
    st.write("⭐ **Bu adımdaki çözümünden ne kadar eminsin?**")
    confidence = st.select_slider(
        "Derecelendir:",
        options=["Hiç Emin Değilim", "Kararsızım", "Biraz Eminim", "Çok Eminim"],
        value="Kararsızım",
        key=f"slider_{st.session_state.current_step}"
    )
    if st.button("📈 Eminlik Derecesini Kaydet"):
        log_kaydet({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": student_id, "basamak": st.session_state.current_step, "tip": "Eminlik", "icerik": confidence})
        st.success(f"Eminlik: {confidence} olarak kaydedildi.")

    st.write("---")
    if st.button("🏁 Çözümü Bitir ve Özetini Al"):
        with st.spinner("Süreç analiz ediliyor..."):
            try:
                hist_full = str(st.session_state.chat_storage)
                r_final = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "system", "content": "Öğrencinin tüm çözüm sürecini matematiksel düşünme becerileri açısından özetleyen teşvik edici bir kapanış mesajı yaz."}, 
                                 {"role": "user", "content": f"Öğrenci: {student_id}, Süreç: {hist_full}"}]
                }).json()
                final_text = r_final['choices'][0]['message']['content']
                st.info(final_text)
                log_kaydet({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": student_id, "basamak": "FİNAL", "tip": "Final Özeti", "icerik": final_text})
            except:
                st.error("Özet hazırlanamadı.")

with col2:
    st.write("💬 **Rehber Bot**")
    chat_container = st.container(height=550)
    
    for m in st.session_state.chat_storage[st.session_state.current_step]:
        chat_container.chat_message(m["role"]).write(m["content"])

    if p := st.chat_input("Düşünceni veya sorunu buraya yaz..."):
        st.session_state.chat_storage[st.session_state.current_step].append({"role": "user", "content": p})
        log_kaydet({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": student_id, "basamak": st.session_state.current_step, "tip": "Öğrenci", "icerik": p})
        
        with st.spinner("Rehber Bot yanıt veriyor..."):
            try:
                head = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": f"SEN BİR MATEMATİK REHBERİSİN. PROTOKOL: {BASAMAK_TALIMATLARI[st.session_state.current_step]}. ASLA ÇÖZÜM VERME."},
                        *st.session_state.chat_storage[st.session_state.current_step]
                    ], "temperature": 0.4
                }
                r = requests.post(GROQ_URL, headers=head, json=payload, timeout=20).json()
                ans = r['choices'][0]['message']['content']
                
                st.session_state.chat_storage[st.session_state.current_step].append({"role": "assistant", "content": ans})
                log_kaydet({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": student_id, "basamak": st.session_state.current_step, "tip": "Bot", "icerik": ans})
                st.rerun()
            except Exception as e:
                st.error("Bağlantı hatası.")
