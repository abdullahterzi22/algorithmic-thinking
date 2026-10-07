import streamlit as st
import requests
import pandas as pd
import os
import base64
import io
import json
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Algoritmik Düşünme Atölyesi",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# GEMINI API AYARLARI
# ============================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error(
        "GEMINI_API_KEY bulunamadı. Streamlit Secrets bölümüne API anahtarınızı ekleyin."
    )
    st.stop()

MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)
DATA_FILE = "tez_verileri_final.csv"

# ============================================================
# BASAMAKLAR
# ============================================================
BASAMAK_TALIMATLARI = {
    "1. Ayrıştırma": """
Amaç: Öğrencinin problemi anlamlı parçalara ayırmasını sağlamak.

Öğrencinin:
- Problemdeki temel bilgileri ve nicelikleri fark etmesini,
- Birbiriyle ilişkili bilgileri ayırt etmesini,
- Problemi anlamlı parçalara ayırmasını,
- Değişen veya birbirine bağlı nicelikleri fark etmesini sağla.

Kesinlikle yapma:
- Çözüm yolu, işlem, formül, denklem veya cevap verme.
- Problemi öğrencinin yerine parçalara ayırma.
- Sonuca götüren doğrudan ipucu verme.
""",

    "2. Soyutlama": """
Amaç: Öğrencinin problemin altında yatan matematiksel yapıyı kendisinin fark etmesini sağlamak.

Öğrencinin:
- Gerekli ve gereksiz bilgileri ayırt etmesini,
- Değişen nicelikleri ve aralarındaki ilişkileri incelemesini,
- Önceki matematiksel bilgilerini problemle ilişkilendirmesini,
- Tablo, grafik, sözel ifade veya şekil gibi temsiller arasında ilişki kurmasını,
- Uygunsa örüntü/genel yapı oluşturmasını,
- Özel bir durumdan daha genel bir ilişkiye ulaşmasını sağla.

Kesinlikle yapma:
- İlişkinin adını veya yöntemini öğrencinin yerine söyleme.
- Denklem, formül, eğim, sabit değişim veya çözüm verme.
- Öğrencinin yerine örüntü oluşturma.
""",

    "3. Algoritma Tasarımı": """
Amaç: Öğrencinin kendi çözüm ve düşünme planını adım adım oluşturmasını ve bunu akış şemasına dönüştürmesini sağlamak.

Öğrencinin:
- İlk olarak hangi bilgiyle başlayacağını belirlemesini,
- İşlem/düşünme adımlarını kendisinin oluşturmasını,
- Adımları uygun sıraya koymasını,
- Bir adımın sonraki adıma nasıl katkı verdiğini düşünmesini,
- Gerekirse alternatif bir yol düşünmesini sağla.

Kesinlikle yapma:
- Algoritmayı öğrencinin yerine oluşturma.
- İşlem sırası, formül, denklem veya çözüm adımı verme.
- Sonucu söyleme.
""",

    "4. Hata Ayıklama": """
Amaç: Öğrencinin oluşturduğu çözümü ve ilişkileri problem koşullarıyla kendisinin sınamasını sağlamak.

Öğrencinin:
- Kullandığı bilgileri ve işlem adımlarını yeniden incelemesini,
- Farklı değerlerle veya başka bir gösterimle kontrol etmesini,
- Sonucun problem koşullarını sağlayıp sağlamadığını sınamasını,
- Şüphe duyduğu adımı belirlemesini,
- Gerekirse kendi çözümünde düzeltme yapmasını,
- Düzeltmesini yeniden test etmesini sağla.

Kesinlikle yapma:
- “Doğru”, “yanlış”, “hata yaptın” diyerek değerlendirme yapma.
- Doğru sonucu, doğru denklemi veya eksik adımı söyleme.
- Öğrencinin yerine düzeltme yapma.
"""
}

STAGE_KEYS = list(BASAMAK_TALIMATLARI.keys())

# 5. aşama artık uygulamanın gerçek süreç aşaması olarak tutuluyor.
METABILISSEL_SORULAR = {
    "1. Ayrıştırma": "Problemi parçalara ayırırken hangi bilgiler ve nicelikler dikkatini çekti?",
    "2. Soyutlama": "Nicelikler arasındaki ilişkiyi fark ederken hangi bilgiler sana yardımcı oldu?",
    "3. Algoritma Tasarımı": "Çözüm adımlarını planlarken bu sırayı neden seçtin?",
    "4. Hata Ayıklama": "Çözümünü kontrol ederken neyi nasıl sınadın?",
    "5. Metabilişsel Yansıtma": "Problemi çözerken düşünme biçimini nasıl değiştirdiğini veya hangi stratejinin sana yardımcı olduğunu düşünüyorsun?"
}

# ============================================================
# ANA SİSTEM PROMPTU
# ============================================================
SYSTEM_PROMPT = """
Sen ortaokul matematik öğrencisine rehberlik eden bir matematik öğretmenisin.

KONU: DOĞRUSAL İLİŞKİLER

TEMEL İLKE: ÖĞRENCİ ÇÖZER. SEN YALNIZCA ÖĞRENCİNİN DÜŞÜNME SÜRECİNİ YÖNLENDİRİRSİN.

KESİN KURALLAR:
1. Asla doğrudan çözüm veya cevap verme.
2. Asla öğrencinin yerine hesaplama yapma.
3. Asla formül veya denklem verme.
4. Asla öğrencinin yerine algoritma oluşturma.
5. Asla öğrencinin yerine matematiksel ilişki kurma.
6. “Doğru”, “yanlış”, “hata yaptın” ifadelerini değerlendirme amacıyla kullanma.
7. Aynı anda yalnızca BİR soru sor.
8. Öğrencinin cevabını bekle.
9. 1-2 kısa cümleden fazla yazma.
10. Hazır övgü kalıpları kullanma.
11. Öğrencinin söylediğini gereksiz yere tekrar etme.
12. Soruyu öğrencinin son cevabına ve YÜKLENEN GERÇEK PROBLEME göre oluştur.
13. Genel, her probleme uyabilecek mekanik sorulardan kaçın.
14. Görselde tablo/grafik/şekil varsa soruyu mümkün olduğunca gerçek görseldeki bilgiye bağla.
15. Öğrencinin neyi işaretlediğini görmek için varsa öğrencinin problem üzerindeki çizimini kullan; çizimin matematiksel doğruluğunu öğrencinin yerine değerlendirme.
16. Öğrenci yardım istediğinde bile cevap verme; yalnızca bir sonraki düşünme adımını açacak tek bir soru sor.
17. Bir basamağın tamamlandığını kendin ilan etme. Öğrencinin açıkça “Basamağı tamamladım” demesini bekle.

ASLA ŞUNLARI YAPMA:
- “Bu doğrusal ilişkidir.”
- “Eğimi bul.”
- “Sabit değişimi hesapla.”
- “Şu formülü kullan.”
- “Önce şunu bul.”
- “Denklemi yaz.”
- “Cevap ...”
- “Doğru cevap ...”
- “Yanlış yaptın.”

YANIT BİÇİMİ:
- Türkçe.
- 1 veya 2 kısa cümle.
- Tam olarak bir soru.
- Öğrencinin son mesajına dayalı.
- Bir sonraki düşünme adımına yönlendirici.
"""

# ============================================================
# VERİ KAYDETME
# ============================================================
def log_kaydet(data):
    try:
        df = pd.DataFrame([data])
        df.to_csv(
            DATA_FILE,
            mode="a",
            index=False,
            header=not os.path.isfile(DATA_FILE),
            encoding="utf-8-sig"
        )
    except Exception as e:
        st.warning(f"Veri kaydedilemedi: {e}")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(student_id, stage, tip, content, extra=None):
    row = {
        "tarih": now_str(),
        "id": student_id,
        "basamak": stage,
        "tip": tip,
        "icerik": content
    }
    if extra:
        row.update(extra)
    log_kaydet(row)

# ============================================================
# GÖRSEL FONKSİYONLARI
# ============================================================
def get_original_image():
    uploaded_file = st.session_state.get("uploaded_file_data")
    if uploaded_file is None:
        return None
    try:
        return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception:
        return None


def get_problem_image_part():
    uploaded_file = st.session_state.get("uploaded_file_data")
    if uploaded_file is None:
        return None
    try:
        return {
            "inline_data": {
                "mime_type": uploaded_file.type,
                "data": base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            }
        }
    except Exception:
        return None


def get_annotation_image_part():
    """Son öğrenci işaretlemesini PNG olarak Gemini'ye gönderir."""
    image_data = st.session_state.get("annotation_image_data")
    if image_data is None:
        return None
    try:
        image = Image.fromarray(image_data.astype("uint8"), mode="RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buffer.getvalue()).decode("utf-8")
            }
        }
    except Exception:
        return None


def get_annotation_background():
    image = get_original_image()
    if image is None:
        return None, None, None
    try:
        max_width = 900
        original_width, original_height = image.size
        display_width = min(original_width, max_width)
        display_height = max(1, int(original_height * display_width / original_width))
        image = image.resize((display_width, display_height), Image.LANCZOS)
        return image, display_width, display_height
    except Exception:
        return None, None, None

# ============================================================
# GEMINI İSTEK FONKSİYONU
# ============================================================
def gemini_generate(parts, temperature=0.2, max_tokens=250, include_problem_image=False,
                     include_annotation=False):
    final_parts = list(parts)

    if include_problem_image:
        original = get_problem_image_part()
        if original is not None:
            final_parts.append(original)

    if include_annotation:
        annotated = get_annotation_image_part()
        if annotated is not None:
            final_parts.append({
                "text": "Aşağıdaki ek görsel, öğrencinin problem üzerinde yaptığı işaretlemeyi içerir. İşaretlemeyi yalnızca öğrencinin dikkatini yönelttiği bölgeyi anlamak için kullan."
            })
            final_parts.append(annotated)

    payload = {
        "contents": [{"role": "user", "parts": final_parts}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.8,
            "maxOutputTokens": max_tokens
        }
    }

    response = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise Exception(f"Gemini API Hatası ({response.status_code}): {detail}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise Exception(f"Gemini beklenmeyen yanıt verdi: {data}")

# ============================================================
# PROBLEM ANALİZİ
# ============================================================
def analyze_problem_image(uploaded_file):
    if uploaded_file is None:
        return ""

    image_data = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    prompt = """
Bu görselde bir ortaokul matematik problemi var.

Görseli çözme. Yalnızca rehber sistemin problemi doğru anlaması için analiz et.

Belirle:
1. Problem metni.
2. Verilen bilgiler.
3. İstenen bilgi.
4. Tablo, grafik veya şekil varsa içeriği.
5. Problemdeki temel nicelikler.
6. Görselde açıkça görülen ilişkiler.
7. Değişen veya birbirine bağlı nicelikler.
8. Önemli matematiksel bilgiler.
9. Okunamayan/belirsiz bölümler.

Tablo veya grafik varsa değerleri mümkün olduğunca doğru oku.
Çözüm yapma, işlem yapma, cevap bulma, denklem oluşturma veya yöntem önerme.
"""
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": uploaded_file.type, "data": image_data}}
    ]
    return gemini_generate(parts, temperature=0.1, max_tokens=600)

# ============================================================
# BOT YARDIMCI FONKSİYONLARI
# ============================================================
def conversation_text(chat_history):
    lines = []
    for message in chat_history:
        role = "ÖĞRENCİ" if message["role"] == "user" else "REHBER"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def ilk_soruyu_olustur(current_step, problem_analysis):
    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAK AMACI:
{BASAMAK_TALIMATLARI[current_step]}

PROBLEM ANALİZİ:
{problem_analysis}

Öğrenci bu basamakta henüz cevap vermedi.
Yüklenen gerçek problem görselini incele.
İlk soruyu bu probleme özgü oluştur.
Yalnızca tek kısa yönlendirici soru sor.
"""
    return gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=100,
        include_problem_image=True,
        include_annotation=bool(st.session_state.get("annotation_image_data"))
    )


def ogrenciye_cevap_ver(current_step, problem_analysis, chat_history, student_message,
                        help_request=False):
    help_instruction = ""
    if help_request:
        help_instruction = """
Öğrenci açıkça yardım istedi. Yardım talebini karşılamak için çözüm veya cevap verme.
Yalnızca öğrencinin mevcut düşüncesini bir sonraki küçük adıma taşıyacak tek bir soru sor.
"""

    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAK PROTOKOLÜ:
{BASAMAK_TALIMATLARI[current_step]}

PROBLEM ANALİZİ:
{problem_analysis}

ÖNCEKİ DİYALOG:
{conversation_text(chat_history)}

ÖĞRENCİNİN SON MESAJI:
{student_message}

{help_instruction}

Görevin:
- Gerçek problem görselini incele.
- Öğrencinin son mesajını dikkate al.
- Varsa öğrencinin problem üzerindeki işaretlemesini yalnızca dikkat ettiği bölgeyi anlamak için kullan.
- Öğrencinin söylemediği bir sonucu onun adına çıkarma.
- Tam olarak bir soru sor.
- 1-2 kısa cümle kullan.
- Çözüm, formül, denklem, işlem sonucu veya doğrudan yöntem verme.
"""

    return gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=120,
        include_problem_image=True,
        include_annotation=bool(st.session_state.get("annotation_image_data"))
    )

# ============================================================
# METABİLİŞSEL YANSITMA İÇİN AYRI 5. AŞAMA
# ============================================================
def metacognitive_response(student_id):
    process = []
    for stage in STAGE_KEYS[:4]:
        messages = st.session_state.chat_storage.get(stage, [])
        process.append(f"===== {stage} =====")
        for m in messages:
            process.append(f"{m['role']}: {m['content']}")

    reflection = st.session_state.get("meta_5_Metabilişsel_Yansıtma", "")
    confidence = st.session_state.get("confidence_5. Metabilişsel Yansıtma", "Kararsızım")

    prompt = f"""
Sen bir ortaokul matematik öğretmenisin.

Öğrencinin algoritmik düşünme sürecini değerlendir.
Problemi yeniden çözme ve cevap verme.

Değerlendirmeyi şu dört süreç boyutuna dayandır:
- Ayrıştırma
- Soyutlama
- Algoritma Tasarımı
- Hata Ayıklama

Öğrencinin kendi ifadelerinden hareket et.
Öğrencinin yerine yeni matematiksel çıkarım yapma.
Kısa ve öğretmen kullanımına uygun bir süreç değerlendirmesi oluştur.

ÖĞRENCİ: {student_id}

SÜREÇ:
{chr(10).join(process)}

5. AŞAMA ÖZ-YANSITMA:
{reflection}

5. AŞAMA EMİNLİK:
{confidence}
"""
    return gemini_generate(
        [{"text": prompt}],
        temperature=0.3,
        max_tokens=600,
        include_problem_image=True,
        include_annotation=bool(st.session_state.get("annotation_image_data"))
    )

# ============================================================
# SESSION STATE
# ============================================================
def reset_process():
    st.session_state.chat_storage = {stage: [] for stage in STAGE_KEYS[:4]}
    st.session_state.completed_stages = []
    st.session_state.current_step = "1. Ayrıştırma"
    st.session_state.annotation_reset += 1
    st.session_state.annotation_image_data = None
    st.session_state.annotation_json = None
    st.session_state.flowchart_storage = {}
    st.session_state.final_text = None

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "problem_analysis" not in st.session_state:
    st.session_state.problem_analysis = None
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {stage: [] for stage in STAGE_KEYS[:4]}
if "current_step" not in st.session_state:
    st.session_state.current_step = "1. Ayrıştırma"
if "completed_stages" not in st.session_state:
    st.session_state.completed_stages = []
if "annotation_reset" not in st.session_state:
    st.session_state.annotation_reset = 0
if "annotation_image_data" not in st.session_state:
    st.session_state.annotation_image_data = None
if "annotation_json" not in st.session_state:
    st.session_state.annotation_json = None
if "final_text" not in st.session_state:
    st.session_state.final_text = None
if "flowchart_storage" not in st.session_state:
    st.session_state.flowchart_storage = {}
if "problem_session_id" not in st.session_state:
    st.session_state.problem_session_id = None

# ============================================================
# SOL MENÜ
# ============================================================
with st.sidebar:
    st.title("👨‍🏫 Araştırma Paneli")

    mode = st.selectbox(
        "Giriş Türü:",
        ["Öğrenci Girişi", "Öğretmen (Admin)"]
    )

    if mode == "Öğretmen (Admin)":
        sifre = st.text_input("Şifre:", type="password")
        if sifre == "tez2024":
            st.success("Admin Paneli Aktif")
            if os.path.isfile(DATA_FILE):
                try:
                    df_csv = pd.read_csv(
                        DATA_FILE,
                        sep=None,
                        engine="python",
                        on_bad_lines="skip"
                    )
                    st.write("### 📊 Veri Kayıtları")
                    st.dataframe(df_csv.tail(50), use_container_width=True)

                    st.write("### 👥 Öğrenci Süreç Özeti")
                    if "id" in df_csv.columns and "basamak" in df_csv.columns and "tip" in df_csv.columns:
                        summary = (
                            df_csv.groupby(["id", "basamak"])
                            .agg(
                                kayıt_sayısı=("tip", "size"),
                                öğrenci_mesajı=("tip", lambda x: int((x == "Öğrenci").sum())),
                                bot_mesajı=("tip", lambda x: int((x == "Bot").sum())),
                                yardım_talebi=("tip", lambda x: int((x == "Yardım Talebi").sum())),
                                çizim=("tip", lambda x: int((x.isin(["Cizim", "Problem Üzeri Çizim"])).sum())),
                                metabiliş=("tip", lambda x: int((x == "Metabiliş").sum())),
                                eminlik=("tip", lambda x: int((x == "Eminlik").sum())),
                                tamamlama=("tip", lambda x: int((x == "Basamak Tamamlama").sum()))
                            )
                            .reset_index()
                        )
                        st.dataframe(summary, use_container_width=True)

                    csv_data = df_csv.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Tüm Verileri İndir",
                        csv_data,
                        "tez_data.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Dosya hatası: {e}")
            else:
                st.info("Henüz veri kaydı yok.")
        st.stop()

    student_id = st.text_input("Öğrenci No:", placeholder="Örn: Hakan")
    if not student_id:
        st.warning("Devam etmek için öğrenci numaranızı girin.")
        st.stop()

    st.divider()
    st.write("🖌️ **Akış Şeması Araçları**")
    tool_map = {
        "Dikdörtgen (İşlem)": "rect",
        "Elips (Başla/Bitir)": "circle",
        "Ok/Çizgi": "line",
        "Serbest Çizim": "freedraw",
        "Düzenle/Taşı": "transform",
        "Çokgen": "polygon"
    }
    selected_tool = st.selectbox("Araç Seçin:", list(tool_map.keys()))
    drawing_mode = tool_map[selected_tool]
    stroke_color = st.color_picker("Çizgi Rengi:", "#000000")
    fill_color = st.color_picker("Kutu Rengi:", "#EEEEEE")

    st.divider()
    st.write("🔎 **Basamaklar**")
    if st.session_state.completed_stages:
        st.caption("Tamamlanan: " + ", ".join(st.session_state.completed_stages))

    # 5. aşama, ilk dört basamak tamamlandıktan sonra açılır.
    available_steps = STAGE_KEYS[:4]
    if len(st.session_state.completed_stages) >= 4:
        available_steps = STAGE_KEYS

    current_index = available_steps.index(st.session_state.current_step) if st.session_state.current_step in available_steps else 0
    selected_step = st.radio("Aşamayı Seçin:", available_steps, index=current_index)

    if selected_step != st.session_state.current_step:
        st.session_state.current_step = selected_step
        st.rerun()

# ============================================================
# ANA BAŞLIK
# ============================================================
st.title("🎯 Algoritmik Problem Çözme Rehberi")
st.write(f"### Mevcut Basamak: {st.session_state.current_step}")

# ============================================================
# PROBLEM YÜKLEME
# ============================================================
if st.session_state.uploaded_file_data is None:
    uploaded = st.file_uploader(
        "📷 Soru Fotoğrafı Yükle",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        st.session_state.uploaded_file_data = uploaded
        st.session_state.problem_session_id = f"{student_id}_{now_str()}"

        with st.spinner("Problem analiz ediliyor..."):
            try:
                st.session_state.problem_analysis = analyze_problem_image(uploaded)
            except Exception as e:
                st.error("Problem analiz edilemedi.")
                st.code(str(e))
                st.session_state.problem_analysis = ""

        reset_process()
        log(student_id, "PROBLEM", "Problem Yüklendi", uploaded.name)
        st.rerun()
else:
    st.image(st.session_state.uploaded_file_data, width=900)

    # --------------------------------------------------------
    # PROBLEM ÜZERİNDE ÖĞRENCİ İŞAREMLEME
    # --------------------------------------------------------
    annotation_image, annotation_width, annotation_height = get_annotation_background()
    if annotation_image is not None:
        st.write("🖍️ **Problem Üzerinde İşaretleme**")
        st.caption(
            "Problem görselinin üzerinde önemli gördüğün bölümleri işaretleyebilirsin. "
            "İşaretlemen rehber sorularının probleme daha iyi bağlanmasına yardımcı olur."
        )

        annotation_col1, annotation_col2 = st.columns([3, 1])
        with annotation_col1:
            annotation_tool_map = {
                "Serbest Çizim": "freedraw",
                "Ok / Çizgi": "line",
                "Dikdörtgen": "rect",
                "Elips": "circle"
            }
            annotation_tool = st.selectbox(
                "İşaretleme aracı:",
                list(annotation_tool_map.keys()),
                key="annotation_tool"
            )
        with annotation_col2:
            annotation_color = st.color_picker(
                "İşaretleme rengi:",
                "#FF0000",
                key="annotation_color"
            )

        annotation_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_color=annotation_color,
            stroke_width=3,
            background_image=annotation_image,
            height=annotation_height,
            width=annotation_width,
            initial_drawing=st.session_state.annotation_json if st.session_state.annotation_json else None,
            drawing_mode=annotation_tool_map[annotation_tool],
            update_streamlit=True,
            key="problem_annotation_" + str(st.session_state.annotation_reset)
        )

        # Her rerun'da mevcut çizimi session state'e al.
        # DİKKAT: streamlit-drawable-canvas'ta image_data bir property
        # olduğundan, çizim henüz oluşmadığında erişmek RuntimeError üretebilir.
        if annotation_result is not None:
            try:
                annotation_json = annotation_result.json_data
            except Exception:
                annotation_json = None

            if annotation_json:
                st.session_state.annotation_json = annotation_json

                # image_data yalnızca gerçekten bir canvas görüntüsü oluştuysa
                # okunur. Bazı sürümlerde bu property, veri yokken RuntimeError verir.
                try:
                    annotation_image_data = annotation_result.image_data
                    if annotation_image_data is not None:
                        st.session_state.annotation_image_data = annotation_image_data
                except (RuntimeError, AttributeError, ValueError):
                    pass

        save_ann_col, clear_ann_col = st.columns(2)
        with save_ann_col:
            if st.button("💾 Problem Üzerindeki İşaretlemeyi Kaydet", key="save_problem_annotation"):
                if st.session_state.annotation_json:
                    log(
                        student_id,
                        st.session_state.current_step,
                        "Problem Üzeri Çizim",
                        json.dumps(st.session_state.annotation_json, ensure_ascii=False),
                        {"yardim_talebi": "Hayır"}
                    )
                    st.success("Problem üzerindeki işaretleme kaydedildi.")
                else:
                    st.info("Önce problem üzerinde bir işaretleme yap.")
        with clear_ann_col:
            if st.button("🧹 İşaretlemeyi Temizle", key="clear_problem_annotation"):
                st.session_state.annotation_reset += 1
                st.session_state.annotation_image_data = None
                st.session_state.annotation_json = None
                st.rerun()

    if st.button("❌ Soruyu Değiştir"):
        st.session_state.uploaded_file_data = None
        st.session_state.problem_analysis = None
        reset_process()
        st.rerun()

# ============================================================
# PROBLEM ANALİZİ
# ============================================================
if st.session_state.problem_analysis:
    with st.expander("🔎 Problem analizini göster"):
        st.write(st.session_state.problem_analysis)

st.divider()

# ============================================================
# İKİ SÜTUN
# ============================================================
col1, col2 = st.columns([1.3, 1], gap="large")

# ============================================================
# SOL SÜTUN
# ============================================================
with col1:
    current_stage = st.session_state.current_step

    if current_stage != "5. Metabilişsel Yansıtma":
        st.write("🖼️ **Tasarım ve Planlama Alanı**")
        st.caption("Akış şemanı burada oluşturabilirsin.")

        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_color=stroke_color,
            stroke_width=3,
            background_color="#ffffff",
            height=450,
            drawing_mode=drawing_mode,
            initial_drawing=st.session_state.flowchart_storage.get(current_stage),
            update_streamlit=True,
            key="canvas_" + current_stage.replace(" ", "_")
        )

        if st.button("🖼️ Tasarımı Kaydet"):
            if canvas_result.json_data:
                st.session_state.flowchart_storage[current_stage] = canvas_result.json_data
                log(
                    student_id,
                    current_stage,
                    "Cizim",
                    json.dumps(canvas_result.json_data, ensure_ascii=False)
                )
                st.success("Tasarım kaydedildi!")
            else:
                st.info("Önce akış şeması alanında bir tasarım oluştur.")

        st.write("---")

        st.info("🧠 **Öz-Yansıtma:** " + METABILISSEL_SORULAR[current_stage])
        meta_key = "meta_" + current_stage.replace(" ", "_")
        m_cevap = st.text_area("Düşünceni buraya yaz...", key=meta_key)

        if st.button("💾 Düşüncemi Kaydet", key="save_meta_" + current_stage):
            log(student_id, current_stage, "Metabiliş", m_cevap)
            st.success("Kaydedildi!")

        st.write("---")

        st.write("⭐ **Bu adımdaki çözümünden ne kadar eminsin?**")
        confidence_options = [
            "1 - Hiç güvenmiyorum",
            "2 - Biraz güveniyorum",
            "3 - Kararsızım",
            "4 - Güveniyorum",
            "5 - Çok güveniyorum"
        ]
        confidence = st.select_slider(
            "Derecelendir:",
            options=confidence_options,
            value="3 - Kararsızım",
            key="confidence_" + current_stage
        )

        if st.button("📈 Eminlik Derecesini Kaydet", key="save_conf_" + current_stage):
            log(student_id, current_stage, "Eminlik", confidence)
            st.success(f"Eminlik: {confidence}")

        st.write("---")

        # Öğrencinin kendisinin basamağı tamamladığını bildirmesi.
        stage_index = STAGE_KEYS[:4].index(current_stage)
        if stage_index < 3:
            next_stage = STAGE_KEYS[stage_index + 1]
            if st.button("✅ Bu Basamağı Tamamladım", key="complete_" + current_stage):
                if current_stage not in st.session_state.completed_stages:
                    st.session_state.completed_stages.append(current_stage)
                log(student_id, current_stage, "Basamak Tamamlama", "Öğrenci basamağı tamamladığını bildirdi.")
                st.session_state.current_step = next_stage
                st.success(f"Bir sonraki basamağa geçiliyor: {next_stage}")
                st.rerun()
        else:
            if st.button("✅ Hata Ayıklama Basamağını Tamamladım", key="complete_debug"):
                if current_stage not in st.session_state.completed_stages:
                    st.session_state.completed_stages.append(current_stage)
                log(student_id, current_stage, "Basamak Tamamlama", "Öğrenci Hata Ayıklama basamağını tamamladığını bildirdi.")
                st.session_state.current_step = "5. Metabilişsel Yansıtma"
                st.rerun()

    else:
        # ----------------------------------------------------
        # 5. AŞAMA: GERÇEK METABİLİŞSEL YANSITMA
        # ----------------------------------------------------
        st.write("🧠 **5. Metabilişsel Yansıtma**")
        st.caption(
            "Bu aşamada çözümün kendisinden çok, nasıl düşündüğünü ve sürecini nasıl yönettiğini değerlendir."
        )

        reflection_key = "meta_5_Metabilişsel_Yansıtma"
        reflection = st.text_area(
            METABILISSEL_SORULAR["5. Metabilişsel Yansıtma"],
            key=reflection_key,
            height=160
        )

        if st.button("💾 Öz-Yansıtmayı Kaydet", key="save_meta_5"):
            log(student_id, current_stage, "Metabiliş", reflection)
            st.success("Öz-yansıtma kaydedildi.")

        st.write("---")
        st.write("⭐ **Tüm süreç hakkında ne kadar eminsin?**")
        confidence5 = st.select_slider(
            "Derecelendir:",
            options=[
                "1 - Hiç güvenmiyorum",
                "2 - Biraz güveniyorum",
                "3 - Kararsızım",
                "4 - Güveniyorum",
                "5 - Çok güveniyorum"
            ],
            value="3 - Kararsızım",
            key="confidence_5. Metabilişsel Yansıtma"
        )

        if st.button("📈 Genel Eminlik Derecesini Kaydet", key="save_conf_5"):
            log(student_id, current_stage, "Eminlik", confidence5)
            st.success(f"Eminlik: {confidence5}")

        st.write("---")
        if st.button("🏁 Çözümü Bitir ve Öğretmen Özeti Oluştur", key="finish_process"):
            if not reflection.strip():
                st.warning("Önce öz-yansıtma cevabını yaz.")
            else:
                with st.spinner("Süreç analiz ediliyor..."):
                    try:
                        st.session_state.final_text = metacognitive_response(student_id)
                        log(student_id, "FİNAL", "Final Özeti", st.session_state.final_text)
                    except Exception as e:
                        st.error("Özet hazırlanamadı.")
                        st.code(str(e))

        if st.session_state.final_text:
            st.success("Süreç değerlendirmesi hazır.")
            st.write(st.session_state.final_text)

# ============================================================
# SAĞ SÜTUN - REHBER BOT
# ============================================================
with col2:
    current_stage = st.session_state.current_step
    st.write("💬 **Rehber Bot**")

    if current_stage == "5. Metabilişsel Yansıtma":
        st.info(
            "Bu aşamada sohbet yerine öz-yansıtma alanını kullan. "
            "Amaç, çözüm sürecini nasıl yönettiğini düşünmek."
        )
    else:
        # ----------------------------------------------------
        # İLK SORU
        # ----------------------------------------------------
        if (
            st.session_state.problem_analysis
            and len(st.session_state.chat_storage[current_stage]) == 0
        ):
            with st.spinner("Rehber hazırlanıyor..."):
                try:
                    first_question = ilk_soruyu_olustur(
                        current_stage,
                        st.session_state.problem_analysis
                    )
                    st.session_state.chat_storage[current_stage].append({
                        "role": "assistant",
                        "content": first_question
                    })
                    log(student_id, current_stage, "Bot", first_question)
                    st.rerun()
                except Exception as e:
                    st.error("İlk soru oluşturulamadı.")
                    st.code(str(e))

        # ----------------------------------------------------
        # SOHBET
        # ----------------------------------------------------
        chat_container = st.container(height=500)
        for message in st.session_state.chat_storage[current_stage]:
            with chat_container.chat_message(message["role"]):
                st.write(message["content"])

        # ----------------------------------------------------
        # ÖĞRENCİDEN AÇIK YARDIM TALEBİ
        # ----------------------------------------------------
        help_col, status_col = st.columns([1, 1])
        with help_col:
            help_clicked = st.button("💡 Yardıma İhtiyacım Var", key="help_" + current_stage)
        with status_col:
            if st.session_state.annotation_json:
                st.caption("🖍️ İşaretlemen rehber tarafından görülebilir.")

        student_message = st.chat_input("Düşünceni veya cevabını yaz...")

        if help_clicked:
            student_message = "Yardıma ihtiyacım var. Bir sonraki adımda neye dikkat etmeliyim?"
            help_request = True
        else:
            help_request = False

        if student_message:
            previous_history = list(st.session_state.chat_storage[current_stage])

            st.session_state.chat_storage[current_stage].append({
                "role": "user",
                "content": student_message
            })

            log(
                student_id,
                current_stage,
                "Öğrenci",
                student_message,
                {"yardim_talebi": "Evet" if help_request else "Hayır"}
            )

            if help_request:
                log(
                    student_id,
                    current_stage,
                    "Yardım Talebi",
                    "Öğrenci açık yardım düğmesine bastı."
                )

            with st.spinner("Rehber Bot düşünüyor..."):
                try:
                    answer = ogrenciye_cevap_ver(
                        current_stage,
                        st.session_state.problem_analysis,
                        previous_history,
                        student_message,
                        help_request=help_request
                    )
                    st.session_state.chat_storage[current_stage].append({
                        "role": "assistant",
                        "content": answer
                    })
                    log(student_id, current_stage, "Bot", answer)
                    st.rerun()
                except Exception as e:
                    st.error("Gemini yanıt oluşturamadı.")
                    st.code(str(e))
