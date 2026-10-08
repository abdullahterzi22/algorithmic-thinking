import streamlit as st
import requests
import pandas as pd
import os
import base64
import io
import json
import re
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ============================================================
# SAYFA / API
# ============================================================
st.set_page_config(page_title="Algoritmik Düşünme Atölyesi", page_icon="🧠", layout="wide")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("GEMINI_API_KEY bulunamadı. Streamlit Secrets bölümüne API anahtarınızı ekleyin.")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
DATA_FILE = "tez_verileri_final.csv"

STAGES = [
    "1. Ayrıştırma",
    "2. Soyutlama",
    "3. Algoritma Tasarımı",
    "4. Hata Ayıklama",
    "5. Metabilişsel Yansıtma",
]

# ============================================================
# PEDAGOJİK PROTOKOLLER
# ============================================================
BASAMAK_TALIMATLARI = {
"1. Ayrıştırma": """
AMAÇ: Öğrencinin problemi geriye doğru çalışma mantığıyla anlamlı parçalara ayırması.

İZLENECEK BİLİŞSEL SÜREÇ:
1. Önce problemde NE İSTENDİĞİNİ belirle.
2. İstenen bilgiye ulaşmak için doğrudan hangi bilgiye ihtiyaç olduğunu öğrencinin kendisinin bulmasını sağla.
3. Bulunan bilgiye ulaşmak için başka hangi bilgiye ihtiyaç olduğunu yine öğrencinin bulmasını sağla.
4. Bu zinciri geriye doğru, problemdeki verilen/temel bilgilere ulaşıncaya kadar sürdür.
5. Öğrenci temel/verilen bilgilere ulaştığında ve bağlantı zinciri kurulmuş olduğunda basamak tamamlanabilir.

ÖĞRENCİNİN YERİNE ZİNCİRİ KURMA. Bir sonraki bilgiyi söyleme.
İşlem, formül, denklem, sonuç veya çözüm yolu verme.
"Önce şunu bul" biçiminde doğrudan yönlendirme yapma.
"Doğru/yanlış/hata yaptın" değerlendirmesi yapma.
"Bunu bulmak için neye ihtiyacın var?" türü tek bir soru ile ilerle.
""",
"2. Soyutlama": """
AMAÇ: Ayrıştırmada ortaya çıkan bilgileri kullanarak problemin altında yatan yapıyı öğrencinin kendisinin fark etmesi.

İZLENECEK BİLİŞSEL SÜREÇ:
- Ayrıştırmada belirlenen bilgiler arasından gerekli ve gereksiz olanları ayırt et.
- Önemli nicelikleri belirle.
- Nicelikler arasındaki ilişkileri incele.
- Bir nicelik değiştiğinde diğerinin nasıl değiştiğini düşün.
- Verilen farklı temsiller arasında ilişki kur.
- Uygunsa örüntü, düzen veya daha genel bir yapı oluştur.
- Önceki matematik bilgisini yeni problemle ilişkilendir.

Ayrıştırma sonuçlarını yok sayma; öğrencinin orada kurduğu zinciri temel al.
Doğrusal ilişkiyi, eğimi, sabit değişimi, formülü, denklemi veya yöntemi öğrencinin yerine söyleme.
Öğrencinin yerine örüntü/genelleme oluşturma.
""",
"3. Algoritma Tasarımı": """
AMAÇ: Ayrıştırma ve Soyutlama aşamalarında elde edilen düşünceleri kullanarak öğrencinin çözüm algoritmasını ve akış şemasını kendisinin oluşturması.

İZLENECEK BİLİŞSEL SÜREÇ:
- Önceki iki aşamada belirlenen bilgiler ve ilişkilerden yararlan.
- İşlem/düşünme adımlarını öğrencinin kendisinin belirlemesini sağla.
- Adımları mantıklı sıraya koymasını sağla.
- Bir adımın çıktısının sonraki adıma nasıl katkı verdiğini düşündür.
- Öğrencinin çizdiği akış şemasını da dikkate al.
- Akış şeması tamamlandığında basamak tamamlanabilir.

Algoritmayı öğrencinin yerine yazma.
İşlem sırasını, formülü, denklemi, sonucu veya çözüm adımlarını söyleme.
Akış şemasının eksik kutusunu öğrencinin yerine doldurma.
""",
"4. Hata Ayıklama": """
AMAÇ: Öğrencinin kendi çözümünü ve akış şemasını test ederek varsa tutarsızlığı kendisinin bulup düzeltmesi.

DÖNGÜ:
TEST ET → ŞÜPHELİ NOKTAYI BELİRLE → İLGİLİ ADIMI İNCELE → DÜZELT → YENİDEN TEST ET.

- Önce öğrenciden kendi çözümünü problem koşullarıyla sınamasını iste.
- Gerekirse farklı bir değer, farklı bir durum, tablo/grafik veya problemdeki başka bir bilgiyle kontrol ettir.
- Tutarsızlık sezilirse nerede olabileceğini öğrencinin kendisinin araştırmasını sağla.
- Düzeltmeyi öğrencinin kendisine yaptır.
- Düzeltmeden sonra yeniden test ettir.
- Çözüm tutarlı görünüyorsa, yaş ve konu düzeyine uygun ve gerçekten mümkünse alternatif bir yol düşünülebilir.
- Uygun alternatif yoksa alternatif çözümü zorunlu kılma.

"Doğru", "yanlış", "hata yaptın", "burada hata var" deme.
Doğru sonucu veya düzeltmeyi söyleme.
"""
}

ALT_BECERILER = {
"1. Ayrıştırma": [
    "A1 - Alt amacı belirleme",
    "A2 - Geriye doğru parçalama",
    "A3 - Alt amaçlar arasındaki ilişkileri belirleme",
],
"2. Soyutlama": [
    "S1 - İlgili bilgiyi belirleme",
    "S2 - Gereksiz bilgiyi ayırt etme",
    "S3 - Nicelikler arası ilişkiyi fark etme",
    "S4 - Temsil/yapı oluşturma",
    "S5 - Örüntü ve genelleme",
],
"3. Algoritma Tasarımı": [
    "AT1 - İşlem/düşünme adımını belirleme",
    "AT2 - Adımları sıralama",
    "AT3 - Ara sonuçları kullanma",
    "AT4 - Prosedürü yapılandırma",
    "AT5 - Akış şemasını oluşturma",
],
"4. Hata Ayıklama": [
    "HA1 - Test etme",
    "HA2 - Problem koşullarıyla karşılaştırma",
    "HA3 - Tutarsızlığı fark etme",
    "HA4 - Sorunlu adımı belirleme",
    "HA5 - Düzeltme",
    "HA6 - Yeniden test etme",
    "HA7 - Uygunsa alternatif çözüm arama",
]
}

METABILISSEL_SORULAR = {
"1. Ayrıştırma": "Problemi geriye doğru parçalarken düşünceni nasıl ilerlettin?",
"2. Soyutlama": "Problemde önemli olan bilgileri ve ilişkileri fark ederken neye dikkat ettin?",
"3. Algoritma Tasarımı": "Akış şemasındaki adımların sırasını belirlerken nasıl düşündün?",
"4. Hata Ayıklama": "Çözümünü kontrol ederken hangi yolu izledin ve neyi yeniden test ettin?",
}

SYSTEM_PROMPT = """
Sen ortaokul matematik öğrencisine rehberlik eden bir öğretmensin.
ÖĞRENCİ ÇÖZER; SEN SADECE DÜŞÜNME SÜRECİNİ YÖNLENDİRİRSİN.

KESİN KURALLAR:
- Çözümü, cevabı, işlemi, formülü, denklemi veya yöntemi verme.
- Öğrencinin yerine hesaplama yapma.
- Öğrencinin yerine düşünce zincirini tamamlama.
- "doğru", "yanlış", "hata yaptın", "burada hata var" ifadelerini değerlendirme amacıyla kullanma.
- Aynı anda yalnızca BİR yönlendirici soru sor.
- 1-2 kısa cümleden fazla yazma.
- Hazır övgü kalıpları kullanma.
- Öğrencinin son cevabını dikkate al.
- Soruyu gerçek problemdeki bilgilerden oluştur.
- Öğrencinin söylemediği matematiksel sonucu onun adına çıkarma.
- Bir sonraki düşünme adımını öğrencinin kendisinin keşfetmesini sağla.
"""

# ============================================================
# GENEL YARDIMCILAR
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_kaydet(data):
    try:
        pd.DataFrame([data]).to_csv(
            DATA_FILE, mode="a", index=False,
            header=not os.path.isfile(DATA_FILE), encoding="utf-8-sig"
        )
    except Exception as e:
        st.warning(f"Veri kaydedilemedi: {e}")

def get_problem_image_part():
    f = st.session_state.get("uploaded_file_data")
    if f is None:
        return None
    try:
        return {"inline_data": {"mime_type": f.type, "data": base64.b64encode(f.getvalue()).decode("utf-8")}}
    except Exception:
        return None

def pil_to_inline_part(image):
    if image is None:
        return None
    try:
        if hasattr(image, "save"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            data = image
        return {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(data).decode("utf-8")}}
    except Exception:
        return None

def canvas_image_to_pil(image_data):
    if image_data is None:
        return None
    try:
        if isinstance(image_data, str):
            raw = image_data.split(",", 1)[1] if "," in image_data else image_data
            return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        if hasattr(image_data, "save"):
            return image_data.convert("RGB")
        # numpy ndarray
        return Image.fromarray(image_data).convert("RGB")
    except Exception:
        return None

def get_annotation_image_part():
    return pil_to_inline_part(canvas_image_to_pil(st.session_state.get("annotation_image_data")))

def get_flowchart_image_part(stage):
    return pil_to_inline_part(canvas_image_to_pil(st.session_state.get("flowchart_image_data", {}).get(stage)))

def get_annotation_background():
    f = st.session_state.get("uploaded_file_data")
    if f is None:
        return None, None, None
    try:
        image = Image.open(io.BytesIO(f.getvalue())).convert("RGB")
        max_width = 900
        w, h = image.size
        dw = min(w, max_width)
        dh = max(1, int(h * dw / w))
        return image.resize((dw, dh), Image.LANCZOS), dw, dh
    except Exception:
        return None, None, None

# ============================================================
# GEMINI
# ============================================================
def gemini_generate(parts, temperature=0.2, max_tokens=250):
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "topP": 0.8, "maxOutputTokens": max_tokens}
    }
    response = requests.post(
        GEMINI_URL, params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"}, json=payload, timeout=60
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

def parse_json_response(raw):
    """Gemini'nin JSON'u bazen markdown kod bloğunda döndürmesini tolere eder."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # JSON gövdesini metin içinde bulmayı dene.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def sanitize_question(q):
    if not q:
        return "Düşünceni bir sonraki adım için nasıl ilerletebilirsin?"
    q = str(q).strip().replace("\n", " ")
    # Öğrenciye görünen yanıt tek soru olmalı.
    q = re.sub(r"^(Soru|Question)\s*:\s*", "", q, flags=re.I).strip()
    # Birden fazla soru işareti varsa ilk soruyu al.
    if "?" in q:
        q = q.split("?", 1)[0].strip() + "?"
    else:
        q = q.rstrip(".") + "?"
    return q

def conversation_text(messages):
    out = []
    for m in messages:
        role = "ÖĞRENCİ" if m["role"] == "user" else "REHBER"
        out.append(f"{role}: {m['content']}")
    return "\n".join(out)

# ============================================================
# PROBLEM ANALİZİ
# ============================================================
def analyze_problem_image(uploaded_file):
    image_data = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    prompt = """
Bu görselde ortaokul matematik problemi var. Problemi ÇÖZME.
Yalnızca rehber öğretmenin kullanacağı problem analizi hazırla:
1. Problem metni
2. Verilen bilgiler
3. İstenen bilgi
4. Tablo/grafik/şekil içeriği
5. Nicelikler
6. Görülebilen ilişkiler
7. Değişen veya birbirine bağlı nicelikler
8. Önemli matematiksel bilgiler
9. Okunamayan/belirsiz bölümler
Tablo ve grafik değerlerini mümkün olduğunca dikkatli oku. İşlem yapma, cevap bulma,
denklem/formül oluşturma, yöntem veya doğrusal ilişkiyi çözüm olarak söyleme.
"""
    return gemini_generate([
        {"text": prompt},
        {"inline_data": {"mime_type": uploaded_file.type, "data": image_data}}
    ], temperature=0.1, max_tokens=500)

# ============================================================
# ÖNCEKİ AŞAMALARDAN KANITLAR
# ============================================================
def stage_summary(stage):
    msgs = st.session_state.chat_storage.get(stage, [])
    if not msgs:
        return "Bu aşamada henüz diyalog yok."
    return conversation_text(msgs)

def previous_stages_context(current_stage):
    idx = STAGES.index(current_stage)
    if idx == 0:
        return "Önceki aşama yok."
    blocks = []
    for s in STAGES[:idx]:
        if s == "5. Metabilişsel Yansıtma":
            continue
        blocks.append(f"===== {s} =====\n{stage_summary(s)}")
    return "\n\n".join(blocks)

# ============================================================
# YAPILANDIRILMIŞ REHBER
# ============================================================
def build_guidance_prompt(current_stage, problem_analysis, chat_history, student_message=None, first=False):
    previous = previous_stages_context(current_stage)
    flowchart_note = ""
    if current_stage in ["3. Algoritma Tasarımı", "4. Hata Ayıklama"]:
        flowchart_note = "Öğrencinin mevcut akış şeması görseli ayrıca gönderilecektir. Görseldeki adımları incele; ancak eksik adımı öğrencinin yerine doldurma."

    if first:
        dialogue = "Öğrenci bu aşamada henüz cevap vermedi."
        task = "Bu aşamayı başlatacak, gerçek probleme dayalı TEK bir soru üret."
    else:
        dialogue = conversation_text(chat_history)
        task = "Öğrencinin son cevabından hareketle yalnızca BİR sonraki bilişsel adımı destekleyen TEK bir soru üret."

    schema = """
YANITI SADECE şu JSON biçiminde ver:
{"alt_beceri":"...", "basamak_tamamlandi":false, "soru":"..."}

Kurallar:
- alt_beceri yalnızca verilen alt becerilerden biri olmalı.
- basamak_tamamlandi yalnızca öğrencinin bu aşamanın bilişsel hedefini gerçekten karşıladığına dair diyalogda yeterli kanıt varsa true olsun.
- Sadece bir soru sor.
- soru 1-2 kısa cümle olsun.
- Öğrenciye çözüm/sonuç/formül/denklem/işlem/yöntem söyleme.
- basamak_tamamlandi=true ise bile soru, öğrencinin son düşüncesini tamamlatan kısa bir son soru olabilir; fakat öğrencinin yerine sonuç üretme.
"""

    return f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK: {current_stage}

BASAMAK PROTOKOLÜ:
{BASAMAK_TALIMATLARI[current_stage]}

BU BASAMAĞIN ALT BECERİLERİ:
{chr(10).join('- '+x for x in ALT_BECERILER[current_stage])}

PROBLEM ANALİZİ:
{problem_analysis}

ÖNCEKİ AŞAMALARDAN GELEN SÜREÇ:
{previous}

{flowchart_note}

MEVCUT DİYALOG:
{dialogue}

ÖĞRENCİNİN SON MESAJI:
{student_message if student_message else 'Henüz yok.'}

ÖZEL GÖREV:
{task}

{schema}
"""

def get_structured_guidance(current_stage, problem_analysis, chat_history, student_message=None, first=False):
    parts = [{"text": build_guidance_prompt(current_stage, problem_analysis, chat_history, student_message, first)}]
    problem_part = get_problem_image_part()
    if problem_part:
        parts.append(problem_part)
    annotation_part = get_annotation_image_part()
    if annotation_part:
        parts.append({"text": "Öğrencinin problem üzerinde yaptığı işaretleme aşağıdaki görseldedir. Yalnızca düşüncesini anlamak için kullan; işaretlemenin anlamını öğrencinin yerine kesinleştirme."})
        parts.append(annotation_part)
    if current_stage in ["3. Algoritma Tasarımı", "4. Hata Ayıklama"]:
        flow_part = get_flowchart_image_part(current_stage)
        if flow_part:
            parts.append({"text": "Öğrencinin bu aşamadaki akış şeması görseli:"})
            parts.append(flow_part)
    raw = gemini_generate(parts, temperature=0.2, max_tokens=220)
    parsed = parse_json_response(raw)
    if parsed is None:
        # Güvenli geri dönüş: metnin kendisini soru kabul et, basamak geçişi yapma.
        return {
            "alt_beceri": ALT_BECERILER[current_stage][0],
            "basamak_tamamlandi": False,
            "soru": sanitize_question(raw)
        }
    alt = parsed.get("alt_beceri")
    if alt not in ALT_BECERILER[current_stage]:
        alt = ALT_BECERILER[current_stage][0]
    completed = bool(parsed.get("basamak_tamamlandi", False))
    return {"alt_beceri": alt, "basamak_tamamlandi": completed, "soru": sanitize_question(parsed.get("soru"))}

# ============================================================
# METABİLİŞSEL / FİNAL
# ============================================================
def final_ozet_olustur(student_id, chat_storage):
    process = "\n\n".join(f"===== {s} =====\n{conversation_text(m)}" for s, m in chat_storage.items() if m)
    prompt = f"""
Sen ortaokul matematik öğretmenisin. Aşağıdaki öğrencinin algoritmik düşünme sürecini değerlendir.
Problemi yeniden çözme ve cevap verme.
Şu boyutları değerlendir: Ayrıştırma, Soyutlama, Algoritma Tasarımı, Hata Ayıklama.
Her boyutta öğrencinin süreçte ne yaptığını kısa ve gözleme dayalı biçimde belirt.
Öğrenci: {student_id}
SÜREÇ:
{process}
"""
    return gemini_generate([{"text": prompt}], temperature=0.3, max_tokens=500)

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "uploaded_file_data": None,
        "problem_analysis": None,
        "chat_storage": {s: [] for s in STAGES[:4]},
        "current_step": STAGES[0],
        "completed_stages": [],
        "annotation_reset": 0,
        "annotation_json": None,
        "annotation_image_data": None,
        "flowchart_storage": {s: None for s in STAGES[:4]},
        "flowchart_image_data": {s: None for s in STAGES[:4]},
        "flowchart_reset": {s: 0 for s in STAGES[:4]},
        "transition_notice": None,
        "final_text": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ============================================================
# AŞAMA GEÇİŞİ
# ============================================================
def advance_stage(current_stage):
    if current_stage not in st.session_state.completed_stages:
        st.session_state.completed_stages.append(current_stage)
    idx = STAGES.index(current_stage)
    if idx < 3:
        nxt = STAGES[idx + 1]
        st.session_state.current_step = nxt
        st.session_state.transition_notice = f"{current_stage} tamamlandı. Şimdi {nxt} basamağına geçiyorsun."
    elif idx == 3:
        st.session_state.current_step = STAGES[4]
        st.session_state.transition_notice = "Hata Ayıklama tamamlandı. Şimdi Metabilişsel Yansıtma basamağına geçiyorsun."
    st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("👨‍🏫 Araştırma Paneli")
    mode = st.selectbox("Giriş Türü:", ["Öğrenci Girişi", "Öğretmen (Admin)"])

    if mode == "Öğretmen (Admin)":
        sifre = st.text_input("Şifre:", type="password")
        if sifre == "tez2024":
            st.success("Admin Paneli Aktif")
            if os.path.isfile(DATA_FILE):
                try:
                    df = pd.read_csv(DATA_FILE, sep=None, engine="python", on_bad_lines="skip")
                    st.write("### 📊 Veri Kayıtları")
                    st.dataframe(df.tail(30), use_container_width=True)
                    st.download_button("📥 Tüm Verileri İndir", df.to_csv(index=False).encode("utf-8-sig"), "tez_data.csv", "text/csv")
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
        "Dikdörtgen (İşlem)": "rect", "Elips (Başla/Bitir)": "circle",
        "Ok/Çizgi": "line", "Serbest Çizim": "freedraw",
        "Düzenle/Taşı": "transform", "Çokgen": "polygon"
    }
    selected_tool = st.selectbox("Araç Seçin:", list(tool_map.keys()))
    drawing_mode = tool_map[selected_tool]
    stroke_color = st.color_picker("Çizgi Rengi:", "#000000")
    fill_color = st.color_picker("Kutu Rengi:", "#EEEEEE")

    st.divider()
    current_index = STAGES.index(st.session_state.current_step)
    selected_step = st.radio("Aşamayı Seçin:", STAGES, index=current_index)
    # Araştırma akışını bozmayacak şekilde yalnızca tamamlanmış veya mevcut aşamaya gidilebilir.
    if selected_step != st.session_state.current_step:
        st.session_state.current_step = selected_step
        st.session_state.transition_notice = None
        st.rerun()

# ============================================================
# BAŞLIK
# ============================================================
st.title("🎯 Algoritmik Problem Çözme Rehberi")
st.write(f"### Mevcut Basamak: {st.session_state.current_step}")

if st.session_state.transition_notice:
    st.success(st.session_state.transition_notice)
    st.session_state.transition_notice = None

# ============================================================
# PROBLEM YÜKLEME
# ============================================================
if st.session_state.uploaded_file_data is None:
    uploaded = st.file_uploader("📷 Soru Fotoğrafı Yükle", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state.uploaded_file_data = uploaded
        with st.spinner("Problem analiz ediliyor..."):
            try:
                st.session_state.problem_analysis = analyze_problem_image(uploaded)
            except Exception as e:
                st.error("Problem analiz edilemedi.")
                st.code(str(e))
                st.session_state.problem_analysis = ""
        st.session_state.chat_storage = {s: [] for s in STAGES[:4]}
        st.session_state.completed_stages = []
        st.session_state.current_step = STAGES[0]
        st.session_state.annotation_json = None
        st.session_state.annotation_image_data = None
        st.session_state.flowchart_storage = {s: None for s in STAGES[:4]}
        st.session_state.flowchart_image_data = {s: None for s in STAGES[:4]}
        st.session_state.flowchart_reset = {s: 0 for s in STAGES[:4]}
        st.rerun()
else:
    st.image(st.session_state.uploaded_file_data, width=900)

    # --------------------------------------------------------
    # PROBLEM ÜZERİNDE İŞARETLEME
    # --------------------------------------------------------
    annotation_image, annotation_width, annotation_height = get_annotation_background()
    if annotation_image is not None:
        st.write("🖍️ **Problem Üzerinde İşaretleme**")
        st.caption("Problem görselinin üzerinde işaretleme yapabilirsin. Bu alan yalnızca senin çizimlerin içindir.")
        c1, c2 = st.columns([3, 1])
        with c1:
            annotation_tool_map = {"Serbest Çizim": "freedraw", "Ok / Çizgi": "line", "Dikdörtgen": "rect", "Elips": "circle"}
            annotation_tool = st.selectbox("İşaretleme aracı:", list(annotation_tool_map.keys()), key="annotation_tool")
        with c2:
            annotation_color = st.color_picker("İşaretleme rengi:", "#FF0000", key="annotation_color")

        annotation_result = st_canvas(
            fill_color="rgba(255,255,255,0)", stroke_color=annotation_color, stroke_width=3,
            background_image=annotation_image, height=annotation_height, width=annotation_width,
            drawing_mode=annotation_tool_map[annotation_tool], update_streamlit=True,
            initial_drawing=st.session_state.annotation_json,
            key=f"problem_annotation_{st.session_state.annotation_reset}"
        )
        # json_data kalıcı temsil; image_data yalnızca güvenli şekilde alınır.
        if annotation_result is not None:
            try:
                if annotation_result.json_data:
                    st.session_state.annotation_json = annotation_result.json_data
            except Exception:
                pass
            try:
                if annotation_result.image_data is not None:
                    st.session_state.annotation_image_data = annotation_result.image_data
            except Exception:
                pass

        if st.button("💾 Problem Üzerindeki İşaretlemeyi Kaydet", key="save_problem_annotation"):
            if st.session_state.annotation_json:
                log_kaydet({"tarih": now_str(), "id": student_id, "basamak": st.session_state.current_step,
                            "tip": "Problem Üzeri Çizim", "icerik": str(st.session_state.annotation_json)})
                st.success("Problem üzerindeki işaretleme kaydedildi.")
            else:
                st.info("Önce problem üzerinde bir işaretleme yap.")
        if st.button("🧹 İşaretlemeyi Temizle", key="clear_problem_annotation"):
            st.session_state.annotation_reset += 1
            st.session_state.annotation_json = None
            st.session_state.annotation_image_data = None
            st.rerun()

    if st.button("❌ Soruyu Değiştir"):
        for key in ["uploaded_file_data", "problem_analysis", "final_text"]:
            st.session_state[key] = None
        st.session_state.chat_storage = {s: [] for s in STAGES[:4]}
        st.session_state.completed_stages = []
        st.session_state.current_step = STAGES[0]
        st.session_state.annotation_reset += 1
        st.session_state.annotation_json = None
        st.session_state.annotation_image_data = None
        st.session_state.flowchart_storage = {s: None for s in STAGES[:4]}
        st.session_state.flowchart_image_data = {s: None for s in STAGES[:4]}
        st.session_state.flowchart_reset = {s: 0 for s in STAGES[:4]}
        st.rerun()

if st.session_state.problem_analysis:
    with st.expander("🔎 Problem analizini göster"):
        st.write(st.session_state.problem_analysis)

st.divider()
col1, col2 = st.columns([1.3, 1], gap="large")

# ============================================================
# SOL: AKIŞ ŞEMASI + METABİLİŞSEL KAYITLAR
# ============================================================
with col1:
    current_stage = st.session_state.current_step
    if current_stage in STAGES[:4]:
        st.write("🖼️ **Tasarım ve Planlama Alanı**")
        if current_stage == "1. Ayrıştırma":
            st.caption("Problemi geriye doğru düşünürken gerekli alt amaçlarını burada not edebilirsin.")
        elif current_stage == "2. Soyutlama":
            st.caption("Önemli bilgiler, ilişkiler ve örüntüler için kullanabilirsin.")
        elif current_stage == "3. Algoritma Tasarımı":
            st.caption("Çözüm algoritmanı akış şeması olarak burada oluştur.")
        else:
            st.caption("Akış şemanı ve çözümünü test ederken burayı kullan.")

        canvas_key = f"canvas_{current_stage.replace(' ', '_')}_{st.session_state.flowchart_reset[current_stage]}"
        canvas_result = st_canvas(
            fill_color=fill_color, stroke_color=stroke_color, stroke_width=3,
            background_color="#ffffff", height=450, drawing_mode=drawing_mode,
            update_streamlit=True,
            initial_drawing=st.session_state.flowchart_storage.get(current_stage),
            key=canvas_key
        )
        if canvas_result is not None:
            try:
                if canvas_result.json_data:
                    st.session_state.flowchart_storage[current_stage] = canvas_result.json_data
            except Exception:
                pass
            try:
                if canvas_result.image_data is not None:
                    st.session_state.flowchart_image_data[current_stage] = canvas_result.image_data
            except Exception:
                pass

        if st.button("🖼️ Tasarımı Kaydet", key=f"save_design_{current_stage}"):
            if st.session_state.flowchart_storage.get(current_stage):
                log_kaydet({
                    "tarih": now_str(), "id": student_id, "basamak": current_stage,
                    "tip": "Cizim", "icerik": str(st.session_state.flowchart_storage[current_stage])
                })
                st.success("Tasarım kaydedildi.")
            else:
                st.info("Önce alanda bir tasarım oluştur.")

    # Metabilişsel cevap / eminlik her aşamada ayrı veri olarak tutulur.
    if current_stage in STAGES[:4]:
        st.write("---")
        st.info("🧠 **Öz-Yansıtma:** " + METABILISSEL_SORULAR[current_stage])
        meta_key = "meta_" + current_stage.replace(" ", "_")
        m_cevap = st.text_area("Düşünceni buraya yaz...", key=meta_key)
        if st.button("💾 Düşüncemi Kaydet", key=f"save_meta_{current_stage}"):
            log_kaydet({"tarih": now_str(), "id": student_id, "basamak": current_stage, "tip": "Metabiliş", "icerik": m_cevap})
            st.success("Kaydedildi!")

        st.write("---")
        confidence = st.select_slider(
            "⭐ **Bu adımdaki çözümünden ne kadar eminsin?**",
            options=["Hiç Emin Değilim", "Kararsızım", "Biraz Eminim", "Çok Eminim"],
            value="Kararsızım", key="confidence_" + current_stage
        )
        if st.button("📈 Eminlik Derecesini Kaydet", key=f"save_conf_{current_stage}"):
            log_kaydet({"tarih": now_str(), "id": student_id, "basamak": current_stage, "tip": "Eminlik", "icerik": confidence})
            st.success(f"Eminlik: {confidence}")

    if current_stage == STAGES[4]:
        st.write("### 🧠 Metabilişsel Yansıtma")
        for s in STAGES[:4]:
            st.info(METABILISSEL_SORULAR[s])
        reflection = st.text_area("Sürecin hakkında düşüncelerini yaz:", height=180, key="final_reflection")
        confidence_final = st.select_slider(
            "Genel çözüm sürecine güvenin:",
            options=["Hiç Emin Değilim", "Kararsızım", "Biraz Eminim", "Çok Eminim"],
            value="Kararsızım", key="confidence_final"
        )
        if st.button("💾 Yansıtmayı Kaydet", key="save_final_reflection"):
            log_kaydet({"tarih": now_str(), "id": student_id, "basamak": current_stage, "tip": "Metabiliş", "icerik": reflection})
            log_kaydet({"tarih": now_str(), "id": student_id, "basamak": current_stage, "tip": "Eminlik", "icerik": confidence_final})
            st.success("Yansıtma kaydedildi.")

        if st.button("🏁 Çözümü Bitir ve Özetini Al", key="finish_solution"):
            with st.spinner("Süreç analiz ediliyor..."):
                try:
                    st.session_state.final_text = final_ozet_olustur(student_id, st.session_state.chat_storage)
                    log_kaydet({"tarih": now_str(), "id": student_id, "basamak": "FİNAL", "tip": "Final Özeti", "icerik": st.session_state.final_text})
                except Exception as e:
                    st.error("Özet hazırlanamadı.")
                    st.code(str(e))
        if st.session_state.final_text:
            st.success("Süreç değerlendirmesi hazır.")
            st.write(st.session_state.final_text)

# ============================================================
# SAĞ: REHBER BOT
# ============================================================
with col2:
    st.write("💬 **Rehber Bot**")
    current_stage = st.session_state.current_step

    if current_stage in STAGES[:4] and st.session_state.problem_analysis:
        history = st.session_state.chat_storage[current_stage]

        # İlk soru: gerçek problem + mevcut aşama.
        if not history:
            with st.spinner("Rehber hazırlanıyor..."):
                try:
                    result = get_structured_guidance(
                        current_stage, st.session_state.problem_analysis, [], first=True
                    )
                    history.append({"role": "assistant", "content": result["soru"], "alt_beceri": result["alt_beceri"]})
                    log_kaydet({
                        "tarih": now_str(), "id": student_id, "basamak": current_stage,
                        "tip": "Bot", "alt_beceri": result["alt_beceri"],
                        "basamak_tamamlandi": result["basamak_tamamlandi"], "icerik": result["soru"]
                    })
                    st.rerun()
                except Exception as e:
                    st.error("İlk soru oluşturulamadı.")
                    st.code(str(e))

        chat_container = st.container(height=550)
        for message in st.session_state.chat_storage[current_stage]:
            with chat_container.chat_message(message["role"]):
                st.write(message["content"])

        student_message = st.chat_input("Düşünceni veya cevabını yaz...", key=f"chat_input_{current_stage}")
        if student_message:
            # Öğrenci mesajı
            st.session_state.chat_storage[current_stage].append({"role": "user", "content": student_message})
            log_kaydet({
                "tarih": now_str(), "id": student_id, "basamak": current_stage,
                "tip": "Öğrenci", "alt_beceri": "", "basamak_tamamlandi": False,
                "icerik": student_message
            })

            # Önceki diyalog: yeni öğrenci mesajını ayrıca gönderiyoruz.
            history_before = st.session_state.chat_storage[current_stage][:-1]
            with st.spinner("Rehber Bot düşünüyor..."):
                try:
                    result = get_structured_guidance(
                        current_stage, st.session_state.problem_analysis,
                        history_before, student_message=student_message, first=False
                    )
                    st.session_state.chat_storage[current_stage].append({
                        "role": "assistant", "content": result["soru"], "alt_beceri": result["alt_beceri"]
                    })
                    log_kaydet({
                        "tarih": now_str(), "id": student_id, "basamak": current_stage,
                        "tip": "Bot", "alt_beceri": result["alt_beceri"],
                        "basamak_tamamlandi": result["basamak_tamamlandi"], "icerik": result["soru"]
                    })

                    # Otomatik geçiş. AI'nin erken geçişini azaltmak için her aşamada minimum öğrenci yanıtı aranır.
                    student_turns = sum(1 for m in st.session_state.chat_storage[current_stage] if m["role"] == "user")
                    min_turns = {"1. Ayrıştırma": 2, "2. Soyutlama": 2, "3. Algoritma Tasarımı": 2, "4. Hata Ayıklama": 2}
                    if result["basamak_tamamlandi"] and student_turns >= min_turns[current_stage]:
                        log_kaydet({
                            "tarih": now_str(), "id": student_id, "basamak": current_stage,
                            "tip": "Basamak Geçişi", "alt_beceri": result["alt_beceri"],
                            "basamak_tamamlandi": True, "icerik": "Basamak otomatik olarak tamamlandı."
                        })
                        advance_stage(current_stage)
                    else:
                        st.rerun()
                except Exception as e:
                    st.error("Gemini yanıt oluşturamadı.")
                    st.code(str(e))

    elif current_stage == STAGES[4]:
        st.info("Dört algoritmik düşünme basamağı tamamlandı. Şimdi kendi düşünme sürecini değerlendir.")
    elif not st.session_state.problem_analysis:
        st.info("Önce bir problem görseli yükle.")
