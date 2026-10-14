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
        "GEMINI_API_KEY bulunamadı. "
        "Streamlit Secrets bölümüne API anahtarınızı ekleyin."
    )
    st.stop()


MODEL_NAME = "gemini-3.5-flash-lite"
APP_VERSION = "v22"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


DATA_FILE = "tez_verileri_final.csv"


# ============================================================
# DOĞRUSAL İLİŞKİLER İÇİN BASAMAK TALİMATLARI
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

Amaç:
Öğrencinin problemi yalnızca düşünme gereksinimleri bakımından
anlamlı parçalara ayırmasını sağlamak.

AYRIŞTIRMA'NIN TEMEL ZİNCİRİ:
İSTENEN → BUNU BULMAK İÇİN GEREKEN BİLGİ →
BU BİLGİYİ BULMAK İÇİN GEREKEN BİLGİ → VERİLEN/TEMEL BİLGİ

Öğrencinin:
- Önce neyin istendiğini belirlemesini,
- İstenen şeyi belirlemek için hangi bilgiye ihtiyaç olduğunu
  kendisinin fark etmesini,
- Gerekli bir bilgi eksikse o bilgiye ulaşmak için hangi
  bilgiye ihtiyaç olduğunu kendisinin düşünmesini,
- Elindeki verilen bilgilerin bu zincirde hangi ihtiyacı
  karşıladığını fark etmesini sağla.

ÇOK ÖNEMLİ:
Ayrıştırma sırasında öğrencinin henüz söylemediği bir veri
kaynağını, niceliği, değeri, süreyi, hızı, işlemi, matematiksel
ilişkiyi, grafik özelliğini veya çözüm adımını sen ortaya çıkarma.
Öğrenciyi B aracına, hıza, süreye, belirli bir grafik noktasına,
eksen kesişimine veya herhangi bir hesaplamaya kendin yönlendirme.

Yasak:
- Çözüm yolu söylemek.
- İşlem önermek.
- Formül vermek.
- Denklem vermek.
- Doğrusal ilişkiyi doğrudan söylemek.
- Cevabı söylemek.
- Hız, süre, değişim miktarı veya başka bir çözüm niceliğini
  öğrencinin yerine ortaya çıkarmak.
- Öğrencinin yerine problemi parçalara ayırmak.
- Öğrencinin henüz söylemediği başka bir veri kaynağını
  (ör. B aracı) soruyla ortaya çıkarmak.
- Sonuca götüren doğrudan ipucu vermek.

""",


    "2. Soyutlama": """

Amaç:
Öğrencinin problemin altında yatan matematiksel ilişkiyi
fark etmesini sağlamak.

Öğrencinin:

- Gerekli bilgileri ayırt etmesini,
- Gereksiz bilgileri fark etmesini,
- Değişen nicelikleri fark etmesini,
- Bir nicelik değiştiğinde diğer nicelikte ne olduğunu
  incelemesini,
- Nicelikler arasındaki ilişkiyi fark etmesini,
- Değişim miktarlarını karşılaştırmasını,
- Önceki matematiksel bilgilerini problemle ilişkilendirmesini,
- Tablo, grafik, sözel ifade veya şekil gibi farklı
  temsiller arasındaki ilişkiyi fark etmesini,
- Uygunsa örüntü veya genel yapı oluşturmasını,
- Özel bir durumdan daha genel bir ilişkiye ulaşmasını
  sağlamaya çalış.

Yasak:

- "Bu doğrusal ilişkidir." demek.
- Doğrusal ilişkiyi öğrencinin yerine belirlemek.
- Denklem vermek.
- Formül vermek.
- Eğim kavramını doğrudan söylemek.
- Sabit değişimi doğrudan söylemek.
- Kullanılacak yöntemi söylemek.
- Çözüm vermek.
- Öğrencinin yerine örüntü oluşturmak.

""",


    "3. Algoritma Tasarımı": """

Amaç:
Öğrencinin doğrusal ilişki içeren problemi çözmek için
kendi düşünme ve işlem planını adım adım oluşturmasını sağlamak. Böylece akış şeması oluşturmak.

Öğrencinin:

- İlk olarak hangi bilgiyi kullanacağını belirlemesini,
- Yapacağı işlemleri kendisinin belirlemesini,
- İşlem ve düşünme adımlarını uygun sıraya koymasını,
- Bir adımdan elde edilen bilginin sonraki adıma
  nasıl katkı sağlayacağını düşünmesini,
- Gerekirse alternatif bir yol oluşturmasını sağla.

Yasak:

- Algoritmayı öğrencinin yerine oluşturmak.
- İşlem sırasını söylemek.
- Denklem vermek.
- Formül vermek.
- Eğim veya sabit terimi doğrudan vermek.
- "Önce şunu bul." şeklinde çözüm adımı vermek.
- Çözümü öğrencinin yerine yapmak.
- Sonucu söylemek.

""",


    "4. Hata Ayıklama": """

Amaç:
Öğrencinin oluşturduğu çözümün ve matematiksel ilişkinin
problem koşullarıyla uyumlu olup olmadığını kendisinin
kontrol etmesini sağlamak.

Öğrencinin:

- Kullandığı bilgileri yeniden kontrol etmesini,
- İşlem adımlarını incelemesini,
- Nicelikler arasındaki ilişkinin tutarlı olup olmadığını
  farklı değerlerle kontrol etmesini,
- Tablo, grafik veya işlem sonuçlarını karşılaştırmasını,
- Değişimin problemde verilen durumla uyumlu olup olmadığını
  incelemesini,
- Sonucun problem koşullarını sağlayıp sağlamadığını
  değerlendirmesini,
- Şüphe duyduğu bir adımı yeniden incelemesini,
- Gerekirse kendi çözümünde düzeltme yapmasını,
- Yaptığı düzeltmeyi yeniden test etmesini sağla.

Yasak:

- "Doğru" demek.
- "Yanlış" demek.
- "Hata yaptın" demek.
- Doğru sonucu söylemek.
- Doğru denklemi söylemek.
- Doğru ilişkiyi öğrencinin yerine kurmak.
- Eksik adımı öğrencinin yerine tamamlamak.

"""
,

    "5. Metabilişsel Yansıtma": """

Amaç:
Öğrencinin tamamladığı algoritmik problem çözme sürecini
metabilişsel açıdan fark etmesini, izlemesini ve değerlendirmesini sağlamak.

Bu basamak üç boyut üzerinden ilerler:

1. PLANLAMA:
- Problemi çözmeye başlamadan önce neyi amaçladığını,
- hangi bilgileri kullanacağını,
- nasıl bir yol izleyeceğini ve neden o yolu seçtiğini
  fark etmesini sağla.

2. İZLEME:
- Çözüm sırasında ne yaptığını,
- kullandığı bilgilerin amacına hizmet edip etmediğini,
- düşünme ve işlem adımlarını nasıl kontrol ettiğini,
- gerektiğinde düşüncesini nasıl değiştirdiğini
  fark etmesini sağla.

3. DEĞERLENDİRME:
- Oluşturduğu çözümün ve kullandığı yolun problem koşullarıyla
  ne ölçüde uyumlu olduğunu,
- hangi kararlarının işe yaradığını,
- neyi farklı yapabileceğini,
- benzer bir problemde neyi yeniden kullanabileceğini
  değerlendirmesini sağla.

İlerleme sırası:
PLANLAMA → İZLEME → DEĞERLENDİRME

Yasak:
- Öğrencinin yerine kendi sürecini yorumlamak.
- "Şurada iyi plan yaptın." gibi değerlendirme yapmak.
- Öğrencinin neden belirli bir adımı seçtiğini onun yerine açıklamak.
- Çözümü yeniden yapmak.
- Yeni bir çözüm yolu öğretmek.
- "Doğru/yanlış" gibi doğrudan değerlendirme ifadeleri kullanmak.

Bu basamakta amaç yeni bir matematiksel çözüm üretmek değil,
öğrencinin kendi düşünme sürecini fark etmesi ve değerlendirmesidir.

"""
}


# ============================================================
# ALGORİTMİK DÜŞÜNME ALT BECERİLERİ
# ============================================================

ALT_BECERILER = {
    "1. Ayrıştırma": [
        ("A1", "Alt amacı belirleme"),
        ("A2", "Parçalara ayırma"),
        ("A3", "İlişkileri belirleme"),
    ],
    "2. Soyutlama": [
        ("S1", "İlgili bilgiyi belirleme"),
        ("S2", "Gereksiz bilgiyi ayırt etme"),
        ("S3", "Nicelikler arası ilişkileri belirleme"),
        ("S4", "Uygun temsil oluşturma"),
        ("S5", "Örüntü ve genel yapı oluşturma"),
    ],
    "3. Algoritma Tasarımı": [
        ("AT1", "İşlem adımlarını belirleme"),
        ("AT2", "Adımları sıralama"),
        ("AT3", "Ara sonuçları kullanma"),
        ("AT4", "Prosedür oluşturma"),
        ("AT5", "Uygun alternatif yolu değerlendirme"),
    ],
    "4. Hata Ayıklama": [
        ("HA1", "Çözümü test etme"),
        ("HA2", "Koşullarla karşılaştırma"),
        ("HA3", "Tutarsızlığı fark etme"),
        ("HA4", "Sorunlu adımı belirleme"),
        ("HA5", "Düzeltme yapma"),
        ("HA6", "Yeniden test etme"),
    ],
    "5. Metabilişsel Yansıtma": [
        ("MY1", "Planlama sürecini fark etme"),
        ("MY2", "Çözüm sürecini izleme"),
        ("MY3", "Çözüm sürecini değerlendirme"),
    ],
}


# ============================================================
# METABİLİŞSEL SORULAR
# ============================================================

METABILISSEL_SORULAR = {

    "1. Ayrıştırma":
        "Bu problemi parçalara ayırırken hangi bilgiler ve nicelikler dikkatini çekti?",

    "2. Soyutlama":
        "Bu problemde nicelikler arasındaki ilişkiyi fark ederken hangi bilgiler sana yardımcı oldu?",

    "3. Algoritma Tasarımı":
        "Çözüm adımlarını planlarken nasıl bir yol izledin ve bu sırayı neden seçtin?",

    "4. Hata Ayıklama":
        "Kurduğun ilişkinin ve bulduğun sonucun problem koşullarıyla uyumlu olduğundan nasıl emin oldun?",

    "5. Metabilişsel Yansıtma":
        "Bu problemde düşünme sürecini planlama, izleme ve değerlendirme açısından nasıl değerlendiriyorsun?"
}


# ============================================================
# ANA SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

Sen ortaokul matematik öğrencisine rehberlik eden
bir matematik öğretmenisin.

SÜRÜM: v22 — öğrenci yanıtını sınıflandır; bağlama özgü takip sorusu üret ve bilgi tekrarını engelle.

KONU:
DOĞRUSAL İLİŞKİLER

TEMEL İLKE:

ÖĞRENCİ ÇÖZER.

SEN YALNIZCA ÖĞRENCİNİN DÜŞÜNME SÜRECİNİ
YÖNLENDİRİRSİN.

Sen problemi öğrencinin yerine çözmezsin.

============================================================
KESİN KURALLAR
============================================================

1. Asla doğrudan çözüm verme.

2. Asla cevap verme.

3. Asla işlem sonucunu söyleme.

4. Asla öğrencinin yerine hesaplama yapma.

5. Asla formülü doğrudan verme.

6. Asla denklem oluşturup öğrenciye verme.

7. Asla eğimi öğrencinin yerine bulma.

8. Asla sabit değişim miktarını öğrencinin yerine belirleme.

9. Asla doğrusal ilişkiyi doğrudan söyleme.

10. Öğrencinin yerine algoritma oluşturma.

11. Öğrencinin hatasını doğrudan söyleme.

12. "Doğru" ifadesini değerlendirme amacıyla kullanma.

13. "Yanlış" ifadesini kullanma.

14. "Hata yaptın" ifadesini kullanma.

15. Aynı anda birden fazla soru sorma.

16. Her mesajda YALNIZCA BİR yönlendirici soru sor.

17. Öğrencinin cevabını bekle.

18. Uzun açıklamalar yapma.

19. Liste halinde çözüm verme.

20. Öğrencinin düşüncesini kendin tamamlamaya çalışma.

21. Öğrencinin söylediğini gereksiz yere tekrar etme.

22. Hazır övgü kalıpları kullanma.

23. "Harika bir başlangıç noktası!",
    "Çok güzel!",
    "Mükemmel!" gibi kalıpları kullanma.

24. Öğrencinin seviyesine uygun kısa ve doğal Türkçe kullan.

============================================================
YANIT BİÇİMİ
============================================================

- Türkçe yaz.
- 1 veya 2 kısa cümle kullan.
- YALNIZCA BİR soru sor.
- Soruyu öğrencinin son cevabına göre oluştur.
- Öğrencinin verdiği cevabı dikkate al.
- Bir sonraki düşünme adımına yönlendir.
- Gereksiz açıklama yapma.

============================================================
DOĞRUSAL İLİŞKİLER İÇİN UYGUN SORU TÜRLERİ
============================================================

"Problemde hangi nicelikler arasında bir ilişki görüyorsun?"

"Problemde hangi nicelik değişiyor?"

"Bir nicelik değiştiğinde diğerinde ne oluyor?"

"Bu iki nicelik arasında nasıl bir ilişki fark ettin?"

"Değerler arasındaki değişimde nasıl bir düzen görüyorsun?"

"Tablodaki değerler arasında nasıl bir ilişki görüyorsun?"

"Grafikte hangi niceliklerin birbiriyle ilişkili olduğunu düşünüyorsun?"

"Bu bilgiyi başka hangi gösterimle ifade edebilirsin?"

"Farklı değerler için aynı ilişki devam ediyor mu?"

"Bu ilişkiyi başka bir değer kullanarak nasıl kontrol edebilirsin?"

"İlk adımının ne olması gerektiğine nasıl karar verdin?"

"Bu adımı neden seçtin?"

"Bir sonraki adımına geçmek için hangi bilgiye ihtiyacın var?"

"Kuruduğun ilişkiyi problemdeki başka bir değerle nasıl sınayabilirsin?"

============================================================
AYRIŞTIRMA İÇİN SORU BANKASI SINIRI
============================================================

Aşağıdaki genel soru türleri yalnızca ilgili basamaklarda kullanılabilir.
MEVCUT BASAMAK "1. Ayrıştırma" ise bu genel soru bankasını kullanma.
Özellikle ilişki, değişim, hız, süre, işlem, hesaplama, grafik
özelliği veya çözüm adımı çağrıştıran sorular üretme.

Ayrıştırmada soru yönü yalnızca geriye doğru olabilir:
- Ne isteniyor?
- Bunu belirlemek için hangi bilgi gerekiyor?
- Bu bilgiye ulaşmak için hangi bilgi gerekiyor?
- Elindeki verilenlerden hangisi bu ihtiyacı karşılıyor?
- Şu ana kadarki bilgiler yeterli mi?
- Eksik olduğunu düşündüğün bilgi nedir?

Öğrenci bir bilgi söylediğinde o bilgi zincirde bir HALKA olarak kabul edilir.
AI o bilgiyi yeniden aratamaz ve öğrencinin söylemediği bir sonraki halkayı
sorunun içine yerleştiremez.

============================================================
ASLA KULLANMA
============================================================

"Bu doğrusal ilişkidir."

"Bu doğrusal bir fonksiyondur."

"y = ax + b kullan."

"Eğimi bulmalısın."

"Önce eğimi bul."

"Sabit değişimi hesapla."

"Denklemi yaz."

"Doğrunun eğimini hesapla."

"Şu formülü kullan."

"Şu işlemi yap."

"Cevap ..."

"Doğru cevap ..."

"Yanlış yaptın."

"Burada hata var."

============================================================

Görseldeki problemi dikkatle incele.

Sorularını GENEL bir matematik problemi üzerinden değil,
yüklenen GERÇEK problem üzerinden oluştur.

Örneğin problemde tablo varsa tabloya,
grafik varsa grafiğe,
şekil varsa şekle,
belirli nicelikler varsa o niceliklere
dayalı soru sor.

Ancak çözümü öğrencinin yerine yapma.

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

        st.warning(
            f"Veri kaydedilemedi: {e}"
        )


# ============================================================
# PROBLEM GÖRSELİNİ GEMINI'YE HAZIRLAMA
# ============================================================

def get_problem_image_part():

    uploaded_file = (
        st.session_state.get(
            "uploaded_file_data"
        )
    )

    if uploaded_file is None:
        return None

    try:

        image_data = base64.b64encode(
            uploaded_file.getvalue()
        ).decode("utf-8")

        mime_type = uploaded_file.type

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": image_data
            }
        }

    except Exception:

        return None


# ============================================================
# GEMINI İSTEK FONKSİYONU
# ============================================================

def image_part_from_canvas(image_data):

    if image_data is None:
        return None

    try:
        if hasattr(image_data, "save"):
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            raw = buffer.getvalue()
        else:
            image = Image.fromarray(image_data)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()

        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(raw).decode("utf-8")
            }
        }
    except Exception:
        return None


def safe_get_canvas_image_data(canvas_result):

    if canvas_result is None:
        return None

    try:
        return canvas_result.image_data
    except Exception:
        return None


def gemini_generate(
    parts,
    temperature=0.2,
    max_tokens=250,
    include_problem_image=False,
    include_flowchart_image=False
):

    final_parts = []
    final_parts.extend(parts)

    if include_problem_image:
        image_part = get_problem_image_part()
        if image_part is not None:
            final_parts.append(image_part)

    if include_flowchart_image:
        flowchart_image = st.session_state.get("flowchart_image_data")
        image_part = image_part_from_canvas(flowchart_image)
        if image_part is not None:
            final_parts.append(image_part)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": final_parts
            }
        ],
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
        raise Exception(
            f"Gemini API Hatası ({response.status_code}): {detail}"
        )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise Exception(f"Gemini beklenmeyen yanıt verdi: {data}")


def parse_guidance_response(raw_text, current_step):
    """Gemini'nin yapılandırılmış yanıtını güvenli biçimde çözer."""

    fallback_skill = ALT_BECERILER[current_step][0][0]
    fallback_question = raw_text.strip()

    # Markdown JSON çevresini temizle
    cleaned = fallback_question.replace("```json", "").replace("```", "").strip()

    # Model bazen JSON'un önüne/arkasına kısa bir ifade ekleyebilir.
    # Bu durumda yalnızca ilk JSON nesnesini almaya çalış.
    if not cleaned.startswith("{"):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

    try:
        data = json.loads(cleaned)
        question = str(data.get("soru", "")).strip()
        alt_beceri = str(data.get("alt_beceri", fallback_skill)).strip()
        completed = bool(data.get("basamak_tamamlandi", False))

        if not question:
            raise ValueError("soru alanı boş")

        valid_codes = {code for code, _ in ALT_BECERILER[current_step]}
        if alt_beceri not in valid_codes:
            alt_beceri = fallback_skill

        return {
            "soru": question,
            "alt_beceri": alt_beceri,
            "basamak_tamamlandi": completed
        }
    except Exception:
        return {
            "soru": fallback_question,
            "alt_beceri": fallback_skill,
            "basamak_tamamlandi": False
        }


def _normalize_text(text):
    """Tekrar kontrolü için Türkçe metni sadeleştirir."""
    text = str(text or "").casefold()
    text = text.translate(str.maketrans({"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _last_assistant_question(chat_history):
    for message in reversed(chat_history or []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _affirmative_reply(text):
    normalized = _normalize_text(text)
    return normalized in {
        "evet", "evet yeterli", "yeterli", "sanirim evet", "bence yeterli",
        "hayir", "hayır", "hayir yeterli degil", "yeterli degil", "değil"
    }


def _previous_student_message(chat_history):
    """Mevcut yanıttan önceki en son öğrenci mesajını döndürür."""
    for message in reversed(chat_history or []):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _question_seen_before(chat_history, question):
    """Aynı veya çok benzer bir sorunun daha önce sorulup sorulmadığını denetler."""
    q = _normalize_text(question)
    if not q:
        return False
    q_words = set(q.split())
    for message in (chat_history or []):
        if message.get("role") != "assistant":
            continue
        old = _normalize_text(message.get("content", ""))
        if old == q:
            return True
        old_words = set(old.split())
        if len(q_words | old_words) and len(q_words & old_words) / len(q_words | old_words) > 0.78:
            return True
    return False


def _ayristirma_guard(result, chat_history, student_message):
    """Ayrıştırmada yanıt türünü dikkate alır; bilgi tekrarını ve çözüm adımına kaymayı engeller."""
    question = str(result.get("soru", "")).strip()
    previous_question = _last_assistant_question(chat_history)
    previous_student = _previous_student_message(chat_history)
    q_norm = _normalize_text(question)
    prev_norm = _normalize_text(previous_question)
    student_norm = _normalize_text(student_message)
    previous_student_norm = _normalize_text(previous_student)

    # 1) Öğrenci grafikteki bilgiyi zaten gösterdiğini söylüyorsa veriyi yeniden aratma.
    graph_already_seen = any(phrase in student_norm for phrase in (
        "grafikte yaziyor zaten", "grafikte var zaten", "grafikte yaziyor",
        "grafikte var", "grafikte gorunuyor", "grafikte goruluyor",
        "soruda yaziyor zaten", "verilenlerde yaziyor"
    ))
    if graph_already_seen:
        # Önceki öğrenci mesajında somut olarak adlandırılmış verileri takip sorusuna taşı.
        if ("2 saat" in previous_student_norm or "iki saat" in previous_student_norm) and (
            "uzak" in previous_student_norm or "mesafe" in previous_student_norm
        ):
            tailored = (
                "Başlangıçtaki ve 2 saat sonraki uzaklıkları seçtin. "
                "Bu iki bilgiyi kullanarak hangi bilgiyi belirlemeyi amaçlıyorsun?"
            )
        elif "baslangic" in previous_student_norm and ("uzak" in previous_student_norm or "mesafe" in previous_student_norm):
            tailored = (
                "Başlangıç uzaklığını belirttin. Bu bilgi, problemde aradığın uzaklıkla "
                "nasıl bir bağlantı kuruyor?"
            )
        elif previous_student:
            tailored = (
                f"Az önce belirttiğin ‘{previous_student[:110]}’ bilgisini düşün. "
                "Bu bilgi, belirlemek istediğin şeyle nasıl bağlantılı?"
            )
        else:
            tailored = "Grafikte gördüğün bilgiyi tekrar aramana gerek yok. Bu bilgi, problemde aradığın şeyle nasıl bağlantılı?"
        result["soru"] = tailored
        result["basamak_tamamlandi"] = False
        return result

    # 2) Kısa evet/hayır yanıtını yalnızca hemen önceki soruya verilmiş cevap olarak yorumla.
    if _affirmative_reply(student_message) and any(
        phrase in prev_norm for phrase in ("yeterli mi", "yeterli degil", "eksik mi")
    ):
        if student_norm.startswith("evet") or student_norm == "yeterli":
            result["soru"] = "Belirlediğin bilgilerden hangisinin, aradığın şeyi açıklamada en önemli olduğunu düşünüyorsun?"
        else:
            result["soru"] = "Mevcut bilgilerle ilgili hangi noktanın eksik kaldığını düşünüyorsun?"
        result["basamak_tamamlandi"] = False
        return result

    # 3) Öğrenci işlem önerdiyse hesaplamaya geçme; işlemle hangi bilgiyi aradığını netleştir.
    solution_markers = (
        "farkini al", "toplarim", "cikaririm", "bolerim", "carparim",
        "hesaplarim", "islem yap", "formul", "denklem", "hizini bul",
        "egim", "bolme yap", "carpma yap", "oranini bul"
    )
    if any(marker in student_norm for marker in solution_markers):
        result["soru"] = (
            "Bu işlemle hangi bilgiyi belirlemeye çalıştığını kendi sözlerinle açıklayabilir misin?"
        )
        result["basamak_tamamlandi"] = False
        return result

    # 4) Öğrenci somut veri veya nicelik söylediyse onu yeniden isteme; sonraki düşünme halkasını sor.
    has_concrete_data = any(token in student_norm for token in (
        "saat", "dakika", "km", "kilometre", "uzaklik", "mesafe", "grafik",
        "baslangic", "sonraki", "arac", "araci", "veri", "nokta", "deger"
    ))
    if has_concrete_data and any(phrase in prev_norm for phrase in (
        "hangi veri", "hangi bilgiyi", "hangi bilgiye", "hangi deger", "hangi değeri"
    )):
        result["soru"] = (
            "Belirttiğin bu bilgiyi, problemde istenen şeye ulaşmak için nasıl bir ihtiyaçla ilişkilendiriyorsun?"
        )
        result["basamak_tamamlandi"] = False
        return result

    # 5) Ayrıştırma basamağında çözüm/işlem çağrıştıran soruları engelle.
    forbidden_question_markers = (
        "hangi adimi atman", "hangi islemi", "nasil hesap", "hangi formul",
        "egim", "hizini", "sureyi bul", "degisim miktari", "denklemi yaz",
        "islem yap", "sonucu bul", "hesaplayabilir misin"
    )
    if any(marker in q_norm for marker in forbidden_question_markers):
        result["soru"] = "Belirlediğin bilgilerden hangisinin, aradığın şeyi açıklamada önemli olduğunu düşünüyorsun?"
        result["basamak_tamamlandi"] = False
        return result

    # 6) Aynı veya benzer soru tekrar ediyorsa önceki yanıt türüne göre farklı bir takip sorusu seç.
    if _question_seen_before(chat_history, question):
        if any(token in student_norm for token in ("b ile", "b araci", "b aracı", "b'nin", "b nin")):
            result["soru"] = "B ile ilgili belirttiğin bilgilerin, aradığın uzaklığı açıklamak açısından neyi netleştirdiğini düşünüyorsun?"
        elif any(token in student_norm for token in ("yeterli", "eksik", "bilmiyorum", "emin degilim")):
            result["soru"] = "Şu ana kadar belirlediklerinden hangisi hâlâ açıklanmaya ihtiyaç duyuyor?"
        elif has_concrete_data:
            result["soru"] = "Belirttiğin bilgiyi, problemde istenenle ilişkilendirdiğinde hangi bağlantıyı kuruyorsun?"
        else:
            result["soru"] = "Şu ana kadar söylediklerinden hangisi, neyi bulmamız gerektiğini anlamana yardımcı oluyor?"
        result["basamak_tamamlandi"] = False

    return result


# ============================================================
# PROBLEM GÖRSELİNİ ANALİZ ET
# ============================================================

def analyze_problem_image(uploaded_file):

    if uploaded_file is None:
        return ""

    image_data = base64.b64encode(
        uploaded_file.getvalue()
    ).decode("utf-8")

    prompt = """

Bu görselde bir ortaokul matematik problemi var.

Görseli ÇÖZME.

Yalnızca problemi rehber öğretmenin
anlayabilmesi için analiz et.

Şunları belirle:

1. Problem metni.

2. Verilen bilgiler.

3. İstenen bilgi.

4. Tablo, grafik veya şekil varsa
   bunların içeriği.

5. Problemde yer alan nicelikler.

6. Nicelikler arasındaki görülebilen ilişkiler.

7. Değişen veya birbirine bağlı nicelikler.

8. Önemli matematiksel bilgiler.

9. Okunamayan veya belirsiz bölümler.

Özellikle tablo veya grafik varsa,
üzerindeki değerleri mümkün olduğunca
doğru şekilde oku.

Çözüm yapma.

İşlem yapma.

Cevap bulma.

Denklem oluşturma.

Eğim hesaplama.

Doğrusal ilişkiyi çözüm olarak belirtme.

Kullanılacak yöntemi söyleme.

"""


    parts = [

        {
            "text":
                prompt
        },

        {
            "inline_data": {

                "mime_type":
                    uploaded_file.type,

                "data":
                    image_data
            }
        }
    ]


    return gemini_generate(

        parts,

        temperature=0.1,

        max_tokens=500,

        include_problem_image=False
    )


# ============================================================
# İLK SORUYU OLUŞTUR
# ============================================================

def conversation_text(chat_history):

    text = ""
    for message in chat_history:
        role = "ÖĞRENCİ" if message["role"] == "user" else "REHBER"
        text += f"{role}: {message['content']}\n"
    return text


def previous_stages_text(current_step):

    stages = list(BASAMAK_TALIMATLARI.keys())
    current_index = stages.index(current_step)
    text = ""

    for stage in stages[:current_index]:
        history = st.session_state.chat_storage.get(stage, [])
        if history:
            text += f"\n===== {stage} =====\n"
            text += conversation_text(history)

    return text


def ilk_soruyu_olustur(current_step, problem_analysis):

    if current_step == "1. Ayrıştırma":
        progression = """
AYRIŞTIRMA İÇİN ÖZEL İLERLEME:
Öğrenciyi yalnızca GERİYE DOĞRU düşünme zincirinde ilerlet.

Zincir:
İSTENEN → GEREKEN BİLGİ → O BİLGİ İÇİN GEREKEN BİLGİ → VERİLEN/TEMEL BİLGİ

İlk soru yalnızca "Ne isteniyor?" veya doğrudan buna eşdeğer,
probleme özgü bir soru olabilir.

Öğrencinin söylemediği sonraki halkayı ilk soruda veya herhangi bir
soruda sen ortaya çıkarma. B, A, hız, süre, değişim, grafik noktası,
eksen, işlem gibi çözüm yönü oluşturan ayrıntıları öğrenci söylemeden
soruna yerleştirme.
"""
    elif current_step == "2. Soyutlama":
        progression = """
SOYUTLAMA İÇİN ÖZEL İLERLEME:
Ayrıştırmada ortaya çıkan bilgileri temel al.
Önce hangi bilgilerin gerçekten gerekli olduğunu, hangilerinin gereksiz olabileceğini düşündür.
Sonra nicelikleri, ilişkileri, değişimleri ve uygun ise örüntü/genel yapıyı fark ettir.
Öğrencinin önceki aşamadaki düşüncelerini yeniden düzenlemesine yardımcı ol; ilişkiyi onun yerine söyleme.
"""
    elif current_step == "3. Algoritma Tasarımı":
        progression = """
ALGORİTMA TASARIMI İÇİN ÖZEL İLERLEME:
Ayrıştırma ve Soyutlama aşamalarındaki öğrenci düşüncelerini kullan.
Öğrencinin çözüm düşüncesini adım adım akış şemasına dönüştürmesine rehberlik et.
İlk adımı, sonraki adımı veya işlemi öğrencinin yerine söyleme.
Akış şeması görselini de incele ve soruyu mevcut çizime bağla.
"""
    elif current_step == "4. Hata Ayıklama":
        progression = """
HATA AYIKLAMA İÇİN ÖZEL İLERLEME:
Öğrencinin oluşturduğu akış şemasını ve önceki aşamalardaki düşüncelerini incele.
Önce çözümü test ettir. Bir tutarsızlık seziliyorsa öğrencinin kendisinin ilgili noktayı bulmasını sağla; ardından düzeltmesini ve yeniden test etmesini iste.
Sorun görünmüyorsa ve yaş düzeyine uygun başka bir yol gerçekten mümkünse alternatif yolu düşündür; alternatif yöntem zorunlu değildir.
"""
    else:
        progression = """
METABİLİŞSEL YANSITMA İÇİN ÖZEL İLERLEME:

Bu basamak yalnızca öğrencinin kendi düşünme sürecine dönük yansıtma içindir.
Yeni bir matematiksel çözüm öğretme veya çözümü yeniden yaptırma.

Sıra:
PLANLAMA → İZLEME → DEĞERLENDİRME

Önce PLANLAMA: öğrencinin çözüm öncesinde amacını, kullanmayı düşündüğü
bilgileri ve izleyeceği yolu nasıl belirlediğini fark ettir.
Sonra İZLEME: öğrencinin çözüm sırasında kendi düşünmesini, kullandığı
bilgileri ve adımlarını nasıl kontrol ettiğini fark ettir.
Son olarak DEĞERLENDİRME: öğrencinin kullandığı yolun ve sonucunun süreç
açısından ne kadar işe yaradığını, neyi farklı yapabileceğini ve benzer
bir problemde neyi yeniden kullanabileceğini değerlendirmesini sağla.

Bir seferde yalnızca tek boyuta odaklan. Öğrenci bir boyutu açıklamadan
diğerine geçme. Öğrencinin yerine onun düşünme sürecini yorumlama.
"""

    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAĞIN AKADEMİK AMACI:
{BASAMAK_TALIMATLARI[current_step]}

ALT BECERİLER:
{ALT_BECERILER[current_step]}

{progression}

AŞAMA ÖNCELİĞİ:
Mevcut basamak 1. Ayrıştırma ise, öğrencinin henüz çözüm yapmasına izin verme. İlk soru yalnızca istenen veya istenene ulaşmak için gerekli bilgi zincirini başlatmalıdır; grafik üzerindeki çözüm yolunu, matematiksel ilişkiyi veya hesaplamayı hedefleme.

Mevcut basamak 5. Metabilişsel Yansıtma ise, yeni matematiksel içerik öğretme. Öğrencinin tamamladığı süreci PLANLAMA → İZLEME → DEĞERLENDİRME sırasıyla kendi ifadeleri üzerinden fark etmesini ve değerlendirmesini sağla.

PROBLEM ANALİZİ:
{problem_analysis}

ÖNEMLİ:
Yüklenen gerçek problem görselini incele.
İlk soru BU PROBLEMDEKİ gerçek bilgilere dayanmalıdır.
Genel, her probleme uyabilecek mekanik bir soru sorma.

YANITI YALNIZCA ŞU JSON YAPISINDA VER:
{{
  "alt_beceri": "...",
  "basamak_tamamlandi": false,
  "soru": "..."
}}

İlk soru için basamak_tamamlandi her zaman false olsun.
Soru alanında yalnızca tek kısa yönlendirici soru bulunsun.
"""

    raw = gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=150,
        include_problem_image=True,
        include_flowchart_image=(current_step in ["3. Algoritma Tasarımı", "4. Hata Ayıklama", "5. Metabilişsel Yansıtma"])
    )

    return parse_guidance_response(raw, current_step)


def ogrenciye_cevap_ver(current_step, problem_analysis, chat_history, student_message):

    conversation = conversation_text(chat_history)
    previous = previous_stages_text(current_step)

    if current_step == "1. Ayrıştırma":
        progression = """
ÖZEL KURAL — AYRIŞTIRMA / GERİYE DOĞRU ZİNCİR:

BU AŞAMADA ÇÖZÜM YAPILMAZ.
Amaç yalnızca öğrencinin problemi şu zincirde kendi başına
parçalamasıdır:

İSTENEN
→ BUNU BULMAK İÇİN GEREKEN BİLGİ
→ BU BİLGİYİ BULMAK İÇİN GEREKEN BİLGİ
→ VERİLEN/TEMEL BİLGİ

KATI KURAL:
Öğrencinin henüz söylemediği bir sonraki halkayı SEN ortaya çıkaramazsın.
Soru, öğrencinin düşünmesi gereken halkayı isim olarak içinde taşımamalıdır.

ÖRNEĞİN:
Öğrenci "A'nın başlangıç uzaklığını bulmamız gerekiyor" dediyse,
"bunu bulmak için hangi bilgiye ihtiyacın var?" denebilir.

Ama öğrenci "A'nın verileri yetmiyor" demeden:
- B aracını,
- B'nin verilerini,
- hızı,
- süreyi,
- aynı süreyi,
- değişim miktarını,
- belirli bir grafik noktasını,
- ekseni/kesişimi,
- herhangi bir işlemi
soruya sokma.

ÖĞRENCİ CEVABINI ÖNCE SINIFLANDIR:
1. Önceki soruya evet/hayır mı dedi? Bu kısa cevap yalnızca önceki soruyu yanıtlar.
2. Yeni bir bilgi veya nicelik mi belirtti?
3. Bir bilginin eksik olduğunu mu söyledi?
4. Önceki düşüncesini düzeltti mi?
5. İşlem/çözüm önerisine mi geçti?
6. Soruyu anlamadığını mı söyledi?

ÖNCEKİ SORU-CEVAP BAĞI ZORUNLUDUR:
Öğrencinin kısa cevabını mutlaka hemen önceki rehber sorusuyla birlikte yorumla.
Örneğin rehber "Elindeki veriler yeterli mi?" diye sorduysa, "evet" cevabı
bu soruya verilmiş bir yanıttır; yeni bir işlem isteme veya "hangi adımı
atman gerekiyor?" diye çözüm planına geçme gerekçesi değildir.

Sonra yalnızca öğrencinin gerçekten söylediği içerikten hareket et.

BİLGİ TEKRARI YASAĞI:
- Öğrenci "3 saatte 252 km" dediyse bunu tekrar sorma.
- Öğrenci "B'nin 2 saatte 340'tan 204'e düştüğünü" söylediyse
  bunu tekrar aratma.
- Öğrenci "başka A bilgisi yok" dediyse, bunu "A bilgisini bulamadı"
  şeklinde yorumlama.
- Öğrencinin söylediği bir veri zaten mevcutsa "başka hangi veri...",
  "hangi saat...", "grafikte ne görüyorsun?" gibi yeni veri avına çıkma.

AYRIŞTIRMADA KESİNLİKLE SORMA:
- Nasıl hesaplanır?
- Hangi işlem yapılır?
- Hangi formül kullanılır?
- Hangi ilişkiyi kullanırsın?
- Hızını nasıl bulursun?
- Süreyi nasıl bulursun?
- Hangi saat bilgisini kullanırsın?
- B aracından nasıl yararlanırsın?
- Grafikte hangi noktaya bakarsın?
- Değişim miktarı nedir?
- Eğim nedir?
- Denklem nasıl bulunur?
- Grafikten sonucu nasıl bulursun?

Bunların tamamı çözüm/soyutlama/algoritma yönlendirmesidir.

Öğrenci bir sonraki halkayı kendisi söylediğinde onu tamamlanmış kabul et.
Sonraki soru yalnızca henüz söylenmemiş bir sonraki GERİYE DOĞRU halkayı
düşündürsün.

Öğrenci yeterli ayrıştırmayı kendi ifadeleriyle tamamladıysa
basamak_tamamlandi=true olabilir; sırf çözüm yoluna ulaşıldığı için
Ayrıştırma tamamlanmış sayma.
"""
    elif current_step == "2. Soyutlama":
        progression = """
ÖZEL KURAL — SOYUTLAMA:
Önceki Ayrıştırma konuşmasını kullan.
Öğrencinin belirlediği bilgileri gerekli/gereksiz, önemli/ikincil, değişen/sabit ve ilişkili/ilişkisiz yönlerden yeniden düşünmesini sağla.
Uygunsa örüntü veya genel yapı arat; uygun değilse zorunlu olarak örüntü isteme.
İlişkiyi öğrencinin yerine adlandırma.
"""
    elif current_step == "3. Algoritma Tasarımı":
        progression = """
ÖZEL KURAL — ALGORİTMA TASARIMI:
Ayrıştırma ve Soyutlama konuşmalarını doğrudan kullan.
Öğrencinin akış şemasındaki mevcut adımları incele.
Eksik olduğunu düşündüğün kısmı söyleme; öğrencinin kendi çizimine bakarak bir sonraki adımı düşünmesini sağlayan tek soru sor.
Akış şeması gerçekten tamamlandığında basamak_tamamlandi=true olabilir.
"""
    elif current_step == "4. Hata Ayıklama":
        progression = """
ÖZEL KURAL — HATA AYIKLAMA:
Öğrencinin akış şemasını ve önceki aşamaları kullan.
Önce test ettir.
Bir tutarsızlık varsa bunu söylemeden, öğrencinin hangi adımı yeniden incelemesi gerektiğini kendisinin bulmasını sağla.
Düzeltmeden sonra yeniden test ettir.
Sorun yoksa ve yaş düzeyine uygun başka bir yol varsa alternatif düşünmesini sağlayabilirsin.
Hata yokken hata varmış gibi davranma.
"""
    else:
        progression = """
ÖZEL KURAL — METABİLİŞSEL YANSITMA:

Bu aşamada matematiksel çözümü yeniden öğretme; öğrencinin kendi düşünme
sürecini incelemesini sağla.

Sıra kesinlikle:
1. PLANLAMA
2. İZLEME
3. DEĞERLENDİRME

PLANLAMA: öğrencinin çözüm öncesinde neyi amaçladığını, hangi bilgileri
kullanmayı düşündüğünü ve nasıl bir yol tasarladığını fark ettir.

İZLEME: öğrencinin çözüm sırasında düşünmesini ve yaptığı adımları nasıl
kontrol ettiğini, gerektiğinde düşüncesini nasıl değiştirdiğini fark ettir.

DEĞERLENDİRME: öğrencinin kullandığı yolun kendisi için ne kadar işe
yaradığını, çözümünü nasıl değerlendirdiğini ve benzer bir problemde
ne yapabileceğini düşündür.

Önceki aşamalardaki gerçek öğrenci cevaplarını kullan. Öğrencinin söylemediği
bir düşünceyi onun adına çıkarma. Bir boyut yeterince ele alınmadan diğer
boyuta geçme. Tek soru sor.

BASAMAK TAMAMLAMA:
Planlama, izleme ve değerlendirme boyutlarının üçü de öğrencinin kendi
ifadeleriyle ele alınmışsa basamak_tamamlandi=true olabilir.
"""

    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAK PROTOKOLÜ:
{BASAMAK_TALIMATLARI[current_step]}

ALT BECERİLER:
{ALT_BECERILER[current_step]}

{progression}

PROBLEM ANALİZİ:
{problem_analysis}

ÖNCEKİ BASAMAKLARIN DİYALOGLARI:
{previous}

MEVCUT BASAMAK DİYALOĞU:
{conversation}

ÖĞRENCİNİN SON MESAJI:
{student_message}

GÖREV:
Yüklenen gerçek problem görselini incele.
Öğrencinin son mesajını önce anlamını koruyarak değerlendir; öğrencinin söylediği bilgiyi kendi varsayımınla değiştirme.
Mevcut basamağın amacını, diğer bütün genel kurallardan ÖNCELE.
Bir sonraki düşünme adımını destekleyen yalnızca TEK bir soru sor.

V23 — PROBLEMİ GERÇEKTEN TAKİP ETME:
- Yalnızca “genel bir rehber” gibi konuşma. Öğrencinin yüklediği problem görselini bu yanıt sırasında incele; problem analizi metnini tek kaynak kabul etme. Görseldeki araç adları, noktalar, zamanlar, uzaklıklar, tablo/grafik etiketleri ve istenen niceliği birbirinden ayırt et.
- Sorunu oluşturan somut bağlamı ve öğrencinin son iki-üç mesajında söylediği belirli bilgileri kullan. Bir sonraki soruda mümkünse öğrencinin adlandırdığı niceliği/nesneyi açıkça an; “bu bilgi”, “hangi bilgi”, “neye ihtiyacın var” gibi belirsiz ifadeleri tek başına kullanma.
- Önce öğrencinin son cevabının, hemen önceki soruya gerçekten cevap verip vermediğini belirle. Cevap verdiyse kabul et ve bir sonraki düşünme halkasına geç; aynı soruyu farklı kelimelerle tekrarlama.
- Öğrenci “grafikte yazıyor zaten” derse bunu, veriyi yeniden istememek gerektiğine dair açık geri bildirim olarak yorumla. Grafikteki veriyi yeniden aratma veya genel bir bağlantı sorusuna kaçma. Görseldeki ve öğrencinin söylediği somut bilgiyi temel alarak, ayrıştırma zincirinde henüz ele alınmamış halkayı düşündüren bir soru sor.
- “Hangi bilgiye ihtiyacın var?”, “Hangi veriyi kullanacaksın?”, “Bu bilgi nasıl bağlantılı?” kalıplarını art arda veya bağlamdan kopuk kullanma. Bu ifadeler ancak problemdeki somut nicelikle birlikte anlamlıysa kullanılabilir.
- Öğrencinin verdiği bir veriyi, kendisinin daha önce söylememiş gibi yeniden sorma. Öğrencinin sözünü ettiği iki veri varsa bunları doğru biçimde yansıt; ancak aralarındaki işlemi, formülü, sonucu veya çözüm yolunu söyleme.
- Görseldeki ayrıntıyı güvenilir biçimde okuyamıyorsan ayrıntı uydurma; öğrenciden yalnızca okunamayan kısmı netleştirmesini iste.

V23 KONUŞMA TUTARLILIĞI KURALI:
- Mevcut diyalogdaki en son rehber sorusunu ve öğrencinin ona verdiği cevabı birlikte oku.
- Öğrencinin söylediği bilgiyi tekrar sorma; aynı düşünceyi farklı kelimelerle yeniden sordurma.
- "Evet/hayır" cevabını yeni bir bilgi gibi ele alma; önceki sorunun cevabı olarak kabul et.
- "Elindeki veriler yeterli mi?" sorusuna "evet" cevabı geldiyse "hangi adımı atman gerekiyor?" veya benzeri işlem/çözüm sorusu sorma.
- Öğrenci işlem önerirse bunu onaylama, genişletme veya hesaplatma. Ayrıştırma sınırında kalarak yalnızca o işlemle hangi bilgiyi belirlemeye çalıştığını sor.
- Öğrenci bir sonraki düşünme halkasını zaten söylediyse o halkayı kabul et ve tekrar sorma.
- Öğrencinin söylediği şeyden sonra bir sonraki sorunun yönü değişmeli; aynı bilgi arama kalıbını döngü halinde sürdürme.

KRİTİK AŞAMA SINIRI:
Eğer MEVCUT BASAMAK "1. Ayrıştırma" ise öğrenciyi cevaba, yönteme,
işleme, matematiksel ilişkiye veya çözüm planına götüren soru sorma.

Ayrıştırmada soru yalnızca şu düşünme yönlerinden biri olabilir:
"Ne isteniyor?"
"Bunu belirlemek için hangi bilgi gerekiyor?"
"Bu bilgiye ulaşmak için hangi bilgi gerekiyor?"
"Elindeki verilenlerden hangisi bu ihtiyacı karşılıyor?"
"Şu ana kadar belirlediklerin yeterli mi?"
"Eksik olduğunu düşündüğün bilgi nedir?"

ÖNEMLİ: Soruda öğrencinin henüz söylemediği çözüm kaynağını veya
çözüm niceliğini isimlendirme. Örneğin öğrenci B'den söz etmediyse
"B aracının bilgisi", hızdan söz etmediyse "hız", süreden söz etmediyse
"süre" kelimelerini çözüm yönlendirmesi olarak kullanma.
Öğrencinin çözüm için gereken bir bilgiyi zaten söylemesi, o bilginin nasıl kullanılacağını sorma hakkı vermez; o bilgi zincirde bir halka olarak kabul edilir.

ÖZELLİKLE AYRIŞTIRMA AŞAMASINDA:
- Öğrencinin son mesajında açıkça verdiği bilgi, nicelik, değer veya grafik bilgisini mevcut zincirin bir halkası olarak kabul et.
- Öğrencinin zaten söylediği bilgiyi tekrar isteme; aynı bilgiyi "başka hangi bilgi...", "hangi değer...", "grafikte ne görüyorsun?" gibi sorularla yeniden aratma.
- Öğrenci gerekli bir grafik bilgisini zaten ifade ettiyse, o bilgiyi tekrar buldurma. Bunun yerine o bilginin istenene ulaşmak için yeterli olup olmadığını veya bundan geriye doğru hangi bilgiye ihtiyaç olduğunu düşündür.
- Öğrenci bir bilginin eksik olduğunu söylüyorsa bunu doğrudan görsel ipucuyla tamamlamaya çalışma. Eksik olan bilginin ne olduğunu veya mevcut bilgilerin yeterli olup olmadığını öğrencinin kendisinin değerlendirmesini sağla.
- Öğrencinin zaten fark ettiği bir eksen, nokta, çizgi, renk, kesişim veya grafik özelliğini yeniden tarif etme.
- Problem analizinde bulunan bir bilgi ile öğrencinin söylediği bilgi aynıysa, öğrencinin cevabını yeni bir bilgi arama gerekçesi olarak kullanma; bir sonraki düşünme halkasına geç.

Öğrencinin söylemediği sonucu onun adına çıkarma.
Çözüm, formül, denklem, işlem, cevap veya doğrudan ipucu verme.

YANITI YALNIZCA ŞU JSON YAPISINDA VER:
{{
  "alt_beceri": "...",
  "basamak_tamamlandi": false,
  "soru": "..."
}}

Basamak tamamlanmadıysa false kullan.
Basamak tamamlandıysa true kullan; bunu yalnızca konuşmada gerçekten yeterli kanıt varsa yap.
Soru alanında yalnızca öğrencinin göreceği 1-2 kısa cümle ve TEK soru olsun.
"""

    raw = gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=180,
        include_problem_image=True,
        include_flowchart_image=(current_step in ["3. Algoritma Tasarımı", "4. Hata Ayıklama", "5. Metabilişsel Yansıtma"])
    )

    result = parse_guidance_response(raw, current_step)

    # v23: Ayrıştırma yanıtını genel, sabit kalıplarla sonradan değiştirme.
    # Öğrencinin problem görseline ve tüm konuşmaya dayalı özgün yanıtı korunur.
    return result


# ============================================================
# FİNAL SÜREÇ ÖZETİ
# ============================================================

def final_ozet_olustur(
    student_id,
    chat_storage
):

    process = ""


    for stage, messages in chat_storage.items():

        process += (
            f"\n\n===== {stage} =====\n"
        )

        for message in messages:

            process += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )


    prompt = f"""

Sen bir ortaokul matematik öğretmenisin.

Konu:
DOĞRUSAL İLİŞKİLER

Aşağıdaki öğrencinin algoritmik düşünme
sürecini kısa biçimde değerlendir.

Problemi yeniden çözme.

Cevabı söyleme.

Yeni çözüm yolu verme.

Öğrencinin yerine matematiksel ilişki kurma.

Şu dört boyutu değerlendir:

1. Ayrıştırma
2. Soyutlama
3. Algoritma Tasarımı
4. Hata Ayıklama

Her boyut için öğrencinin süreçteki
yaklaşımını kısaca belirt.

Özellikle:

- Problemi parçalara ayırabilme,
- Nicelikleri ve ilişkileri fark edebilme,
- Değişimi ve örüntüyü inceleyebilme,
- Farklı gösterimler arasında ilişki kurabilme,
- Kendi çözüm planını oluşturabilme,
- Oluşturduğu ilişkiyi kontrol edebilme

açısından değerlendir.

Sonunda genel bir süreç değerlendirmesi yap.

Öğrenci:

{student_id}


SÜREÇ:

{process}

"""


    return gemini_generate(

        [
            {
                "text":
                    prompt
            }
        ],

        temperature=0.3,

        max_tokens=500,

        include_problem_image=True
    )


# ============================================================
# DAYANIKLI SOHBET KONTROL NOKTASI
# ============================================================

CHAT_BACKUP_FILE = "tez_sohbet_yedekleri.json"

def _load_chat_backups():
    try:
        if not os.path.isfile(CHAT_BACKUP_FILE):
            return {}
        with open(CHAT_BACKUP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_chat_backup(student_id):
    if not student_id:
        return
    try:
        backups = _load_chat_backups()
        backups[str(student_id)] = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_step": st.session_state.get("current_step", "1. Ayrıştırma"),
            "chat_storage": st.session_state.get("chat_storage", {}),
            "completed_stages": st.session_state.get("completed_stages", []),
            "last_alt_beceri": st.session_state.get("last_alt_beceri", {}),
            "problem_analysis": st.session_state.get("problem_analysis"),
            "problem_session_id": st.session_state.get("problem_session_id"),
        }
        with open(CHAT_BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(backups, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _restore_chat_backup(student_id):
    if not student_id:
        return False
    try:
        backups = _load_chat_backups()
        backup = backups.get(str(student_id))
        if not isinstance(backup, dict):
            return False
        saved_chat = backup.get("chat_storage")
        if not isinstance(saved_chat, dict):
            return False
        normalized_chat = {stage: saved_chat.get(stage, []) for stage in BASAMAK_TALIMATLARI}
        if not any(normalized_chat.values()):
            return False
        st.session_state.chat_storage = normalized_chat
        saved_step = backup.get("current_step", "1. Ayrıştırma")
        st.session_state.current_step = saved_step if saved_step in BASAMAK_TALIMATLARI else "1. Ayrıştırma"
        st.session_state.completed_stages = [s for s in backup.get("completed_stages", []) if s in BASAMAK_TALIMATLARI]
        st.session_state.last_alt_beceri = backup.get("last_alt_beceri", {}) if isinstance(backup.get("last_alt_beceri", {}), dict) else {}
        if backup.get("problem_analysis"):
            st.session_state.problem_analysis = backup["problem_analysis"]
        st.session_state.problem_session_id = backup.get("problem_session_id")
        return True
    except Exception:
        return False


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:

    st.session_state.uploaded_file_data = None


if "problem_session_id" not in st.session_state:

    st.session_state.problem_session_id = None


if "backup_restored_for_student" not in st.session_state:

    st.session_state.backup_restored_for_student = None


if "problem_analysis" not in st.session_state:

    st.session_state.problem_analysis = None


if "chat_storage" not in st.session_state:

    st.session_state.chat_storage = {

        stage: []

        for stage
        in BASAMAK_TALIMATLARI
    }


if "current_step" not in st.session_state:

    st.session_state.current_step = (
        "1. Ayrıştırma"
    )


if "annotation_reset" not in st.session_state:

    st.session_state.annotation_reset = 0


if "annotation_json" not in st.session_state:

    st.session_state.annotation_json = None


if "annotation_image_data" not in st.session_state:

    st.session_state.annotation_image_data = None


if "flowchart_storage" not in st.session_state:

    st.session_state.flowchart_storage = {}


if "flowchart_image_data" not in st.session_state:

    st.session_state.flowchart_image_data = None


if "last_alt_beceri" not in st.session_state:

    st.session_state.last_alt_beceri = {}


if "completed_stages" not in st.session_state:

    st.session_state.completed_stages = []


# ============================================================
# SOL MENÜ
# ============================================================

with st.sidebar:

    st.title(
        "👨‍🏫 Araştırma Paneli"
    )


    mode = st.selectbox(

        "Giriş Türü:",

        [
            "Öğrenci Girişi",
            "Öğretmen (Admin)"
        ]
    )


    # --------------------------------------------------------
    # ADMIN
    # --------------------------------------------------------

    if mode == "Öğretmen (Admin)":

        sifre = st.text_input(

            "Şifre:",

            type="password"
        )


        if sifre == "tez2024":

            st.success(
                "Admin Paneli Aktif"
            )


            if os.path.isfile(
                DATA_FILE
            ):

                try:

                    df_csv = pd.read_csv(

                        DATA_FILE,

                        sep=None,

                        engine="python",

                        on_bad_lines="skip"
                    )


                    st.write(
                        "### 📊 Veri Kayıtları"
                    )


                    st.dataframe(

                        df_csv.tail(30),

                        use_container_width=True
                    )


                    csv_data = (

                        df_csv

                        .to_csv(
                            index=False
                        )

                        .encode(
                            "utf-8-sig"
                        )
                    )


                    st.download_button(

                        "📥 Tüm Verileri İndir",

                        csv_data,

                        "tez_data.csv",

                        "text/csv"
                    )


                except Exception as e:

                    st.error(
                        f"Dosya hatası: {e}"
                    )


            else:

                st.info(
                    "Henüz veri kaydı yok."
                )


        st.stop()


    # --------------------------------------------------------
    # ÖĞRENCİ
    # --------------------------------------------------------

    student_id = st.text_input(

        "Öğrenci No:",

        placeholder="Örn: Hakan"
    )


    if not student_id:

        st.warning(
            "Devam etmek için öğrenci numaranızı girin."
        )

        st.stop()


    if st.session_state.backup_restored_for_student != student_id:
        _restore_chat_backup(student_id)
        st.session_state.backup_restored_for_student = student_id


    st.divider()


    # --------------------------------------------------------
    # AKIŞ ŞEMASI ARAÇLARI
    # --------------------------------------------------------

    st.write(
        "🖌️ **Akış Şeması Araçları**"
    )


    tool_map = {

        "Dikdörtgen (İşlem)": "rect",

        "Elips (Başla/Bitir)": "circle",

        "Ok/Çizgi": "line",

        "Serbest Çizim": "freedraw",

        "Düzenle/Taşı": "transform",

        "Çokgen": "polygon"
    }


    selected_tool = st.selectbox(

        "Araç Seçin:",

        list(tool_map.keys())
    )


    drawing_mode = tool_map[
        selected_tool
    ]


    stroke_color = st.color_picker(

        "Çizgi Rengi:",

        "#000000"
    )


    fill_color = st.color_picker(

        "Kutu Rengi:",

        "#EEEEEE"
    )


    st.divider()


    # --------------------------------------------------------
    # BASAMAK SEÇİMİ
    # --------------------------------------------------------

    steps = list(
        BASAMAK_TALIMATLARI.keys()
    )


    current_index = steps.index(

        st.session_state.current_step
    )


    selected_step = st.radio(

        "Aşamayı Seçin:",

        steps,

        index=current_index
    )


    if (
        selected_step
        !=
        st.session_state.current_step
    ):

        st.session_state.current_step = (
            selected_step
        )

        st.rerun()


# ============================================================
# ANA BAŞLIK
# ============================================================

st.title(
    "🎯 Algoritmik Problem Çözme Rehberi · v22"
)


st.write(

    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)

if st.session_state.current_step == "5. Metabilişsel Yansıtma":
    st.info(
        "🧠 Bu aşama üç boyutta ilerler: **Planlama → İzleme → Değerlendirme**. "
        "Amaç çözümü yeniden yapmak değil, kendi düşünme sürecini fark etmek ve değerlendirmektir."
    )


# ============================================================
# PROBLEM GÖRSELİNİ ÇİZİM ALANINA HAZIRLAMA
# ============================================================

def get_annotation_background():
    """Yüklenen problem görselini öğrenci işaretleme alanına hazırlar."""
    uploaded_file = st.session_state.get("uploaded_file_data")

    if uploaded_file is None:
        return None, None, None

    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        # Problem görselini ekranda yaklaşık 900 px genişliğe
        # sığdırırken en-boy oranını koru.
        max_width = 900
        original_width, original_height = image.size

        display_width = min(
            original_width,
            max_width
        )

        display_height = max(
            1,
            int(
                original_height
                * display_width
                / original_width
            )
        )

        image = image.resize(
            (display_width, display_height),
            Image.LANCZOS
        )

        return image, display_width, display_height

    except Exception:
        return None, None, None


# ============================================================
# PROBLEM YÜKLEME
# ============================================================

if (
    st.session_state.uploaded_file_data
    is None
):

    uploaded = st.file_uploader(

        "📷 Soru Fotoğrafı Yükle",

        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )


    if uploaded:

        st.session_state.uploaded_file_data = (
            uploaded
        )


        with st.spinner(
            "Problem analiz ediliyor..."
        ):

            try:

                st.session_state.problem_analysis = (

                    analyze_problem_image(
                        uploaded
                    )
                )


            except Exception as e:

                st.error(
                    "Problem analiz edilemedi."
                )

                st.code(
                    str(e)
                )

                st.session_state.problem_analysis = ""


        st.session_state.problem_session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        st.session_state.chat_storage = {
            stage: []
            for stage in BASAMAK_TALIMATLARI
        }
        st.session_state.current_step = "1. Ayrıştırma"
        st.session_state.annotation_reset += 1
        st.session_state.annotation_json = None
        st.session_state.annotation_image_data = None
        st.session_state.flowchart_storage = {}
        st.session_state.flowchart_image_data = None
        st.session_state.completed_stages = []
        st.session_state.last_alt_beceri = {}

        st.rerun()


# ============================================================
# YÜKLENMİŞ PROBLEM
# ============================================================

else:

    if st.session_state.problem_session_id is None:
        st.session_state.problem_session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    st.image(

        st.session_state.uploaded_file_data,

        width=900
    )


    # --------------------------------------------------------
    # PROBLEM ÜZERİNDE ÖĞRENCİ İŞAREMLEME ALANI
    # --------------------------------------------------------
    annotation_image, annotation_width, annotation_height = (
        get_annotation_background()
    )

    if annotation_image is not None:
        st.write("🖍️ **Problem Üzerinde İşaretleme**")
        st.caption(
            "Problem görselinin üzerinde işaretleme yapabilirsin. "
            "Bu alan yalnızca senin çizimlerin içindir."
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
            drawing_mode=annotation_tool_map[annotation_tool],
            update_streamlit=True,
            initial_drawing=st.session_state.annotation_json,
            key=(
                "problem_annotation_"
                + str(st.session_state.annotation_reset)
            )
        )

        # Çizimi her rerun'da session state'te tut. Böylece sohbet
        # gönderildiğinde problem üzerindeki işaretlemeler kaybolmaz.
        try:
            if annotation_result is not None and annotation_result.json_data:
                st.session_state.annotation_json = annotation_result.json_data
                current_annotation_image = safe_get_canvas_image_data(annotation_result)
                if current_annotation_image is not None:
                    st.session_state.annotation_image_data = current_annotation_image
        except Exception:
            pass

        if st.button(
            "💾 Problem Üzerindeki İşaretlemeyi Kaydet",
            key="save_problem_annotation"
        ):
            if annotation_result is not None and annotation_result.json_data:
                log_kaydet({
                    "tarih":
                        datetime.now()
                        .strftime("%Y-%m-%d %H:%M:%S"),
                    "id": student_id,
                    "basamak": st.session_state.current_step,
                    "tip": "Problem Üzeri Çizim",
                    "icerik": str(annotation_result.json_data)
                })

                st.success(
                    "Problem üzerindeki işaretleme kaydedildi."
                )
            else:
                st.info(
                    "Önce problem üzerinde bir işaretleme yap."
                )

        if st.button(
            "🧹 İşaretlemeyi Temizle",
            key="clear_problem_annotation"
        ):
            st.session_state.annotation_reset += 1
            st.session_state.annotation_json = None
            st.session_state.annotation_image_data = None
            st.rerun()


    if st.button(
        "❌ Soruyu Değiştir"
    ):

        st.session_state.uploaded_file_data = (
            None
        )

        st.session_state.problem_analysis = (
            None
        )

        st.session_state.problem_session_id = None

        try:
            backups = _load_chat_backups()
            backups.pop(str(student_id), None)
            with open(CHAT_BACKUP_FILE, "w", encoding="utf-8") as f:
                json.dump(backups, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


        st.session_state.chat_storage = {

            stage: []

            for stage
            in BASAMAK_TALIMATLARI
        }

        st.session_state.annotation_reset += 1
        st.session_state.annotation_json = None
        st.session_state.annotation_image_data = None
        st.session_state.flowchart_storage = {}
        st.session_state.flowchart_image_data = None
        st.session_state.completed_stages = []
        st.session_state.last_alt_beceri = {}


        st.rerun()


# ============================================================
# PROBLEM ANALİZİ
# ============================================================

if st.session_state.problem_analysis:

    with st.expander(
        "🔎 Problem analizini göster"
    ):

        st.write(
            st.session_state.problem_analysis
        )


st.divider()


# ============================================================
# İKİ SÜTUN
# ============================================================

col1, col2 = st.columns(

    [1.3, 1],

    gap="large"
)


# ============================================================
# SOL SÜTUN
# ============================================================

with col1:

    st.write(
        "🖼️ **Tasarım ve Planlama Alanı**"
    )


    st.caption(
        "Akış şemanı burada oluşturabilirsin."
    )


    canvas_result = st_canvas(

        fill_color=fill_color,

        stroke_color=stroke_color,

        stroke_width=3,

        background_color="#ffffff",

        height=450,

        drawing_mode=drawing_mode,

        update_streamlit=True,

        initial_drawing=st.session_state.flowchart_storage.get(
            st.session_state.current_step
        ),

        key=(
            "canvas_"
            +
            st.session_state.current_step
            .replace(" ", "_")
        )
    )

    try:
        if canvas_result is not None and canvas_result.json_data:
            st.session_state.flowchart_storage[st.session_state.current_step] = canvas_result.json_data
            current_flowchart_image = safe_get_canvas_image_data(canvas_result)
            if current_flowchart_image is not None:
                st.session_state.flowchart_image_data = current_flowchart_image
    except Exception:
        pass


    # --------------------------------------------------------
    # TASARIMI KAYDET
    # --------------------------------------------------------

    if st.button(
        "🖼️ Tasarımı Kaydet"
    ):

        if canvas_result.json_data:

            log_kaydet({

                "tarih":
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "id":
                    student_id,

                "basamak":
                    st.session_state.current_step,

                "tip":
                    "Cizim",

                "icerik":
                    str(
                        canvas_result.json_data
                    )
            })


            st.success(
                "Tasarım kaydedildi!"
            )


    st.write("---")


    # --------------------------------------------------------
    # ÖZ-YANSITMA
    # --------------------------------------------------------

    st.info(

        "🧠 **Öz-Yansıtma:** "
        +
        METABILISSEL_SORULAR[
            st.session_state.current_step
        ]
    )


    meta_key = (

        "meta_"
        +
        st.session_state.current_step
        .replace(" ", "_")
    )


    m_cevap = st.text_area(

        "Düşünceni buraya yaz...",

        key=meta_key
    )


    if st.button(
        "💾 Düşüncemi Kaydet"
    ):

        log_kaydet({

            "tarih":
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "id":
                student_id,

            "basamak":
                st.session_state.current_step,

            "tip":
                "Metabiliş",

            "icerik":
                m_cevap
        })


        st.success(
            "Kaydedildi!"
        )


    st.write("---")


    # --------------------------------------------------------
    # EMİNLİK
    # --------------------------------------------------------

    st.write(

        "⭐ **Bu adımdaki çözümünden "
        "ne kadar eminsin?**"
    )


    confidence = st.select_slider(

        "Derecelendir:",

        options=[

            "Hiç Emin Değilim",

            "Kararsızım",

            "Biraz Eminim",

            "Çok Eminim"
        ],

        value="Kararsızım",

        key=(

            "confidence_"
            +
            st.session_state.current_step
        )
    )


    if st.button(
        "📈 Eminlik Derecesini Kaydet"
    ):

        log_kaydet({

            "tarih":
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "id":
                student_id,

            "basamak":
                st.session_state.current_step,

            "tip":
                "Eminlik",

            "icerik":
                confidence
        })


        st.success(
            f"Eminlik: {confidence}"
        )


    st.write("---")


    # --------------------------------------------------------
    # FİNAL
    # --------------------------------------------------------

    if st.button(
        "🏁 Çözümü Bitir ve Özetini Al"
    ):

        with st.spinner(
            "Süreç analiz ediliyor..."
        ):

            try:

                final_text = (
                    final_ozet_olustur(

                        student_id,

                        st.session_state.chat_storage
                    )
                )


                st.success(
                    "Süreç değerlendirmesi hazır."
                )


                st.write(
                    final_text
                )


                log_kaydet({

                    "tarih":
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "id":
                        student_id,

                    "basamak":
                        "FİNAL",

                    "tip":
                        "Final Özeti",

                    "icerik":
                        final_text
                })


            except Exception as e:

                st.error(
                    "Özet hazırlanamadı."
                )

                st.code(
                    str(e)
                )


# ============================================================
# SAĞ SÜTUN - REHBER BOT
# ============================================================

with col2:

    st.write(
        "💬 **Rehber Bot**"
    )


    current_stage = (
        st.session_state.current_step
    )


    # --------------------------------------------------------
    # İLK SORU
    # --------------------------------------------------------

    if (

        st.session_state.problem_analysis

        and

        len(
            st.session_state
            .chat_storage[
                current_stage
            ]
        ) == 0

    ):

        with st.spinner(
            "Rehber hazırlanıyor..."
        ):

            try:

                first_result = ilk_soruyu_olustur(
                    current_stage,
                    st.session_state.problem_analysis
                )

                first_question = first_result["soru"]
                first_skill = first_result["alt_beceri"]
                st.session_state.last_alt_beceri[current_stage] = first_skill

                st.session_state.chat_storage[current_stage].append({
                    "role": "assistant",
                    "content": first_question,
                    "alt_beceri": first_skill,
                    "basamak_tamamlandi": False
                })

                log_kaydet({
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": student_id,
                    "problem_session_id": st.session_state.get("problem_session_id"),
                    "basamak": current_stage,
                    "alt_beceri": first_skill,
                    "basamak_tamamlandi": False,
                    "tip": "Bot",
                    "icerik": first_question
                })

                _save_chat_backup(student_id)
                st.rerun()


            except Exception as e:

                st.error(
                    "İlk soru oluşturulamadı."
                )

                st.code(
                    str(e)
                )


    # --------------------------------------------------------
    # SOHBET
    # --------------------------------------------------------

    if current_stage in st.session_state.completed_stages:
        if current_stage == "5. Metabilişsel Yansıtma":
            st.success("🎯 Beş aşamalı süreç tamamlandı. Artık genel süreç değerlendirmesine geçebilirsin.")
        else:
            st.success("Bu basamak tamamlandı. Bir sonraki basamağa geçildi.")

    chat_container = st.container(
        height=550
    )


    for message in (

        st.session_state
        .chat_storage[
            current_stage
        ]

    ):

        with chat_container.chat_message(

            message["role"]

        ):

            st.write(
                message["content"]
            )


    # --------------------------------------------------------
    # ÖĞRENCİ MESAJI
    # --------------------------------------------------------

    student_message = st.chat_input(

        "Düşünceni veya cevabını yaz..."
    )


    if student_message:

        # ----------------------------------------------------
        # ÖĞRENCİ MESAJINI KAYDET
        # ----------------------------------------------------

        st.session_state \
            .chat_storage[
                current_stage
            ].append({

                "role":
                    "user",

                "content":
                    student_message
            })


        log_kaydet({

            "tarih":
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "id":
                student_id,

            "problem_session_id":
                st.session_state.get("problem_session_id"),

            "basamak":
                current_stage,

            "alt_beceri":
                st.session_state.last_alt_beceri.get(current_stage, ""),

            "basamak_tamamlandi":
                False,

            "tip":
                "Öğrenci",

            "icerik":
                student_message
        })

        # Gemini çağrısından ÖNCE öğrenci mesajını kalıcı kontrol
        # noktasına yaz.
        _save_chat_backup(student_id)


        # ----------------------------------------------------
        # GEMINI YANITI
        # ----------------------------------------------------

        with st.spinner(
            "Rehber Bot düşünüyor..."
        ):

            try:

                result = ogrenciye_cevap_ver(
                    current_stage,
                    st.session_state.problem_analysis,
                    st.session_state.chat_storage[current_stage][:-1],
                    student_message
                )

                answer = result["soru"]
                alt_beceri = result["alt_beceri"]
                stage_completed = result["basamak_tamamlandi"]

                st.session_state.last_alt_beceri[current_stage] = alt_beceri

                st.session_state.chat_storage[current_stage].append({
                    "role": "assistant",
                    "content": answer,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed
                })

                log_kaydet({
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": student_id,
                    "problem_session_id": st.session_state.get("problem_session_id"),
                    "basamak": current_stage,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed,
                    "tip": "Bot",
                    "icerik": answer
                })

                student_turns = sum(
                    1 for m in st.session_state.chat_storage[current_stage]
                    if m["role"] == "user"
                )

                stages = list(BASAMAK_TALIMATLARI.keys())
                stage_index = stages.index(current_stage)

                _save_chat_backup(student_id)

                if stage_completed and student_turns >= 2:
                    if current_stage not in st.session_state.completed_stages:
                        st.session_state.completed_stages.append(current_stage)

                    if stage_index < len(stages) - 1:
                        next_stage = stages[stage_index + 1]
                        st.session_state.current_step = next_stage

                        log_kaydet({
                            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "id": student_id,
                            "basamak": current_stage,
                            "alt_beceri": alt_beceri,
                            "basamak_tamamlandi": True,
                            "tip": "Basamak Geçişi",
                            "icerik": f"{current_stage} tamamlandı → {next_stage}"
                        })

                st.rerun()


            except Exception as e:

                st.error(
                    "Gemini yanıt oluşturamadı."
                )

                st.code(
                    str(e)
                )
