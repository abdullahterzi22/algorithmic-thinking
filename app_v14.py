import streamlit as st
import requests
import pandas as pd
import os
import base64
import json
from datetime import datetime
from PIL import Image


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Algoritmik Düşünme Atölyesi",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# AYARLAR
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL_NAME}:generateContent"
)

DATA_FILE = "tez_verileri_final.csv"

ADMIN_PASSWORD = "tez2024"


# ============================================================
# GEMINI API ANAHTARI
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY bulunamadı. "
        "Streamlit Cloud → Settings → Secrets bölümüne "
        "GEMINI_API_KEY ekleyin."
    )
    st.stop()


# ============================================================
# ALGORİTMİK DÜŞÜNME BASAMAKLARI
# ============================================================

BASAMAKLAR = [
    "Ayrıştırma",
    "Soyutlama",
    "Algoritma Tasarımı",
    "Hata Ayıklama",
    "Metabilişsel Yansıtma"
]


# ============================================================
# ALT BECERİLER
# ============================================================

ALT_BECERILER = {

    "Ayrıştırma": {
        "A1": "Alt amacı belirleme",
        "A2": "Problemi anlamlı parçalara ayırma",
        "A3": "Parçalar arasındaki ilişkileri belirleme"
    },

    "Soyutlama": {
        "S1": "Problem için ilgili bilgileri belirleme",
        "S2": "Gereksiz veya ikincil bilgileri ayırt etme",
        "S3": "Temel ilişkileri belirleme",
        "S4": "Problemi uygun bir temsil ile ifade etme",
        "S5": "Örüntü veya genel yapı fark etme"
    },

    "Algoritma Tasarımı": {
        "AT1": "Yapılacak işlemleri belirleme",
        "AT2": "İşlemleri uygun sıraya koyma",
        "AT3": "Ara sonuçları ve kontrol noktalarını belirleme",
        "AT4": "Adım adım prosedür oluşturma",
        "AT5": "Oluşturulan yöntemin benzer problemlerde kullanılabilirliğini düşünme"
    },

    "Hata Ayıklama": {
        "HA1": "Çözümü veya yöntemi test etme",
        "HA2": "Sonuçları problem koşullarıyla karşılaştırma",
        "HA3": "Tutarsızlıkları fark etme",
        "HA4": "Sorunun oluştuğu adımı belirleme",
        "HA5": "Gerekli düzeltmeyi düşünme",
        "HA6": "Düzeltme sonrasında yeniden test etme"
    }
}


# ============================================================
# BASAMAK AÇIKLAMALARI
# ============================================================

BASAMAK_ACIKLAMALARI = {

    "Ayrıştırma": """
Bu basamakta öğrencinin problemi bir bütün olarak incelemek yerine
verilenleri, isteneni, alt amaçları ve problemdeki ilişkileri fark etmesi
beklenir.
""",

    "Soyutlama": """
Bu basamakta öğrenci problemin çözümünde gerekli olan bilgileri
gereksiz bilgilerden ayırır, daha önce öğrendiği bilgilerle ilişki kurar,
temel yapıyı veya örüntüyü fark eder ve problemi uygun biçimde temsil eder.
""",

    "Algoritma Tasarımı": """
Bu basamakta öğrenci çözüm için kendi işlem ve adım sırasını oluşturur.
AI öğrencinin yerine işlem sırası veya yöntem söylemez.
""",

    "Hata Ayıklama": """
Bu basamakta öğrenci kendi oluşturduğu yöntemi veya çözümünü
koşullarla karşılaştırır, tutarsızlıkları fark eder ve gerekli düzeltmeyi
kendisinin bulması sağlanır.
""",

    "Metabilişsel Yansıtma": """
Öğrenci kullandığı düşünme sürecini, hangi noktada zorlandığını,
hangi stratejinin işe yaradığını ve bundan sonraki problemlerde ne
yapabileceğini değerlendirir.
"""
}


# ============================================================
# METABİLİŞSEL SORULAR
# ============================================================

METABILISSEL_SORULAR = [
    "Bu problem üzerinde düşünürken en çok hangi noktada zorlandın?",
    "Problemi çözme sürecinde hangi düşünme adımının sana yardımcı olduğunu düşünüyorsun?",
    "Başka benzer bir problemle karşılaştığında bu deneyiminden neyi kullanabilirsin?",
    "Çözüm sürecine yeniden başlasaydın hangi noktayı farklı ele alırdın?"
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "student_id": "",
    "problem_image_bytes": None,
    "problem_image_name": "",
    "problem_image_mime": "",
    "problem_analysis": "",
    "current_stage": 0,
    "chat_history": {},
    "stage_completed": {},
    "selected_alt_beceri": {},
    "stage_first_question_done": {},
    "last_ai_question": "",
    "metacognitive_answer": "",
    "confidence": 3,
    "admin_mode": False,
    "problem_uploaded": False
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSV KAYIT
# ============================================================

def csv_kaydet(kayit):

    try:

        df_new = pd.DataFrame([kayit])

        if os.path.exists(DATA_FILE):

            df_old = pd.read_csv(
                DATA_FILE,
                encoding="utf-8-sig"
            )

            df = pd.concat(
                [df_old, df_new],
                ignore_index=True
            )

        else:

            df = df_new

        df.to_csv(
            DATA_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as e:
        st.warning(f"Veri kaydı sırasında sorun oluştu: {e}")


# ============================================================
# GÖRSELİ BASE64'E ÇEVİRME
# ============================================================

def image_to_base64():

    if not st.session_state.problem_image_bytes:
        return None

    return base64.b64encode(
        st.session_state.problem_image_bytes
    ).decode("utf-8")


# ============================================================
# GEMINI ÇAĞRISI
# ============================================================

def gemini_request(
    prompt,
    include_problem=True,
    temperature=0.2
):

    headers = {
        "Content-Type": "application/json"
    }

    parts = []

    # --------------------------------------------------------
    # PROBLEM GÖRSELİ
    # --------------------------------------------------------

    if (
        include_problem
        and st.session_state.problem_image_bytes
    ):

        image_b64 = image_to_base64()

        parts.append({
            "inline_data": {
                "mime_type": st.session_state.problem_image_mime,
                "data": image_b64
            }
        })

    # --------------------------------------------------------
    # METİN
    # --------------------------------------------------------

    parts.append({
        "text": prompt
    })

    payload = {

        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],

        "generationConfig": {
            "temperature": temperature,
            "topP": 0.8,
            "maxOutputTokens": 180
        }
    }

    try:

        response = requests.post(
            GEMINI_URL,
            headers=headers,
            params={
                "key": GEMINI_API_KEY
            },
            json=payload,
            timeout=90
        )

    except requests.exceptions.Timeout:

        return None, "Gemini bağlantısı zaman aşımına uğradı."

    except requests.exceptions.RequestException as e:

        return None, f"Gemini bağlantı hatası: {e}"

    # --------------------------------------------------------
    # HTTP HATALARI
    # --------------------------------------------------------

    if response.status_code != 200:

        try:
            error_data = response.json()
        except Exception:
            error_data = response.text

        return None, (
            f"Gemini API Hatası ({response.status_code}): "
            f"{error_data}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception:

        return None, "Gemini geçerli bir JSON yanıtı döndürmedi."

    # --------------------------------------------------------
    # CEVABI AL
    # --------------------------------------------------------

    try:

        candidates = data.get("candidates", [])

        if not candidates:
            return None, "Gemini yanıt oluşturamadı."

        content = candidates[0].get("content", {})
        response_parts = content.get("parts", [])

        text_parts = []

        for part in response_parts:

            if "text" in part:
                text_parts.append(part["text"])

        result = "\n".join(text_parts).strip()

        if not result:
            return None, "Gemini boş yanıt döndürdü."

        return result, None

    except Exception as e:

        return None, f"Gemini yanıtı okunamadı: {e}"


# ============================================================
# GÜVENLİ YEDEK SORU
# ============================================================

def guvenli_yedek_soru(stage):

    sorular = {

        "Ayrıştırma":
            "Problemde verilen bilgilerden hangilerini ayrı ayrı ele alabileceğini düşünüyorsun?",

        "Soyutlama":
            "Bu problemde çözümünü oluştururken hangi bilgilerin gerçekten gerekli olduğunu nasıl belirleyebilirsin?",

        "Algoritma Tasarımı":
            "Belirlediğin bilgileri kullanarak ilk olarak hangi işlemi yapacağını nasıl belirleyebilirsin?",

        "Hata Ayıklama":
            "Oluşturduğun çözümün problemdeki koşulları karşıladığını nasıl kontrol edebilirsin?",

        "Metabilişsel Yansıtma":
            "Bu problemi çözerken kullandığın düşünme sürecini nasıl değerlendiriyorsun?"
    }

    return sorular.get(
        stage,
        "Bir sonraki adımını nasıl belirleyebilirsin?"
    )


# ============================================================
# PROMPT OLUŞTURMA
# ============================================================

def pedagojik_prompt(
    stage,
    student_message="",
    first_question=False
):

    alt_beceriler = ALT_BECERILER.get(stage, {})

    alt_beceri_text = "\n".join(
        [
            f"- {k}: {v}"
            for k, v in alt_beceriler.items()
        ]
    )

    history = st.session_state.chat_history.get(
        stage,
        []
    )

    # Son konuşmaları al
    recent_history = history[-8:]

    history_text = ""

    if recent_history:

        for item in recent_history:

            role = item.get("role", "")
            text = item.get("text", "")

            if role == "student":
                history_text += f"ÖĞRENCİ: {text}\n"

            elif role == "assistant":
                history_text += f"AI: {text}\n"

    if first_question:

        student_context = (
            "Öğrenci henüz bu basamakta cevap vermedi."
        )

    else:

        student_context = (
            f"ÖĞRENCİNİN SON MESAJI:\n{student_message}"
        )

    prompt = f"""
SENİN ROLÜN

Sen ortaokul matematik öğrencisinin algoritmik düşünme
sürecine rehberlik eden pedagojik bir yapay zekâsın.

Senin görevin problemi çözmek DEĞİL,
öğrencinin kendi düşünmesini sağlamaktır.

ÇOK ÖNEMLİ:

Problem görseli sana ayrıca gönderilmiştir.

Önce problem görselini dikkatlice incele.

Soracağın soru mutlaka bu görseldeki GERÇEK problemle
ilişkili olmalıdır.

Problemi görmeden veya problemdeki bilgileri dikkate almadan
genel bir matematik sorusu üretme.

MEVCUT BASAMAK:

{stage}

BASAMAĞIN AMACI:

{BASAMAK_ACIKLAMALARI.get(stage, "")}

BU BASAMAKTAKİ ALT BECERİLER:

{alt_beceri_text}

ÖNCEKİ KONUŞMA:

{history_text}

{student_context}

GÖREVİN:

Öğrencinin mevcut düşünme durumunu dikkate al.

Yalnızca mevcut algoritmik düşünme basamağına hizmet eden
EN UYGUN TEK BİR ALT BECERİ seç.

Ardından öğrencinin bir sonraki düşünme adımını kendisinin
bulmasını sağlayacak TEK BİR YÖNLENDİRİCİ SORU sor.

KESİN KURALLAR:

1. SADECE BİR SORU SOR.

2. Yanıtın en fazla 1-3 kısa cümle olsun.

3. Yanıt mutlaka bir soru işareti (?) ile bitsin.

4. Problemin cevabını söyleme.

5. İşlem yapma.

6. Hesaplama yapma.

7. Formül verme.

8. EBOB, EKOK veya başka bir yöntemi öğrenci adına seçme.

9. Çözüm adımlarını öğrencinin yerine oluşturma.

10. Hazır alt problem verme.

11. Öğrencinin yapması gereken işlemi doğrudan söyleme.

12. "Yanlış yaptın", "hata yaptın", "doğru cevap",
"yanlış", "doğru çözüm" gibi ifadeler kullanma.

13. Genel ve içi boş sorular sorma.

14. Daha önce sorulmuş bir soruyu tekrar sorma.

15. Öğrencinin verdiği bilgiyi tekrar edip ardından
genel bir soru sorma.

16. Öğrencinin cevabındaki belirli bir düşünceyi kullanarak
bir sonraki düşünme adımına yönlendir.

17. Problem görselindeki sayılar, koşullar, tablo, grafik,
şekil veya ilişkiler gerekiyorsa bunları görselden dikkate al.

18. Öğrencinin yerine problemi yorumlama.

19. Öğrenciye cevabı düşündürecek ipucu verebilirsin fakat
cevabın kendisini veremezsin.

20. Övgü cümlelerini gereksiz yere kullanma.
"Harika", "mükemmel", "çok güzel" gibi kalıp ifadeleri
kullanma.

21. Öğrenci problemdeki bir bilgiyi yanlış okuyorsa,
doğrudan düzeltmek yerine ilgili bilgiyi yeniden incelemesini
sağlayan bir soru sor.

22. Her seferinde yalnızca BİR düşünme adımına odaklan.

23. Öğrencinin cevabı yeterliyse aynı noktayı tekrar sorma;
bir sonraki uygun alt beceriye geç.

24. Soru ortaokul öğrencisinin anlayabileceği doğal Türkçe ile
sorulmalıdır.

25. Yanıtında liste, madde işareti veya numaralandırma kullanma.

SADECE ÖĞRENCİYE SORACAĞIN SORUYU ÜRET.
"""


    return prompt


# ============================================================
# İLK SORUYU OLUŞTUR
# ============================================================

def ilk_soruyu_olustur(stage):

    prompt = pedagojik_prompt(
        stage=stage,
        first_question=True
    )

    result, error = gemini_request(
        prompt=prompt,
        include_problem=True,
        temperature=0.2
    )

    if error:

        return guvenli_yedek_soru(stage), error

    return result.strip(), None


# ============================================================
# ÖĞRENCİ CEVABINA YANIT
# ============================================================

def ogrenciye_cevap_ver(
    stage,
    student_message
):

    prompt = pedagojik_prompt(
        stage=stage,
        student_message=student_message,
        first_question=False
    )

    result, error = gemini_request(
        prompt=prompt,
        include_problem=True,
        temperature=0.2
    )

    if error:

        return guvenli_yedek_soru(stage), error

    return result.strip(), None


# ============================================================
# METABİLİŞSEL SORU
# ============================================================

def metabilissel_soru_olustur():

    stage = "Metabilişsel Yansıtma"

    prompt = f"""
Sen ortaokul matematik öğrencisinin problem çözme sürecini
yansıtmasına yardımcı olan bir öğretmensin.

Öğrenci aşağıdaki problemi ve algoritmik düşünme basamaklarını
kullanarak çalışmıştır.

PROBLEM GÖRSELİNİ DİKKATLİCE İNCELE.

Öğrencinin çalışma sürecindeki cevapları:

{json.dumps(
    st.session_state.chat_history,
    ensure_ascii=False,
    indent=2
)}

Görevin öğrencinin kendi düşünme sürecini değerlendirmesine
yardımcı olacak TEK bir metabilişsel soru sormaktır.

Sadece bir soru sor.

Çözümü değerlendirme.

Doğru veya yanlış deme.

Öğrencinin cevabını söyleme.

1-2 kısa cümle kullan.

Soru öğrencinin kendi düşünme sürecine odaklansın.
"""

    result, error = gemini_request(
        prompt=prompt,
        include_problem=True,
        temperature=0.2
    )

    if error:
        return METABILISSEL_SORULAR[0], error

    return result.strip(), None


# ============================================================
# PROBLEM GÖRSELİNİ YÜKLE
# ============================================================

def problem_yukle(uploaded_file):

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()

    st.session_state.problem_image_bytes = image_bytes
    st.session_state.problem_image_name = uploaded_file.name
    st.session_state.problem_image_mime = uploaded_file.type
    st.session_state.problem_uploaded = True

    # Yeni problem olduğunda süreç sıfırlanır
    st.session_state.current_stage = 0
    st.session_state.chat_history = {}
    st.session_state.stage_completed = {}
    st.session_state.selected_alt_beceri = {}
    st.session_state.stage_first_question_done = {}
    st.session_state.last_ai_question = ""
    st.session_state.problem_analysis = ""

    st.success("Problem yüklendi.")


# ============================================================
# ÖĞRENCİ MESAJINI KAYDET
# ============================================================

def student_message_kaydet(
    stage,
    message
):

    if stage not in st.session_state.chat_history:

        st.session_state.chat_history[stage] = []

    st.session_state.chat_history[stage].append({
        "role": "student",
        "text": message,
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    })


# ============================================================
# AI MESAJINI KAYDET
# ============================================================

def ai_message_kaydet(
    stage,
    message
):

    if stage not in st.session_state.chat_history:

        st.session_state.chat_history[stage] = []

    st.session_state.chat_history[stage].append({
        "role": "assistant",
        "text": message,
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    })


# ============================================================
# AŞAMA TAMAMLAMA
# ============================================================

def asama_tamamla(stage):

    st.session_state.stage_completed[stage] = True

    student_id = st.session_state.student_id

    conversation = st.session_state.chat_history.get(
        stage,
        []
    )

    kayit = {

        "zaman": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "ogrenci_id": student_id,

        "problem": st.session_state.problem_image_name,

        "basamak": stage,

        "alt_beceri":
            st.session_state.selected_alt_beceri.get(
                stage,
                ""
            ),

        "konusma": json.dumps(
            conversation,
            ensure_ascii=False
        )
    }

    csv_kaydet(kayit)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Algoritmik Düşünme")

    st.divider()

    secim = st.radio(
        "Kullanıcı türü",
        [
            "Öğrenci",
            "Öğretmen / Yönetici"
        ]
    )

    if secim == "Öğrenci":

        st.session_state.admin_mode = False

        st.subheader("Öğrenci Bilgisi")

        student_id = st.text_input(
            "Öğrenci kodu",
            value=st.session_state.student_id
        )

        if student_id:

            st.session_state.student_id = student_id

        st.divider()

        st.subheader("Algoritmik Düşünme")

        for i, stage in enumerate(BASAMAKLAR):

            if stage == "Metabilişsel Yansıtma":
                continue

            if st.session_state.stage_completed.get(
                stage,
                False
            ):
                st.success(
                    f"✓ {i + 1}. {stage}"
                )

            elif i == st.session_state.current_stage:

                st.info(
                    f"▶ {i + 1}. {stage}"
                )

            else:

                st.write(
                    f"○ {i + 1}. {stage}"
                )

    else:

        st.session_state.admin_mode = True

        st.subheader("Öğretmen / Yönetici")

        password = st.text_input(
            "Şifre",
            type="password"
        )

        if password == ADMIN_PASSWORD:

            st.success("Yönetici girişi aktif.")

        else:

            st.warning(
                "Yönetici panelini görmek için şifre girin."
            )


# ============================================================
# ÖĞRETMEN PANELİ
# ============================================================

if st.session_state.admin_mode:

    st.title("👩‍🏫 Öğretmen / Araştırmacı Paneli")

    if os.path.exists(DATA_FILE):

        try:

            df = pd.read_csv(
                DATA_FILE,
                encoding="utf-8-sig"
            )

            st.metric(
                "Toplam kayıt",
                len(df)
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            csv_data = df.to_csv(
                index=False,
                encoding="utf-8-sig"
            )

            st.download_button(
                "CSV verilerini indir",
                data=csv_data,
                file_name="tez_verileri_final.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"CSV okunamadı: {e}"
            )

    else:

        st.info(
            "Henüz kayıt bulunmuyor."
        )

    st.stop()


# ============================================================
# ANA BAŞLIK
# ============================================================

st.title("🧠 Algoritmik Düşünme Atölyesi")

st.write(
    "Matematik problemlerini adım adım düşün, "
    "kendi çözüm yolunu oluştur ve sürecini değerlendir."
)


# ============================================================
# PROBLEM YÜKLEME
# ============================================================

st.header("1. Problem")

uploaded_file = st.file_uploader(
    "Problem görselini yükleyin",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    key="problem_uploader"
)

if uploaded_file is not None:

    if (
        not st.session_state.problem_uploaded
        or
        uploaded_file.name
        != st.session_state.problem_image_name
    ):

        problem_yukle(uploaded_file)


# ============================================================
# PROBLEMİ GÖSTER
# ============================================================

if st.session_state.problem_image_bytes:

    st.image(
        st.session_state.problem_image_bytes,
        caption="Çalışılacak problem",
        use_container_width=True
    )

else:

    st.info(
        "Başlamak için problem görselini yükleyin."
    )

    st.stop()


# ============================================================
# BASAMAK
# ============================================================

current_index = st.session_state.current_stage

if current_index >= 4:

    current_index = 4

current_stage = BASAMAKLAR[current_index]


st.header(
    f"2. {current_index + 1}. Basamak: {current_stage}"
)

st.info(
    BASAMAK_ACIKLAMALARI[current_stage]
)


# ============================================================
# METABİLİŞSEL YANSITMA
# ============================================================

if current_stage == "Metabilişsel Yansıtma":

    st.header("🪞 Metabilişsel Yansıtma")

    if (
        "Metabilişsel Yansıtma"
        not in st.session_state.stage_first_question_done
    ):

        question, error = metabilissel_soru_olustur()

        st.session_state.last_ai_question = question

        ai_message_kaydet(
            "Metabilişsel Yansıtma",
            question
        )

        st.session_state.stage_first_question_done[
            "Metabilişsel Yansıtma"
        ] = True

        if error:
            st.warning(error)

    question = st.session_state.last_ai_question

    st.markdown(
        f"**🤖 AI:** {question}"
    )

    answer = st.text_area(
        "Düşünceni yaz:",
        key="metacognitive_input"
    )

    confidence = st.slider(
        "Bu problemde kendi düşünme sürecine ne kadar güveniyorsun?",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button(
        "Yansıtmayı Kaydet",
        type="primary"
    ):

        if answer.strip():

            st.session_state.metacognitive_answer = answer
            st.session_state.confidence = confidence

            kayit = {

                "zaman":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "ogrenci_id":
                    st.session_state.student_id,

                "problem":
                    st.session_state.problem_image_name,

                "basamak":
                    "Metabilişsel Yansıtma",

                "metabilissel_yansitma":
                    answer,

                "guven":
                    confidence,

                "konusmalar":
                    json.dumps(
                        st.session_state.chat_history,
                        ensure_ascii=False
                    )
            }

            csv_kaydet(kayit)

            st.success(
                "Yansıtman kaydedildi."
            )

    st.stop()


# ============================================================
# ALT BECERİLERİ GÖSTER
# ============================================================

st.subheader("Bu basamakta odaklanılan beceriler")

alt_beceriler = ALT_BECERILER[current_stage]

for code, description in alt_beceriler.items():

    st.write(
        f"**{code} — {description}**"
    )


# ============================================================
# İLK SORUYU OLUŞTUR
# ============================================================

if current_stage not in st.session_state.chat_history:

    st.session_state.chat_history[current_stage] = []


if not st.session_state.stage_first_question_done.get(
    current_stage,
    False
):

    with st.spinner(
        "Problem görseli inceleniyor..."
    ):

        question, error = ilk_soruyu_olustur(
            current_stage
        )

    if error:
        st.warning(error)

    st.session_state.last_ai_question = question

    ai_message_kaydet(
        current_stage,
        question
    )

    st.session_state.stage_first_question_done[
        current_stage
    ] = True


# ============================================================
# SOHBET ALANI
# ============================================================

st.subheader("💬 Seninle birlikte düşünelim")

history = st.session_state.chat_history.get(
    current_stage,
    []
)

for message in history:

    if message["role"] == "student":

        with st.chat_message("user"):

            st.write(
                message["text"]
            )

    else:

        with st.chat_message("assistant"):

            st.write(
                message["text"]
            )


# ============================================================
# ÖĞRENCİ MESAJI
# ============================================================

student_message = st.chat_input(
    "Düşünceni yaz..."
)


if student_message:

    student_message = student_message.strip()

    if student_message:

        # ----------------------------------------------------
        # Öğrenci mesajını kaydet
        # ----------------------------------------------------

        student_message_kaydet(
            current_stage,
            student_message
        )

        # ----------------------------------------------------
        # Gemini'ye gönder
        #
        # BURASI KRİTİK:
        #
        # gemini_request() içinde
        # include_problem=True
        #
        # olduğu için problem görseli HER MESAJDA
        # Gemini'ye yeniden gönderiliyor.
        # ----------------------------------------------------

        with st.spinner(
            "Problemi ve cevabını birlikte değerlendiriyorum..."
        ):

            ai_response, error = ogrenciye_cevap_ver(
                current_stage,
                student_message
            )

        if error:

            st.warning(error)

        # ----------------------------------------------------
        # AI cevabını kaydet
        # ----------------------------------------------------

        ai_message_kaydet(
            current_stage,
            ai_response
        )

        st.session_state.last_ai_question = ai_response

        # ----------------------------------------------------
        # Ekranı yenile
        # ----------------------------------------------------

        st.rerun()


# ============================================================
# BASAMAĞI TAMAMLAMA
# ============================================================

st.divider()

st.subheader("Basamak değerlendirmesi")

selected_skill = st.selectbox(
    "Bu basamakta hangi beceri üzerinde çalıştığını düşünüyorsun?",
    options=[
        f"{code} — {desc}"
        for code, desc in alt_beceriler.items()
    ],
    key=f"skill_{current_stage}"
)

st.session_state.selected_alt_beceri[
    current_stage
] = selected_skill


col1, col2 = st.columns(2)


with col1:

    if st.button(
        "✓ Bu basamağı tamamladım",
        type="primary",
        use_container_width=True
    ):

        asama_tamamla(
            current_stage
        )

        if current_index < 4:

            st.session_state.current_stage = (
                current_index + 1
            )

            st.success(
                f"{current_stage} basamağı kaydedildi."
            )

            st.rerun()


with col2:

    if st.button(
        "🔄 Bu basamağı yeniden başlat",
        use_container_width=True
    ):

        st.session_state.chat_history[
            current_stage
        ] = []

        st.session_state.stage_first_question_done[
            current_stage
        ] = False

        st.session_state.last_ai_question = ""

        st.rerun()


# ============================================================
# ALT BİLGİ
# ============================================================

st.divider()

st.caption(
    "Algoritmik Düşünme Atölyesi — "
    "AI yalnızca rehberlik eder; çözüm öğrenci tarafından oluşturulur."
)
