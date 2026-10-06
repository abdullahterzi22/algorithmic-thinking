# ============================================================
# ALGORİTMİK PROBLEM ÇÖZME REHBERİ
# Konu: DOĞRUSAL İLİŞKİLER
# ============================================================

import streamlit as st
import requests
import pandas as pd
import os
import json
import base64
import io
import re
from datetime import datetime

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Algoritmik Problem Çözme Atölyesi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SABİTLER
# ============================================================

MODEL_NAME = "gemini-2.5-flash"

DATA_FILE = "tez_verileri_final.csv"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


# ============================================================
# GEMINI API ANAHTARI
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    st.error(
        "⚠️ Gemini API anahtarı bulunamadı.\n\n"
        "Streamlit Cloud → Settings → Secrets bölümüne "
        '`GEMINI_API_KEY = "AIza...."` şeklinde ekleyin.'
    )
    st.stop()


# ============================================================
# BASAMAKLAR
# ============================================================

BASAMAKLARI = [
    "Ayrıştırma",
    "Soyutlama",
    "Algoritma Tasarımı",
    "Hata Ayıklama"
]


# ============================================================
# ALT BECERİLER
# ============================================================

ALT_BECERILER = {

    "Ayrıştırma": {
        "A1": "Alt amacı belirleme",
        "A2": "Problemi anlamlı parçalara ayırma",
        "A3": "Parçalar arasındaki ilişkiyi kurma"
    },

    "Soyutlama": {
        "S1": "Problem için gerekli bilgileri seçme",
        "S2": "Gereksiz bilgileri ayırt etme",
        "S3": "Matematiksel ilişkileri belirleme",
        "S4": "Uygun matematiksel temsil oluşturma",
        "S5": "Örüntü veya temel yapıyı fark etme"
    },

    "Algoritma Tasarımı": {
        "AT1": "Gerekli işlemleri belirleme",
        "AT2": "İşlemleri uygun sıraya koyma",
        "AT3": "Ara sonuçları sonraki adımlarda kullanma",
        "AT4": "Çözüm prosedürü oluşturma",
        "AT5": "Çözümün benzer problemlere uygulanabilirliğini düşünme"
    },

    "Hata Ayıklama": {
        "HA1": "Çözümü test etme",
        "HA2": "Sonucu problem koşullarıyla karşılaştırma",
        "HA3": "Tutarsızlığı fark etme",
        "HA4": "Hatanın bulunduğu adımı belirleme",
        "HA5": "Hatalı adımı düzeltme",
        "HA6": "Düzeltilmiş çözümü yeniden test etme"
    }
}


# ============================================================
# BASAMAKLARA ÖZGÜ PEDAGOJİK ÇERÇEVE
# ============================================================

BASAMAK_ACIKLAMALARI = {

    "Ayrıştırma": """
Öğrencinin problem durumunu anlamlandırmasına yardım et.

Doğrusal ilişkiler bağlamında özellikle:
- hangi niceliklerin bulunduğunu,
- hangi niceliğin değiştiğini,
- hangi niceliğin buna bağlı olduğunu,
- sorunun öğrenciden ne istediğini,
- problemdeki parçaların birbirleriyle nasıl ilişkili olduğunu

öğrencinin kendisinin fark etmesine yardım et.

Bunları doğrudan söyleme.
Öğrencinin yerine parçalama yapma.
Hazır alt problem listesi verme.
""",

    "Soyutlama": """
Öğrencinin problemdeki matematiksel yapıyı günlük hikâyeden ayırmasına yardım et.

Doğrusal ilişkiler bağlamında özellikle:
- gerekli nicelikleri seçme,
- gereksiz bağlam ayrıntılarını ayırt etme,
- iki nicelik arasındaki değişimi fark etme,
- sabit değişim olup olmadığını inceleme,
- tablo, grafik, sözel ifade veya cebirsel temsil gibi uygun bir gösterim düşünme,
- örüntü veya temel yapıyı fark etme

konularında rehberlik et.

Ancak öğrenci kendisi ifade etmeden
"eğim", "sabit terim", "y = mx + b" gibi kavramları hazır olarak verme.
""",

    "Algoritma Tasarımı": """
Öğrencinin kendi çözüm yolunu yapılandırmasına yardım et.

Doğrusal ilişkiler bağlamında öğrenci;
- hangi bilgiyi kullanacağını,
- hangi işlemi veya düşünme eylemini yapacağını,
- bunların hangi sırada ilerleyeceğini,
- elde ettiği ara sonuçları nasıl kullanacağını,
- oluşturduğu yöntemin benzer bir probleme nasıl aktarılabileceğini

kendisi düşünmelidir.

Hazır çözüm sırası verme.
"Önce bunu yap, sonra bunu yap" şeklinde algoritma sunma.
""",

    "Hata Ayıklama": """
Öğrencinin oluşturduğu çözümü sınamasına yardım et.

Doğrusal ilişkiler bağlamında;
- başka bir değer üzerinden kontrol,
- tablo-grafik-sözel ifade arasındaki tutarlılık,
- problem koşullarına uygunluk,
- değişim miktarının tutarlılığı,
- başlangıç değerinin anlamı,
- birimlerin ve değerlerin uygunluğu

gibi kontrolleri öğrencinin kendisinin düşünmesini sağla.

Hatanın ne olduğunu söyleme.
Doğru sonucu verme.
Hangi adımın yanlış olduğunu doğrudan gösterme.
"""
}


# ============================================================
# YASAKLI İFADELER
# ============================================================

YASAK_IFADELER = [
    "doğru cevap",
    "cevap şudur",
    "sonuç şudur",
    "sonucun",
    "yanlış yaptın",
    "yanlış cevap",
    "hata yaptın",
    "şunu yap",
    "önce bunu yap",
    "sonra bunu yap",
    "işlemi yap",
    "formülü kullan",
    "çözüm şu",
    "cevap ",
    "ebob",
    "ekok"
]


# ============================================================
# GÜVENLİ YEDEK SORULAR
# ============================================================

FALLBACK_SORULAR = {

    "Ayrıştırma":
        "Bu problemde birbirinden ayrı düşünmen gereken iki nicelik ya da durum hangileri olabilir?",

    "Soyutlama":
        "Problemin hikâyesinden bağımsız düşündüğünde hangi iki nicelik arasındaki ilişkiyi incelemek daha anlamlı görünüyor?",

    "Algoritma Tasarımı":
        "Kurduğun ilişkiyi kullanarak sonuca ulaşmak için hangi düşünme adımını bundan sonra ele alacağını nasıl belirlersin?",

    "Hata Ayıklama":
        "Bulduğun ilişkinin gerçekten problemdeki durumu temsil edip etmediğini başka bir değer üzerinden nasıl sınayabilirsin?"
}


# ============================================================
# BAŞLANGIÇ SORULARI
# ============================================================

BASLANGIC_SORULARI = {

    "Ayrıştırma":
        "Bu problemi anlamlandırırken birbirinden ayırarak düşünmek isteyeceğin temel durumlar hangileri?",

    "Soyutlama":
        "Problemin günlük anlatımından matematiksel yapıya geçerken hangi nicelikler arasındaki ilişkiye odaklanmak istersin?",

    "Algoritma Tasarımı":
        "Kurmak istediğin çözüm yolunu oluştururken hangi düşünme adımından başlamayı uygun görüyorsun?",

    "Hata Ayıklama":
        "Oluşturduğun çözümün problem koşullarına uyup uymadığını kontrol etmek için neyi sınamak istersin?"
}


# ============================================================
# SESSION STATE
# ============================================================

if "student_id" not in st.session_state:
    st.session_state.student_id = ""

if "mode" not in st.session_state:
    st.session_state.mode = "Öğrenci"

if "current_step" not in st.session_state:
    st.session_state.current_step = "Ayrıştırma"

if "completed_stages" not in st.session_state:
    st.session_state.completed_stages = []

if "stage_ready" not in st.session_state:
    st.session_state.stage_ready = {
        "Ayrıştırma": False,
        "Soyutlama": False,
        "Algoritma Tasarımı": False,
        "Hata Ayıklama": False
    }

if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {
        stage: []
        for stage in BASAMAKLARI
    }

if "last_ai_result" not in st.session_state:
    st.session_state.last_ai_result = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "reflection_storage" not in st.session_state:
    st.session_state.reflection_storage = {}

if "confidence_storage" not in st.session_state:
    st.session_state.confidence_storage = {}

if "subskill_status" not in st.session_state:
    st.session_state.subskill_status = {
        stage: {
            key: False
            for key in ALT_BECERILER[stage]
        }
        for stage in BASAMAKLARI
    }

if "interaction_log" not in st.session_state:
    st.session_state.interaction_log = []


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def kelime_sayisi(text):
    return len(text.strip().split())


def temizle_text(text):
    if not text:
        return ""

    text = text.replace("**", "")
    text = text.replace("###", "")
    text = re.sub(r"\n\s*[-•]\s*", " ", text)
    text = re.sub(r"\n\s*\d+[\.\)]\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def soru_sayisi(text):
    return text.count("?")


def ai_yaniti_guvenli_mi(text):
    if not text:
        return False

    text = text.strip()

    # Çok uzun cevapları reddet
    if kelime_sayisi(text) > 55:
        return False

    # Tam olarak bir soru
    if soru_sayisi(text) != 1:
        return False

    # Liste biçimini reddet
    if re.search(r"(^|\n)\s*[-•]", text):
        return False

    if re.search(r"(^|\n)\s*\d+[\.\)]", text):
        return False

    lower = text.lower()

    for yasak in YASAK_IFADELER:
        if yasak.lower() in lower:
            return False

    return True


def guvenli_yedek_soru(basamak):
    return FALLBACK_SORULAR.get(
        basamak,
        "Düşünceni bir sonraki adım için nasıl geliştirebilirsin?"
    )


def log_kaydet(
    event_type,
    stage=None,
    user_text="",
    ai_text="",
    metadata=None
):

    kayit = {
        "timestamp": datetime.now().isoformat(),
        "student_id": st.session_state.student_id,
        "event_type": event_type,
        "stage": stage or "",
        "user_text": user_text,
        "ai_text": ai_text,
        "metadata": json.dumps(
            metadata or {},
            ensure_ascii=False
        )
    }

    st.session_state.interaction_log.append(kayit)

    try:
        df_new = pd.DataFrame([kayit])

        if os.path.exists(DATA_FILE):
            df_old = pd.read_csv(DATA_FILE)

            df_all = pd.concat(
                [df_old, df_new],
                ignore_index=True
            )
        else:
            df_all = df_new

        df_all.to_csv(
            DATA_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as e:
        print("CSV kayıt hatası:", e)


# ============================================================
# PROBLEM GÖRSELİ
# ============================================================

def problem_gorselini_hazirla(uploaded_file):

    try:

        image = Image.open(uploaded_file).convert("RGBA")

        max_width = 1100
        max_height = 850

        width, height = image.size

        ratio = min(
            max_width / width,
            max_height / height,
            1
        )

        new_size = (
            int(width * ratio),
            int(height * ratio)
        )

        image = image.resize(
            new_size,
            Image.LANCZOS
        )

        return image

    except Exception:
        return None


# ============================================================
# GÖRSELİ GEMINI'YE ÇEVİR
# ============================================================

def image_to_inline_data():

    if not st.session_state.uploaded_image_bytes:
        return None

    try:

        encoded = base64.b64encode(
            st.session_state.uploaded_image_bytes
        ).decode("utf-8")

        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": encoded
            }
        }

    except Exception:
        return None


# ============================================================
# KONUŞMA GEÇMİŞİNİ METNE ÇEVİR
# ============================================================

def konusma_gecmisini_metne_donustur(basamak):

    gecmis = st.session_state.chat_storage.get(
        basamak,
        []
    )

    if not gecmis:
        return "Bu basamakta henüz öğrenci yanıtı bulunmuyor."

    satirlar = []

    for mesaj in gecmis:

        role = mesaj.get("role", "")
        content = mesaj.get("content", "")

        if role == "user":
            satirlar.append(
                f"ÖĞRENCİ: {content}"
            )

        elif role == "assistant":
            satirlar.append(
                f"YAPAY ZEKA: {content}"
            )

    return "\n".join(satirlar)


# ============================================================
# GERÇEK GEMINI CONTENTS OLUŞTUR
# ============================================================

def gemini_contents_olustur(basamak):

    gecmis = st.session_state.chat_storage.get(
        basamak,
        []
    )

    contents = []

    # Gemini konuşmasının user ile başlaması gerekiyor.
    #
    # İlk bot mesajı yalnızca başlangıç sorusudur.
    # Öğrenci mesajı gelmeden önceki assistant mesajını
    # API geçmişine göndermiyoruz.

    ilk_ogrenci_bulundu = False

    for mesaj in gecmis:

        role = mesaj.get("role")
        content = mesaj.get("content", "")

        if role == "user":

            parts = [
                {
                    "text": content
                }
            ]

            # Problem görselini ilk öğrenci mesajına ekle
            if not ilk_ogrenci_bulundu:

                image_data = image_to_inline_data()

                if image_data:
                    parts.append(image_data)

                ilk_ogrenci_bulundu = True

            contents.append({
                "role": "user",
                "parts": parts
            })

        elif role == "assistant":

            # user gelmeden model mesajı gönderme
            if ilk_ogrenci_bulundu:

                contents.append({
                    "role": "model",
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                })

    return contents


# ============================================================
# SYSTEM PROMPT
# ============================================================

def sistem_promptu_olustur(basamak):

    alt_beceriler = ALT_BECERILER[basamak]

    alt_beceri_metni = "\n".join(
        f"- {k}: {v}"
        for k, v in alt_beceriler.items()
    )

    tamamlananlar = []

    for key, value in st.session_state.subskill_status[
        basamak
    ].items():

        if value:
            tamamlananlar.append(key)

    tamamlanan_metni = (
        ", ".join(tamamlananlar)
        if tamamlananlar
        else "Henüz belirlenmiş bir alt beceri tamamlanmadı."
    )

    gecmis = konusma_gecmisini_metne_donustur(
        basamak
    )

    return f"""
SEN BİR MATEMATİK ÇÖZÜCÜ DEĞİLSİN.
Sen, ortaokul öğrencisinin DOĞRUSAL İLİŞKİLER konusundaki
problem çözme sürecini destekleyen pedagojik bir rehbersin.

ÖĞRENCİNİN YERİNE PROBLEMİ ÇÖZME.

==================================================
ÇALIŞILAN BASAMAK
==================================================

{basamak}

Bu basamağın amacı:

{BASAMAK_ACIKLAMALARI[basamak]}

==================================================
ALT BECERİLER
==================================================

{alt_beceri_metni}

Bu basamakta daha önce öğrencide görüldüğü kabul edilen
alt beceriler:

{tamamlanan_metni}

==================================================
EN ÖNEMLİ DİYALOG KURALLARI
==================================================

1. Her yanıtında YALNIZCA BİR SORU sor.

2. Yanıtında tam olarak BİR adet soru işareti (?) bulunmalıdır.

3. Öğrenci yanıt vermeden yeni bir düşünme adımına geçme.

4. Öğrencinin son söylediğini dikkate al.

5. Öğrencinin zaten söylediği bir şeyi tekrar sorma.

6. Önceki sorunu farklı kelimelerle yeniden sorma.

7. Öğrencinin cevabında zaten bulunan bilgiyi yeniden isteme.

8. Öğrencinin düşüncesini kısa biçimde yansıtabilirsin;
   ardından yalnızca bir soru sor.

9. Hazır alt problem listesi oluşturma.

10. Numaralı öneriler verme.

11. Madde işaretli soru listesi verme.

12. Öğrenci için çözüm planını sen oluşturma.

13. "Önce bunu yap, sonra bunu yap" biçiminde yönlendirme yapma.

14. İşlem sonucu verme.

15. Sayısal hesaplama yapma.

16. Formül verme.

17. Öğrencinin kendisinin söylemediği matematiksel kavramı
    gereksiz yere isimlendirme.

18. Özellikle öğrenci kendisi ifade etmeden
    "y = mx + b", "eğim", "sabit terim" gibi kavramları
    hazır olarak verme.

19. "Doğru cevap", "yanlış cevap", "hata yaptın" gibi
    değerlendirme ifadeleri kullanma.

20. Öğrencinin cevabını doğrudan doğru/yanlış olarak
    sınıflandırma.

21. "Harika bir başlangıç noktası!", "Mükemmel!",
    "Çok güzel!" gibi kalıp övgüler kullanma.

22. "Hadi birlikte çözelim" deme.

23. Yanıtın 1-3 kısa cümle olsun.

24. Yanıtın doğal Türkçe olsun.

25. Öğrencinin yaş düzeyine uygun konuş.

26. Öğrenciye cevap vermesi için alan bırak.

27. Bir sorunun içinde iki ayrı soru sorma.

28. "ve ...?" ile ikinci bir soru ekleme.

29. Öğrenci düşüncesini kendisi geliştirmeli.

30. Amacın öğrencinin cevabı bulması değil,
    düşünme sürecini kendisinin kurmasıdır.

==================================================
DOĞRUSAL İLİŞKİLER İÇİN ÖZEL KURAL
==================================================

Problemde iki nicelik arasındaki ilişki varsa öğrencinin
ilişkiyi kendisinin fark etmesine yardım et.

Örneğin öğrenci henüz ilişkiyi fark etmemişse doğrudan
"bu doğrusal ilişkidir" deme.

Öğrencinin:
- değişen nicelikleri,
- sabit kalan özellikleri,
- iki nicelik arasındaki ilişkiyi,
- değişimin düzenini,
- tabloyu,
- grafiği,
- sözel ilişkiyi,
- cebirsel ifadeyi

kendisi fark etmesine fırsat ver.

Öğrenci bunlardan birini kendisi ifade ederse,
o düşünce üzerinden ilerleyebilirsin.

==================================================
GÖRSEL PROBLEM VARSA
==================================================

Yüklenen problem görselini dikkatlice incele.

Görselde tablo, grafik, şekil veya metin varsa
öğrencinin görseldeki ilgili bilgiyi kendisinin bulmasını sağla.

Görseldeki sayıları gereksiz yere sen söyleme.

Örneğin:
"Grafikte hangi iki değerin birlikte değiştiğini görüyorsun?"
gibi öğrencinin incelemesini sağlayan tek bir soru sorabilirsin.

Ancak her problemde aynı kalıp soruyu kullanma.

==================================================
KONUŞMA GEÇMİŞİ
==================================================

Aşağıdaki konuşmayı mutlaka dikkate al:

{gecmis}

Özellikle SON ÖĞRENCİ CEVABINA göre yanıt üret.

Öğrencinin önceki cevabını görmezden gelme.

==================================================
ÇIKTI
==================================================

Yalnızca öğrencinin bir sonraki düşünmesini sağlayacak
tek ve kısa bir soru üret.

Başka açıklama ekleme.
Liste oluşturma.
Çözüm verme.
"""


# ============================================================
# GEMINI ÇAĞRISI
# ============================================================

def gemini_yanit_uret(basamak):

    contents = gemini_contents_olustur(
        basamak
    )

    if not contents:
        return guvenli_yedek_soru(basamak), {
            "status": "no_history"
        }

    system_instruction = sistem_promptu_olustur(
        basamak
    )

    payload = {

        "system_instruction": {
            "parts": [
                {
                    "text": system_instruction
                }
            ]
        },

        "contents": contents,

        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 180,

            "responseMimeType": "text/plain"
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    url = (
        GEMINI_URL
        + "?key="
        + GEMINI_API_KEY
    )

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        # ----------------------------------------------------
        # HATA KONTROLLERİ
        # ----------------------------------------------------

        if response.status_code != 200:

            hata_metni = response.text[:1000]

            return (
                guvenli_yedek_soru(basamak),
                {
                    "status": "api_error",
                    "http_status": response.status_code,
                    "error": hata_metni
                }
            )

        data = response.json()

        try:

            text = (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

        except Exception:

            return (
                guvenli_yedek_soru(basamak),
                {
                    "status": "empty_response",
                    "raw": data
                }
            )

        text = temizle_text(text)

        # ----------------------------------------------------
        # PEDAGOJİK GÜVENLİK KONTROLÜ
        # ----------------------------------------------------

        if not ai_yaniti_guvenli_mi(text):

            return (
                guvenli_yedek_soru(basamak),
                {
                    "status": "unsafe_ai_output",
                    "original": text
                }
            )

        return (
            text,
            {
                "status": "success"
            }
        )

    except requests.exceptions.Timeout:

        return (
            guvenli_yedek_soru(basamak),
            {
                "status": "timeout"
            }
        )

    except Exception as e:

        return (
            guvenli_yedek_soru(basamak),
            {
                "status": "exception",
                "error": str(e)
            }
        )


# ============================================================
# ALT BECERİLERİ İÇİN YEREL DURUM GÜNCELLEME
# ============================================================

def alt_beceri_durumunu_guncelle(
    basamak,
    ogrenci_cevabi
):

    """
    Bu fonksiyon LLM'nin 'öğrenci bunu tamamladı' kararını
    doğrudan kabul etmek yerine yalnızca güçlü dilsel
    göstergeleri yakalar.

    Araştırma açısından kesin değerlendirme değildir.
    Nihai değerlendirme öğretmen/araştırmacı tarafından
    yapılabilir.
    """

    text = ogrenci_cevabi.lower()

    if basamak == "Ayrıştırma":

        if any(
            ifade in text
            for ifade in [
                "istenen",
                "isteniyor",
                "bulmam",
                "bulunması"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["A1"] = True

        if any(
            ifade in text
            for ifade in [
                "ayır",
                "parça",
                "iki kısım",
                "üç kısım"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["A2"] = True

        if any(
            ifade in text
            for ifade in [
                "arasındaki ilişki",
                "bağlantı",
                "birbirine bağlı"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["A3"] = True

    elif basamak == "Soyutlama":

        if any(
            ifade in text
            for ifade in [
                "gerekli",
                "önemli bilgi",
                "kullanacağım"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["S1"] = True

        if any(
            ifade in text
            for ifade in [
                "gereksiz",
                "önemsiz",
                "lüzumsuz"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["S2"] = True

        if any(
            ifade in text
            for ifade in [
                "ilişki",
                "bağlantı",
                "değişim"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["S3"] = True

        if any(
            ifade in text
            for ifade in [
                "tablo",
                "grafik",
                "denklem",
                "ifade",
                "gösterim"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["S4"] = True

        if any(
            ifade in text
            for ifade in [
                "örüntü",
                "düzen",
                "sabit artış",
                "sabit değişim"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["S5"] = True

    elif basamak == "Algoritma Tasarımı":

        if any(
            ifade in text
            for ifade in [
                "işlem",
                "hesap",
                "bulacağım"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["AT1"] = True

        if any(
            ifade in text
            for ifade in [
                "sıra",
                "ardından",
                "daha sonra"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["AT2"] = True

        if any(
            ifade in text
            for ifade in [
                "ara sonuç",
                "bulduğum",
                "elde ettiğim"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["AT3"] = True

        if any(
            ifade in text
            for ifade in [
                "yöntem",
                "adım",
                "prosedür"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["AT4"] = True

        if any(
            ifade in text
            for ifade in [
                "benzer",
                "başka problem",
                "başka soruda"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["AT5"] = True

    elif basamak == "Hata Ayıklama":

        if any(
            ifade in text
            for ifade in [
                "kontrol",
                "test",
                "deneyeceğim"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["HA1"] = True

        if any(
            ifade in text
            for ifade in [
                "koşul",
                "şarta",
                "verilenlere"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["HA2"] = True

        if any(
            ifade in text
            for ifade in [
                "tutarsız",
                "uyuşmuyor",
                "uymuyor"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["HA3"] = True

        if any(
            ifade in text
            for ifade in [
                "adımda",
                "işlemde",
                "burada hata"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["HA4"] = True

        if any(
            ifade in text
            for ifade in [
                "düzelteceğim",
                "düzelt",
                "değiştireceğim"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["HA5"] = True

        if any(
            ifade in text
            for ifade in [
                "tekrar kontrol",
                "yeniden test",
                "tekrar dene"
            ]
        ):
            st.session_state.subskill_status[
                basamak
            ]["HA6"] = True


# ============================================================
# BASAMAK TAMAMLAMA
# ============================================================

def basamak_tamamlanabilir_mi(basamak):

    durum = st.session_state.subskill_status[
        basamak
    ]

    tamamlanan = sum(
        1 for value in durum.values()
        if value
    )

    toplam = len(durum)

    # Çok katı otomatik tamamlama yerine eşik kullanıyoruz.
    #
    # Ancak araştırmada nihai kararın öğretmen/araştırmacı
    # tarafından verilmesi daha güvenlidir.

    if basamak == "Ayrıştırma":
        return tamamlanan >= 3

    if basamak == "Soyutlama":
        return tamamlanan >= 4

    if basamak == "Algoritma Tasarımı":
        return tamamlanan >= 4

    if basamak == "Hata Ayıklama":
        return tamamlanan >= 4

    return False


# ============================================================
# SONRAKİ BASAMAĞA GEÇ
# ============================================================

def sonraki_basamaga_gec():

    current_index = BASAMAKLARI.index(
        st.session_state.current_step
    )

    if current_index >= len(BASAMAKLARI) - 1:
        return

    mevcut = BASAMAKLARI[current_index]

    st.session_state.completed_stages.append(
        mevcut
    )

    sonraki = BASAMAKLARI[current_index + 1]

    st.session_state.current_step = sonraki

    st.session_state.stage_ready[mevcut] = True

    # Yeni basamakta başlangıç sorusu
    if not st.session_state.chat_storage[sonraki]:

        st.session_state.chat_storage[sonraki].append({
            "role": "assistant",
            "content": BASLANGIC_SORULARI[sonraki]
        })


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎯 Algoritmik Düşünme")

    st.session_state.mode = st.radio(
        "Çalışma modu",
        [
            "Öğrenci",
            "Öğretmen / Araştırmacı"
        ],
        index=(
            0
            if st.session_state.mode == "Öğrenci"
            else 1
        )
    )

    st.divider()

    st.session_state.student_id = st.text_input(
        "Öğrenci kodu",
        value=st.session_state.student_id,
        placeholder="Örn. O01"
    )

    st.divider()

    st.markdown("### Çalışma basamakları")

    for stage in BASAMAKLARI:

        if stage == st.session_state.current_step:
            st.markdown(
                f"🟢 **{stage}**"
            )

        elif stage in st.session_state.completed_stages:
            st.markdown(
                f"✅ {stage}"
            )

        else:
            st.markdown(
                f"⚪ {stage}"
            )

    st.divider()

    st.caption(
        "Konu: Doğrusal İlişkiler"
    )


# ============================================================
# ÖĞRETMEN / ARAŞTIRMACI PANELİ
# ============================================================

if st.session_state.mode == "Öğretmen / Araştırmacı":

    st.title("📊 Öğretmen / Araştırmacı Paneli")

    st.subheader("Basamak durumu")

    durum_rows = []

    for stage in BASAMAKLARI:

        durum = st.session_state.subskill_status[
            stage
        ]

        for key, description in ALT_BECERILER[
            stage
        ].items():

            durum_rows.append({
                "Basamak": stage,
                "Kod": key,
                "Alt beceri": description,
                "Durum": (
                    "Gözlendi"
                    if durum[key]
                    else "Henüz gözlenmedi"
                )
            })

    st.dataframe(
        pd.DataFrame(durum_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Konuşma kayıtları")

    all_rows = []

    for stage in BASAMAKLARI:

        for message in st.session_state.chat_storage[
            stage
        ]:

            all_rows.append({
                "Basamak": stage,
                "Rol": message["role"],
                "Mesaj": message["content"]
            })

    if all_rows:

        st.dataframe(
            pd.DataFrame(all_rows),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    if os.path.exists(DATA_FILE):

        try:

            df = pd.read_csv(DATA_FILE)

            st.subheader("Kayıt dosyası")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = df.to_csv(
                index=False,
                encoding="utf-8-sig"
            )

            st.download_button(
                "⬇️ CSV indir",
                data=csv_data,
                file_name=DATA_FILE,
                mime="text/csv"
            )

        except Exception as e:

            st.warning(
                f"CSV okunamadı: {e}"
            )

    st.stop()


# ============================================================
# ANA BAŞLIK
# ============================================================

st.title(
    "🎯 Algoritmik Problem Çözme Rehberi"
)

st.markdown(
    """
**Doğrusal İlişkiler**

Bu ortamda amaç problemi senin yerine çözmek değil;
kendi düşünme yolunu oluşturmanı desteklemektir.
"""
)


# ============================================================
# PROBLEM GÖRSELİ YÜKLE
# ============================================================

st.subheader("📷 Problem")

uploaded_file = st.file_uploader(
    "Problem görselini yükle",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


if uploaded_file is not None:

    image = problem_gorselini_hazirla(
        uploaded_file
    )

    if image is not None:

        st.session_state.uploaded_image = image

        uploaded_file.seek(0)

        st.session_state.uploaded_image_bytes = (
            uploaded_file.read()
        )


# ============================================================
# BASAMAK SEÇİMİ
# ============================================================

st.divider()

current_step = st.session_state.current_step

st.subheader(
    f"🔹 {current_step}"
)


# ============================================================
# ÇALIŞMA ALANI
# ============================================================

left_col, right_col = st.columns(
    [1.15, 1],
    gap="large"
)


# ============================================================
# SOL PANEL — PROBLEM + ÇİZİM
# ============================================================

with left_col:

    st.markdown(
        "### Problem üzerinde çalış"
    )

    if st.session_state.uploaded_image is not None:

        canvas_bg = (
            st.session_state.uploaded_image
            .convert("RGBA")
        )

        canvas_width = min(
            canvas_bg.width,
            900
        )

        canvas_height = int(
            canvas_bg.height
            * canvas_width
            / canvas_bg.width
        )

        canvas_bg = canvas_bg.resize(
            (
                canvas_width,
                canvas_height
            ),
            Image.LANCZOS
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.15)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            background_image=canvas_bg,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            point_display_radius=0,
            key=f"canvas_{current_step}"
        )

        if canvas_result.image_data is not None:

            st.session_state[
                f"canvas_{current_step}"
            ] = canvas_result.image_data

    else:

        st.info(
            "Çalışmak istediğin problem görselini yukarıdan yükle."
        )


# ============================================================
# SAĞ PANEL — CHAT
# ============================================================

with right_col:

    st.markdown(
        "### 💬 Düşünme Rehberi"
    )

    chat_container = st.container(
        height=520
    )

    with chat_container:

        for message in st.session_state.chat_storage[
            current_step
        ]:

            if message["role"] == "assistant":

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        message["content"]
                    )

            else:

                with st.chat_message(
                    "user"
                ):

                    st.write(
                        message["content"]
                    )


# ============================================================
# İLK SORUYU GÖSTER
# ============================================================

if not st.session_state.chat_storage[
    current_step
]:

    ilk_soru = BASLANGIC_SORULARI[
        current_step
    ]

    st.session_state.chat_storage[
        current_step
    ].append({
        "role": "assistant",
        "content": ilk_soru
    })

    st.rerun()


# ============================================================
# ÖĞRENCİ MESAJI
# ============================================================

student_message = st.chat_input(
    "Düşünceni buraya yaz..."
)


if student_message:

    student_message = student_message.strip()

    if student_message:

        # ----------------------------------------------
        # Öğrenci mesajını kaydet
        # ----------------------------------------------

        st.session_state.chat_storage[
            current_step
        ].append({
            "role": "user",
            "content": student_message
        })

        # ----------------------------------------------
        # Alt beceri durumunu güncelle
        # ----------------------------------------------

        alt_beceri_durumunu_guncelle(
            current_step,
            student_message
        )

        # ----------------------------------------------
        # Gemini'den rehber soru
        # ----------------------------------------------

        ai_answer, ai_meta = gemini_yanit_uret(
            current_step
        )

        # ----------------------------------------------
        # AI cevabını kaydet
        # ----------------------------------------------

        st.session_state.chat_storage[
            current_step
        ].append({
            "role": "assistant",
            "content": ai_answer
        })

        st.session_state.last_ai_result = ai_meta

        # ----------------------------------------------
        # Log
        # ----------------------------------------------

        log_kaydet(
            event_type="chat",
            stage=current_step,
            user_text=student_message,
            ai_text=ai_answer,
            metadata=ai_meta
        )

        # ----------------------------------------------
        # Otomatik basamak değerlendirmesi
        # ----------------------------------------------

        if basamak_tamamlanabilir_mi(
            current_step
        ):

            st.session_state.stage_ready[
                current_step
            ] = True

        st.rerun()


# ============================================================
# ALT BECERİ DURUMU
# ============================================================

st.divider()

with st.expander(
    "🔎 Bu basamaktaki düşünme göstergeleri",
    expanded=False
):

    current_status = (
        st.session_state.subskill_status[
            current_step
        ]
    )

    for key, description in ALT_BECERILER[
        current_step
    ].items():

        status = current_status[key]

        if status:

            st.markdown(
                f"✅ **{key}** — {description}"
            )

        else:

            st.markdown(
                f"⬜ **{key}** — {description}"
            )


# ============================================================
# BASAMAK TAMAMLAMA ALANI
# ============================================================

st.divider()

stage_ready = st.session_state.stage_ready[
    current_step
]

if stage_ready:

    st.success(
        f"{current_step} basamağı için yeterli düşünme göstergesi oluştu."
    )

    if current_step != BASAMAKLARI[-1]:

        if st.button(
            "➡️ Sonraki basamağa geç",
            type="primary"
        ):

            log_kaydet(
                event_type="stage_completed",
                stage=current_step,
                metadata={
                    "subskills":
                    st.session_state.subskill_status[
                        current_step
                    ]
                }
            )

            sonraki_basamaga_gec()

            st.rerun()

    else:

        st.success(
            "🎉 Dört algoritmik düşünme basamağı tamamlandı."
        )


else:

    st.caption(
        "Bir sonraki basamağa geçmeden önce mevcut basamaktaki "
        "düşünme göstergelerinin oluşması beklenir."
    )


# ============================================================
# METABİLİŞSEL YANSITMA
# ============================================================

st.divider()

st.subheader(
    "🧠 Metabilişsel Yansıtma"
)

st.write(
    "Bu basamakta kendi düşünme sürecini değerlendir."
)

reflection = st.text_area(
    "Bu basamakta nasıl düşündüğünü, neyi fark ettiğini "
    "ve düşünceni nasıl geliştirdiğini kendi cümlelerinle yaz.",
    value=st.session_state.reflection_storage.get(
        current_step,
        ""
    ),
    key=f"reflection_{current_step}"
)

confidence = st.slider(
    "Bu basamaktaki düşünmene ne kadar güveniyorsun?",
    min_value=1,
    max_value=5,
    value=st.session_state.confidence_storage.get(
        current_step,
        3
    ),
    key=f"confidence_{current_step}"
)

if st.button(
    "💾 Yansıtmayı kaydet",
    key=f"save_reflection_{current_step}"
):

    st.session_state.reflection_storage[
        current_step
    ] = reflection

    st.session_state.confidence_storage[
        current_step
    ] = confidence

    log_kaydet(
        event_type="metacognitive_reflection",
        stage=current_step,
        metadata={
            "reflection": reflection,
            "confidence": confidence
        }
    )

    st.success(
        "Yansıtman kaydedildi."
    )


# ============================================================
# SON ÖZET
# ============================================================

if (
    len(st.session_state.completed_stages)
    == len(BASAMAKLARI)
):

    st.divider()

    st.subheader(
        "📋 Çalışma Özeti"
    )

    for stage in BASAMAKLARI:

        durum = st.session_state.subskill_status[
            stage
        ]

        tamamlanan = [
            key
            for key, value in durum.items()
            if value
        ]

        st.markdown(
            f"**{stage}:** "
            + (
                ", ".join(tamamlanan)
                if tamamlanan
                else "Gösterge oluşmadı."
            )
        )

    st.success(
        "Algoritmik problem çözme çalışması tamamlandı."
    )


# ============================================================
# GELİŞTİRİCİ / DEBUG BİLGİSİ
# ============================================================

if st.session_state.mode == "Öğretmen / Araştırmacı":

    st.divider()

    st.subheader("Teknik durum")

    st.write({
        "model": MODEL_NAME,
        "gemini_key_loaded": bool(GEMINI_API_KEY),
        "current_stage": st.session_state.current_step,
        "uploaded_image": (
            st.session_state.uploaded_image
            is not None
        ),
        "last_ai_result": (
            st.session_state.last_ai_result
        )
    })
