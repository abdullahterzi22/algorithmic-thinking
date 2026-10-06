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
# 3. ALGORİTMİK DÜŞÜNME BASAMAKLARI
# ============================================================

BASAMAKLARI = [
    "1. Ayrıştırma",
    "2. Soyutlama",
    "3. Algoritma Tasarımı",
    "4. Hata Ayıklama",
    "5. Metabilişsel Yansıtma"
]


# ============================================================
# 4. SADELEŞTİRİLMİŞ PEDAGOJİK PROTOKOL
# ============================================================

# Önemli: Modelin aynı anda çok fazla kuralı görmesini önlemek için
# ortak kurallar kısa tutuldu. Her basamak için yalnızca o basamağın
# görevi ayrıca gönderiliyor.
PEDAGOJIK_TEMEL = """
Sen ortaokul matematik öğrencisine rehberlik eden bir düşünme yardımcısısın.

Temel amaç: Öğrencinin matematiksel düşünmesini destekle; problemi öğrencinin yerine çözme.

ZORUNLU DİYALOG KURALLARI:
- Her mesajda yalnızca BİR ana soru sor.
- Öğrenci cevap vermeden yeni bir soru sorma veya sonraki adıma geçme.
- Cevabı, formülü, işlemi veya hazır çözüm planını öğrencinin yerine verme.
- Öğrencinin söylemediği bilgileri varsayma.
- Hata varsa “yanlış” deme; öğrencinin kendi cevabını kontrol etmesini sağlayan tek bir soru sor.
- Öğrenci “bilmiyorum” derse soruyu daha küçük bir adıma indir.
- Kısa ve doğal Türkçe kullan; genellikle 1-3 kısa cümle yeterlidir.
- Uzun açıklama, soru listesi ve hazır alt problem listesi verme.
- “Harika bir başlangıç!”, “Harika bir soru!” gibi kalıp övgüleri kullanma.

ÇIKTI KURALI:
Yalnızca öğrencinin göreceği kısa rehberlik mesajını üret.
"""

BASAMAK_TALIMATLARI = {
    "1. Ayrıştırma": """
MEVCUT AŞAMA: AYRIŞTIRMA
Amaç: Öğrencinin problemin verilenlerini, istenenini ve gerekirse çözüm için gerekli alt parçaları kendisinin fark etmesini sağla.
Bu aşamada çözüm yöntemini veya işlemleri söyleme.
Önce öğrencinin son cevabındaki eksik tek noktaya odaklan ve yalnızca bir soru sor.
Alt problemleri öğrencinin yerine listeleme.
""",

    "2. Soyutlama": """
MEVCUT AŞAMA: SOYUTLAMA
Amaç: Öğrencinin problemdeki önemli matematiksel bilgileri, ilişkileri, düzenlilikleri ve yapıları kendisinin fark etmesini sağla.
Öğrenci kendisi ifade etmeden “doğrusal ilişki”, “eğim”, “fonksiyon” gibi kavramları çözüm olarak ortaya koyma.
Son cevaba bağlı tek bir düşünme sorusu sor.
""",

    "3. Algoritma Tasarımı": """
MEVCUT AŞAMA: ALGORİTMA TASARIMI
Amaç: Öğrencinin kendi çözüm adımlarını oluşturmasını sağla.
Hazır algoritma veya “önce bunu, sonra bunu yap” biçiminde çözüm planı verme.
Öğrencinin önerdiği adımı açıklamasını veya bir sonraki adımı kendisinin belirlemesini sağlayan tek bir soru sor.
""",

    "4. Hata Ayıklama": """
MEVCUT AŞAMA: HATA AYIKLAMA
Amaç: Öğrencinin kendi çözümünü kontrol ederek olası hatayı kendisinin fark etmesini sağla.
Hatanın yerini veya doğru cevabı doğrudan söyleme.
Öğrencinin işlemini, sonucunu veya problem koşulunu kontrol etmesini sağlayan tek bir soru sor.
""",

    "5. Metabilişsel Yansıtma": """
MEVCUT AŞAMA: METABİLİŞSEL YANSITMA
Amaç: Öğrencinin kendi düşünme ve problem çözme sürecini değerlendirmesini sağla.
Öğrencinin ne yaptığını, nerede zorlandığını, neyin işe yaradığını veya benzer bir problemde neyi değiştireceğini düşünmesine yardımcı ol.
Yalnızca bir yansıtma sorusu sor.
"""
}


# ============================================================
# 5. GÖRSEL ANALİZ PROTOKOLÜ
# ============================================================
GORSEL_PROTOKOL = """
GÖRSEL KURALI:
Problem görseli, grafik, tablo veya çizim varsa görseli dikkatle incele; ancak görseldeki matematiksel bilgileri öğrencinin yerine açıklama.
Gerekirse öğrenciden belirli değeri veya ilişkiyi görsel üzerinde göstermesini iste.
Görsel okunmuyorsa bilgi uydurma; öğrenciden gördüğü değeri söylemesini iste.
Çizim varsa anlamını öğrencinin yerine yorumlama.
Bu kurallar mevcut aşamanın tek-soru kuralına tabidir.
"""


# ============================================================
# 6. SİSTEM PROMPTU
# ============================================================
def sistem_promptu_olustur(step):
    return "\n\n".join([
        PEDAGOJIK_TEMEL,
        BASAMAK_TALIMATLARI[step],
        GORSEL_PROTOKOL
    ])


# ============================================================
# 8. METABİLİŞSEL SORULAR
# ============================================================

METABILISSEL_SORULAR = {

    "1. Ayrıştırma":
        "Problemi parçalara ayırırken neleri fark ettin?",

    "2. Soyutlama":
        "Problemde hangi bilgilerin önemli olduğunu fark ettin?",

    "3. Algoritma Tasarımı":
        "Çözüm planını oluştururken seni hangi düşünce yönlendirdi?",

    "4. Hata Ayıklama":
        "Çözümünü kontrol ederken özellikle neye dikkat ettin?",

    "5. Metabilişsel Yansıtma":
        "Bu problemi çözerken kendi düşünme sürecin hakkında ne fark ettin?"
}


# ============================================================
# 9. VERİ KAYDI
# ============================================================

DATA_FILE = "tez_verileri_v37.csv"


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
            f"Veri kaydı sırasında hata oluştu: {e}"
        )


# ============================================================
# 10. GÖRSELİ BASE64
# ============================================================

def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# 11. CANVAS + ORİJİNAL GÖRSEL
# ============================================================

def canvas_gorselini_birlestir(
    original_image,
    canvas_image
):

    try:

        original = original_image.convert(
            "RGBA"
        )

        overlay = Image.fromarray(
            canvas_image.astype("uint8")
        ).convert("RGBA")

        overlay = overlay.resize(
            original.size
        )

        combined = Image.alpha_composite(
            original,
            overlay
        )

        return combined.convert("RGB")

    except Exception:

        return original_image.convert(
            "RGB"
        )


# ============================================================
# 12. GEMINI CEVAP OKUMA
# ============================================================

def gemini_cevabini_oku(response):

    try:

        data = response.json()

    except Exception:

        return (
            None,
            f"API yanıtı JSON olarak okunamadı. "
            f"HTTP kodu: {response.status_code}"
        )

    if response.status_code != 200:

        error_data = data.get(
            "error",
            {}
        )

        message = error_data.get(
            "message",
            "Bilinmeyen Gemini API hatası."
        )

        status = error_data.get(
            "status",
            ""
        )

        return (
            None,
            f"{message} {status}"
        )

    candidates = data.get(
        "candidates",
        []
    )

    if not candidates:

        return (
            None,
            "Gemini herhangi bir cevap üretmedi."
        )

    candidate = candidates[0]

    content = candidate.get(
        "content",
        {}
    )

    parts = content.get(
        "parts",
        []
    )

    texts = []

    for part in parts:

        if isinstance(part, dict):

            text = part.get(
                "text"
            )

            if text:

                texts.append(
                    text
                )

    if not texts:

        finish_reason = candidate.get(
            "finishReason",
            "Bilinmiyor"
        )

        return (
            None,
            f"Model metin üretmedi. "
            f"Finish reason: {finish_reason}"
        )

    return (
        "\n".join(texts).strip(),
        None
    )


# ============================================================
# 13. BASİT PEDAGOJİK GÜVENLİK FİLTRESİ
# ============================================================

YASAKLI_IFADELER = [

    "dijital çeşitlilik",
    "dijital ilişki",
    "sayısal çeşitlilik",

    "önce bunu yap",
    "sonra bunu yap",

    "cevap şudur",
    "cevabın",
    "sonuç şudur",

    "formül şudur",
    "formül:",

    "hızını bul",
    "hızını hesapla",

    "eğimini bul",
    "eğimini hesapla",

    "denklemini yaz",

    "doğrusal ilişkidir",
    "doğrusal bir ilişkidir"
]


def basit_pedagojik_kontrol(metin):

    if not metin:

        return False, [
            "BOS_CEVAP"
        ]

    metin_kucuk = metin.lower()

    bulunanlar = []

    for ifade in YASAKLI_IFADELER:

        if ifade in metin_kucuk:

            bulunanlar.append(
                ifade
            )

    soru_sayisi = metin.count("?")

    if soru_sayisi > 1:

        bulunanlar.append(
            "BIRDEN_FAZLA_SORU"
        )

    # Çok uzun cevaplar için basit kontrol.
    kelime_sayisi = len(
        metin.split()
    )

    if kelime_sayisi > 90:

        bulunanlar.append(
            "FAZLA_UZUN_CEVAP"
        )

    return (
        len(bulunanlar) == 0,
        bulunanlar
    )


# ============================================================
# 14. KISA PEDAGOJİK DENETLEYİCİ PROMPTU
# ============================================================
def gemini_sor(
    student_message,
    step,
    chat_history,
    image=None
):
    """Tek aşamalı ve sade AI çağrısı.

    Önceki sürümde ikinci bir Gemini denetleyicisi cevabı yeniden yazıyordu.
    Bu, özellikle yeniden üretim sırasında talimatların öğrenci mesajı gibi
    gönderilmesine ve anlamsız cevaplara yol açabiliyordu. Bu sürümde ilk
    olarak yalnızca tek bir model çağrısı yapılır; Python tarafında basit
    kontroller uygulanır. Uygun değilse modelin ürettiği metin öğrenciye
    gösterilmez ve aşamaya uygun güvenli bir soru kullanılır.
    """

    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY alanına API anahtarını yazmalısın.", {}

    system_instruction = sistem_promptu_olustur(step) + f"""

ÇOK ÖNEMLİ:
- Öğrencinin son mesajını aynen anlamaya çalış; öğrencinin yazdığını değiştirme,
  düzeltme veya yeniden yazma.
- Öğrencinin yazmadığı bir ifadeyi ona söylemiş gibi kabul etme.
- Öğrencinin cevabını özetlemek zorunda değilsin.
- Sadece öğrencinin son mesajındaki matematiksel düşünmeye bağlı TEK bir soru sor.
- Cevap verme. İşlem yapma. Çözüm anlatma.
- Yanıtına numaralı liste, madde listesi veya başlık koyma.
- Yanıtın en fazla 2 kısa cümle olsun.
- Sonunda yalnızca bir soru işareti bulunmalı.

MEVCUT AŞAMA: {step}
"""

    contents = []

    # Önceki konuşmayı koruyoruz; ancak modelin eski bot cevaplarını taklit
    # etmesini azaltmak için yalnızca son birkaç mesajı gönderiyoruz.
    recent_history = chat_history[-6:]

    for message in recent_history:
        role = message.get("role")
        if role not in ["user", "assistant"]:
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": str(message.get("content", ""))}]
        })

    current_parts = []

    if image is not None:
        current_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_to_base64(image)
            }
        })

    current_parts.append({
        "text": (
            "ÖĞRENCİNİN SON MESAJI — DEĞİŞTİRME, YORUMU DIŞINDA TEKRARLAMA:\n"
            + str(student_message)
        )
    })

    contents.append({"role": "user", "parts": current_parts})

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.10,
            "topP": 0.70,
            "maxOutputTokens": 100
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    try:
        response = requests.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        answer, error = gemini_cevabini_oku(response)
        if error:
            return None, error, {}

        ok, categories = basit_pedagojik_kontrol(answer)

        # Ek, deterministik biçim kontrolü.
        if ok and answer.count("?") == 1 and len(answer.split()) <= 60:
            return answer.strip(), None, {
                "denemeler": 1,
                "son_kategori": "UYGUN",
                "denetim": [{
                    "deneme": 1,
                    "cevap": answer,
                    "kategori": "UYGUN",
                    "python_kontrol": "UYGUN"
                }]
            }

        # Model cevabı biçimsel olarak uygunsuzsa ikinci bir LLM çağrısı
        # yapmıyoruz. Böylece denetleyici modelin cevabı bozma ihtimalini
        # ortadan kaldırıyoruz.
        fallback = GUVENLI_YEDEK_SORULAR.get(
            step,
            "Bir sonraki adımı kendin belirlemek için hangi bilgiyi kontrol edebilirsin?"
        )

        return fallback, None, {
            "denemeler": 1,
            "son_kategori": categories[0] if categories else "BICIM_KONTROLU",
            "denetim": [{
                "deneme": 1,
                "cevap": answer,
                "kategori": categories,
                "python_kontrol": "UYGUN_DEGIL",
                "fallback": fallback
            }]
        }

    except requests.exceptions.Timeout:
        return None, "Gemini API zaman aşımına uğradı.", {}
    except requests.exceptions.ConnectionError:
        return None, "Gemini API bağlantısı kurulamadı. İnternet bağlantını kontrol et.", {}
    except Exception as e:
        return None, f"Beklenmeyen hata: {str(e)}", {}


GUVENLI_YEDEK_SORULAR = {
    "1. Ayrıştırma": "Problemde verilen bilgilerden hangisini önce belirlemek istersin?",
    "2. Soyutlama": "Problemdeki bilgilerden hangisinin çözüm için önemli olduğunu nasıl anlayabilirsin?",
    "3. Algoritma Tasarımı": "Çözümüne başlamak için atacağın ilk adım ne olabilir?",
    "4. Hata Ayıklama": "Bu adımının problemde verilen koşullarla uyumlu olup olmadığını nasıl kontrol edebilirsin?",
    "5. Metabilişsel Yansıtma": "Bu problemi çözerken düşünme biçimin hakkında ne fark ettin?"
}

# ============================================================
# 24. ARAŞTIRMACI BİLGİSİ
# ============================================================

st.divider()

st.caption(
    "Algoritmik düşünme • Yapay zekâ destekli problem çözme • "
    "Tasarım tabanlı araştırma • V36 Sade"
)
