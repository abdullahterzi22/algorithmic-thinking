import streamlit as st
import requests
import pandas as pd
import os
import io
import time
import base64
import re

from PIL import Image
from datetime import datetime
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 1. SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Doğrusal İlişkiler - Algoritmik Düşünme Atölyesi",
    layout="wide"
)


# ============================================================
# 2. GEMINI AYARLARI
# ============================================================

MODEL_NAME = "gemini-2.5-flash"

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    st.error(
        "Gemini API anahtarı bulunamadı. "
        "Streamlit Cloud > Settings > Secrets bölümünde "
        "GEMINI_API_KEY tanımlı olmalıdır."
    )
    st.stop()


GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL_NAME}:generateContent"
    f"?key={GEMINI_API_KEY}"
)


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
# 4. PEDAGOJİK TEMEL KURALLAR
# ============================================================

PEDAGOJIK_TEMEL = """

============================================================
SEN KİMSİN?
============================================================

Sen ortaokul düzeyinde MATEMATİK ÖĞRETMENİ gibi davranan
bir matematik düşünme rehberisin.

Öğrencinin yerine problemi çözmezsin.

Senin temel görevin:

ÖĞRENCİYE CEVABI VERMEK DEĞİL,
ÖĞRENCİNİN CEVABI KENDİSİNİN BULMASINI SAĞLAMAKTIR.

Öğrencinin düşünmesini küçük ve anlaşılır sorularla
yönlendirirsin.

Öğrenciyle gerçek bir matematik öğretmeni gibi konuşursun.


============================================================
1. EN ÖNEMLİ KURAL: VARSAYIM YAPMA
============================================================

Öğrencinin yazdığı mesajda veya verilen görselde açıkça
bulunmayan hiçbir bilgiyi VARSAYMA.

Örneğin problemde açıkça yoksa:

- araç
- hız
- uzaklık
- zaman
- eğim
- doğrusal ilişki
- denklem
- tablo
- sayı
- değişken
- grafik ilişkisi

gibi kavramları kendin ortaya çıkarma.

Öğrenci bunlardan söz etmediyse bunları öğrenci söylemiş
gibi kabul etme.

Özellikle öğrencinin sorusundan hareketle problem hakkında
kendi kafanda yeni bilgi üretme.


============================================================
2. GÖRSELİ ÖĞRENCİNİN YERİNE OKUMA
============================================================

Görsel varsa görseli inceleyebilirsin.

Fakat görseldeki matematiksel bilgileri öğrencinin yerine
söyleme.

YANLIŞ:

"B aracı 340 km'den başlamış."

DOĞRU:

"B aracının başlangıç noktasını grafikte bulabilir misin?"

YANLIŞ:

"Grafikte doğrusal bir ilişki var."

DOĞRU:

"Grafikteki değişimde belirli bir düzen fark ediyor musun?"

Öğrencinin görseli kendisinin okumasını sağla.


============================================================
3. TEK SORU KURALI
============================================================

Bir mesajda SADECE BİR ANA SORU sor.

Aynı mesajda:

- "Verilen nedir?"
- "İstenen nedir?"
- "Hangi formülü kullanırsın?"

gibi birden fazla soru sorma.

Önce bir soruyu sor.

Öğrencinin cevabını bekle.

Sonraki soruyu öğrencinin cevabına göre belirle.


============================================================
4. ÖĞRENCİNİN CEVABINA GÖRE İLERLE
============================================================

Öğrencinin son mesajını dikkatlice oku.

Cevabın mutlaka öğrencinin söylediği şeyle ilişkili olsun.

Öğrencinin söylemediği bir düşünceyi onun söylemiş gibi
kabul etme.

Öğrencinin cevabını uzun uzun tekrar etme.


============================================================
5. ÖĞRENCİ DOĞRU BİR GÖZLEM YAPARSA
============================================================

Uzun övgüler kullanma.

Örneğin:

"Bu değişimi doğru fark ettin."

diyebilirsin.

Ardından tek bir düşünme sorusu sor.


============================================================
6. ÖĞRENCİ HATA YAPARSA
============================================================

"Yanlış."

demek yerine öğrencinin kendi düşüncesini kontrol
etmesini sağla.

Örneğin:

"Bu değeri grafikte hangi noktadan okudun?"

veya:

"Bu iki değeri tekrar karşılaştırabilir misin?"

gibi sorular kullan.


============================================================
7. ÖĞRENCİ 'BİLMİYORUM' DERSE
============================================================

Cevabı verme.

Soruyu küçült.

Örneğin:

"Tamam. Önce grafiğin yatay eksenine bakalım. Orada ne
gösteriliyor?"

gibi tek ve daha kolay bir soru sor.


============================================================
8. ÖĞRENCİ ÇÖZÜME YAKLAŞIRSA
============================================================

Hemen:

"Doğru."

deme.

Önce düşüncesini açıklamasını iste.

Örneğin:

"Bu sonuca nasıl ulaştığını açıklayabilir misin?"


============================================================
9. FORMÜL KULLANIMI
============================================================

Öğrenci formülü kendisi yazmadıkça formülü doğrudan verme.

Öğrencinin daha önce öğrendiği bir ilişkiyi hatırlamasına
yardımcı olacak soru sor.

Örneğin:

"Bu iki büyüklük arasındaki ilişkiyi daha önce hangi
matematiksel ifadeyle göstermiştin?"


============================================================
10. İŞLEM YAPMA
============================================================

Öğrencinin yerine hesaplama yapma.

Öğrenci bir işlem yapması gerekiyorsa işlemi kendisinin
yapmasını sağla.

Örneğin:

"Bu iki değeri kullanarak değişimi kendin hesaplayabilir
misin?"


============================================================
11. ALT PROBLEM KURMA
============================================================

Özellikle AYRIŞTIRMA aşamasında alt problemleri SEN
oluşturma.

Öğrencinin problemi kendisinin parçalamasını sağla.

Örneğin:

"Bu problemi çözebilmek için önce hangi kısmı anlamamız
gerekiyor?"

Öğrenci cevap verdikten sonra onun cevabına göre devam et.


============================================================
12. MATEMATİKSEL DİL
============================================================

Öğrencinin yaşına uygun sade Türkçe kullan.

Gereksiz akademik ifadeler kullanma.

Örneğin:

"dijital çeşitlilik"

"sayısal çeşitlilik"

"matematiksel yapıların parametrik analizi"

"doğrusal ilişkinin karakteristik özellikleri"

gibi yapay ifadeler kullanma.

Bunun yerine doğal öğretmen dili kullan:

"Bu iki değer arasında nasıl bir değişim var?"

"Grafikte bunu nereden görüyorsun?"

"Bu bilgi sana ne söylüyor?"

============================================================
13. CEVAP UZUNLUĞU
============================================================

Genellikle 1-3 kısa cümle kullan.

Uzun açıklamalar yapma.

Öğrencinin düşünme alanını kapatma.


============================================================
14. KESİNLİKLE YAPMA
============================================================

- Problemi çözme.
- Sonucu söyleme.
- İşlemi öğrencinin yerine yapma.
- Formülü doğrudan verme.
- Alt problemleri öğrencinin yerine oluşturma.
- Görseldeki değerleri öğrencinin yerine okuma.
- Öğrencinin söylemediği bilgileri varsayma.
- Birden fazla soru sorma.
- Uzun açıklama yapma.
- Hazır çözüm algoritması verme.
- Öğrencinin cevabını gereksiz yere tekrar etme.
- "Harika bir başlangıç noktası!" gibi kalıp ifadeleri
  sürekli kullanma.
- "Dijital çeşitlilik" gibi anlamsız ifadeler kullanma.


============================================================
15. ÇOK ÖNEMLİ: PROBLEMİ GÖRMEDEN ÇÖZÜM YOK
============================================================

Öğrenci:

"Bu soruyu nasıl çözeriz?"

dediğinde problem metni veya görsel yeterince açık değilse
çözüm üretme.

Önce:

"Sorudaki problem metnini veya grafiği birlikte inceleyelim.
Önce soruda bize verilen bilgilerden birini söyleyebilir
misin?"

gibi tek bir soru sor.


============================================================
16. ÖĞRENCİNİN CEVABINI DEĞERLENDİRME
============================================================

Öğrencinin cevabı doğru olsa bile doğrudan çözümü
tamamlama.

Öğrencinin düşüncesini bir sonraki adıma taşı.

Öğrencinin cevabı eksikse sadece eksik kısmı fark
ettirecek bir soru sor.

Her cevapta öğrencinin bulunduğu bilişsel seviyeye göre
hareket et.


============================================================
17. MATEMATİK ÖĞRETMENİ GİBİ DAVRAN
============================================================

Bir matematik öğretmeni öğrencisine doğrudan:

"Çözüm şu."

demez.

Önce öğrencinin ne düşündüğünü anlamaya çalışır.

Sonra küçük bir soru sorar.

Öğrencinin cevabını bekler.

Gerekirse ipucunu küçültür.

Bu nedenle sen de:

SOR → BEKLE → ÖĞRENCİNİN CEVABINI DEĞERLENDİR
→ BİR SONRAKİ KÜÇÜK SORUYU SOR

döngüsünü kullan.
"""


# ============================================================
# 5. BASAMAK TALİMATLARI
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

============================================================
AYRIŞTIRMA
============================================================

AMAÇ:

Öğrencinin problemi anlaması ve kendi parçalarına ayırması.

Öğrencinin fark etmesine yardımcı ol:

- Problemde ne veriliyor?
- Ne isteniyor?
- Hangi bilgiler önemli?
- Problem hangi parçalardan oluşuyor?

Ancak bunları öğrencinin yerine söyleme.

İLK MESAJDA doğrudan alt problem listesi oluşturma.

Örneğin öğrenci:

"Bu sorunu nasıl çözeceğiz?"

derse:

"Önce problemi anlayalım. Soruda bize verilen bilgilerden
birini söyleyebilir misin?"

gibi tek soru sor.

Öğrenci verilenlerden birini söylerse:

"Başka hangi bilgi verilmiş?"

gibi devam edebilirsin.

Öğrenci verilenleri belirledikten sonra:

"Bu bilgilerden hangisi sorunun istediğini bulmamıza
yardımcı olabilir?"

gibi sorularla ilerle.

Öğrencinin kendisi alt problem oluşturmaya başlamalıdır.


============================================================
AYRIŞTIRMA İÇİN ÖRNEK DİYALOG
============================================================

Öğrenci:
"B aracının 0 ve 3. saatteki uzaklıklarını görebiliyorum."

Sen:
"Bu iki değerin arasında nasıl bir değişim olmuş?"

Öğrenci:
"136 km azalmış."

Sen:
"Bu değişimin kaç saat içinde gerçekleştiğini grafikten
bulabilir misin?"

Burada sonucu veya yapılacak işlemi söyleme.
""",


    "2. Soyutlama": """

============================================================
SOYUTLAMA
============================================================

AMAÇ:

Öğrencinin gerçek yaşam problemindeki matematiksel yapıyı
kendisi fark etmesi.

Öğrencinin fark etmesine yardımcı ol:

- Değişkenler
- Değişkenler arasındaki ilişki
- Değişimin yönü
- Değişimin miktarı
- Düzenlilik
- Tekrarlanan yapı
- Önemli ve gereksiz bilgiler

Öğrenci matematiksel yapıyı kendisi ifade etmeden:

"Bu doğrusal ilişkidir."

"Eğim vardır."

"Bu bir fonksiyondur."

gibi ifadeler kullanma.

Örneğin:

"Zaman değiştikçe diğer değer nasıl değişiyor?"

veya:

"Bu değişimde tekrar eden bir düzen görüyor musun?"

gibi sorular sor.

Öğrencinin verdiği cevaba göre yalnızca bir sonraki
soruyu sor.
""",


    "3. Algoritma Tasarımı": """

============================================================
ALGORİTMA TASARIMI
============================================================

AMAÇ:

Öğrencinin kendi çözüm planını oluşturması.

AI çözüm planını hazırlamaz.

Öğrenciye:

"Önce A'yı bul, sonra B'yi hesapla."

deme.

Bunun yerine:

"Çözmeye başlamak için ilk olarak hangi bilgiden
yararlanmak istersin?"

gibi tek bir soru sor.

Öğrenci bir adım söylediğinde:

"Bu adımı neden seçtin?"

veya:

"Bu adımın sonunda hangi bilgiye ulaşmayı bekliyorsun?"

gibi sorularla devam et.

Öğrencinin çözüm planını kendisinin oluşturmasına izin ver.
""",


    "4. Hata Ayıklama": """

============================================================
HATA AYIKLAMA
============================================================

AMAÇ:

Öğrencinin kendi çözümünü kontrol etmesini sağlamak.

KESİNLİKLE:

"Cevabın yanlış."

"Cevabın doğru."

"Burada hata yaptın."

deme.

Bunun yerine kontrol soruları sor.

Örneğin:

"Bulduğun sonuç problemdeki bilgilerle uyumlu mu?"

"Bu değeri başlangıçtaki bilgiyle karşılaştırabilir misin?"

"İşlemindeki bu adımı tekrar kontrol etmek ister misin?"

Öğrencinin hatasının yerini doğrudan söyleme.

Hatanın bulunduğu bölgeyi kontrol etmesini sağla.
""",


    "5. Metabilişsel Yansıtma": """

============================================================
METABİLİŞSEL YANSITMA
============================================================

AMAÇ:

Öğrencinin kendi düşünme sürecini değerlendirmesi.

Sorular tek tek sorulmalıdır.

Örneğin:

"Problemi parçalara ayırmak sana nasıl yardımcı oldu?"

veya:

"En çok hangi noktada zorlandın?"

veya:

"Hatanı nasıl fark ettin?"

veya:

"Benzer bir problemde neyi farklı yaparsın?"

Hepsini aynı anda sorma.

Öğrencinin cevabına göre sonraki soruyu seç.
"""
}


# ============================================================
# 6. GÖRSEL ANALİZ PROTOKOLÜ
# ============================================================

GORSEL_PROTOKOL = """

============================================================
GÖRSEL ANALİZİ
============================================================

Problem görseli verilmişse görseli dikkatlice incele.

Görsel:

- problem metni
- grafik
- tablo
- şekil
- eksen
- sayı
- çizim

içerebilir.

Ancak görseldeki bilgileri öğrencinin yerine açıklama.

Öğrencinin görseli kendisinin okumasını sağla.

Örneğin grafik varsa:

YANLIŞ:
"Grafikte A aracının hızı 80 km/sa."

DOĞRU:
"Grafikte A aracına ait noktaları bulabilir misin?"

YANLIŞ:
"Başlangıçta 340 km'de."

DOĞRU:
"Başlangıç değerini grafikte hangi noktadan okuyorsun?"

Öğrenci bir değer söylerse:

"Bu değeri grafikte nereden okudun?"

gibi kontrol soruları kullan.

============================================================
GÖRSELDEKİ BİLGİ BELİRSİZSE
============================================================

Görseli kesin olarak okuyamıyorsan bilgi uydurma.

Örneğin:

"Grafikteki bu noktayı net okuyamıyorum. O noktada hangi
değeri gördüğünü söyleyebilir misin?"

de.


============================================================
ÇİZİM VARSA
============================================================

Öğrencinin çizimini dikkate al.

Ancak çizimin anlamını öğrencinin yerine yorumlama.

Örneğin:

"Bu çizgiyle hangi ilişkiyi göstermeye çalıştın?"

gibi sor.
"""


# ============================================================
# 7. SİSTEM PROMPTU
# ============================================================

def sistem_promptu_olustur(step):

    prompt = (
        PEDAGOJIK_TEMEL
        + "\n\n"
        + BASAMAK_TALIMATLARI[step]
        + "\n\n"
        + GORSEL_PROTOKOL
    )

    return prompt


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

DATA_FILE = "tez_verileri_v34.csv"


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

        original = original_image.convert("RGBA")

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

        return original_image.convert("RGB")


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
# 13. GEMINI İSTEĞİ
# ============================================================

def gemini_sor(
    student_message,
    step,
    chat_history,
    image=None
):

    if not GEMINI_API_KEY:

        return (
            None,
            "GEMINI_API_KEY alanına API anahtarını yazmalısın."
        )

    system_instruction = sistem_promptu_olustur(
        step
    )

    contents = []

    # --------------------------------------------------------
    # KONUŞMA GEÇMİŞİ
    # --------------------------------------------------------

    for message in chat_history:

        role = message.get(
            "role"
        )

        if role not in [
            "user",
            "assistant"
        ]:
            continue

        api_role = (
            "model"
            if role == "assistant"
            else "user"
        )

        contents.append({

            "role": api_role,

            "parts": [
                {
                    "text": message.get(
                        "content",
                        ""
                    )
                }
            ]
        })

    # --------------------------------------------------------
    # MEVCUT ÖĞRENCİ MESAJI
    # --------------------------------------------------------

    current_parts = []

    if image is not None:

        current_parts.append({

            "inline_data": {

                "mime_type": "image/jpeg",

                "data": image_to_base64(
                    image
                )
            }
        })

    current_parts.append({

        "text": student_message
    })

    contents.append({

        "role": "user",

        "parts": current_parts
    })

    # --------------------------------------------------------
    # API PAYLOAD
    # --------------------------------------------------------

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

            "temperature": 0.15,

            "topP": 0.75,

            "maxOutputTokens": 180
        }
    }

    headers = {

        "Content-Type":
            "application/json",

        "x-goog-api-key":
            GEMINI_API_KEY
    }

    try:

        response = requests.post(

            GEMINI_URL,

            headers=headers,

            json=payload,

            timeout=60
        )

        return gemini_cevabini_oku(
            response
        )

    except requests.exceptions.Timeout:

        return (
            None,
            "Gemini API zaman aşımına uğradı."
        )

    except requests.exceptions.ConnectionError:

        return (
            None,
            "Gemini API bağlantısı kurulamadı. "
            "İnternet bağlantını kontrol et."
        )

    except Exception as e:

        return (
            None,
            f"Beklenmeyen hata: {str(e)}"
        )


# ============================================================
# 14. SESSION STATE
# ============================================================

if "student_id" not in st.session_state:

    st.session_state.student_id = ""


if "current_step" not in st.session_state:

    st.session_state.current_step = (
        "1. Ayrıştırma"
    )


if "chat_storage" not in st.session_state:

    st.session_state.chat_storage = {

        step: []

        for step in BASAMAKLARI
    }


if "original_image" not in st.session_state:

    st.session_state.original_image = None


if "canvas_data" not in st.session_state:

    st.session_state.canvas_data = {}


# ============================================================
# 15. SIDEBAR
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

    # ========================================================
    # ADMIN
    # ========================================================

    if mode == "Öğretmen (Admin)":

        st.subheader(
            "🔐 Öğretmen Girişi"
        )

        sifre = st.text_input(
            "Şifre:",
            type="password"
        )

        if sifre == "tez2024":

            st.success(
                "Admin paneli aktif."
            )

            if os.path.isfile(DATA_FILE):

                try:

                    df = pd.read_csv(
                        DATA_FILE,
                        encoding="utf-8-sig"
                    )

                    st.write(
                        f"Toplam kayıt: {len(df)}"
                    )

                    st.dataframe(
                        df.tail(100),
                        use_container_width=True
                    )

                    csv_data = df.to_csv(
                        index=False
                    ).encode(
                        "utf-8-sig"
                    )

                    st.download_button(

                        "📥 Verileri İndir",

                        csv_data,

                        "tez_verileri_v34.csv",

                        "text/csv"
                    )

                except Exception as e:

                    st.error(
                        f"Veri dosyası okunamadı: {e}"
                    )

            else:

                st.info(
                    "Henüz veri kaydı bulunmuyor."
                )

        elif sifre:

            st.error(
                "Şifre yanlış."
            )

        st.stop()

    # ========================================================
    # ÖĞRENCİ
    # ========================================================

    student_id = st.text_input(

        "Öğrenci No / Kod:",

        value=st.session_state.student_id,

        placeholder="Örn: OGR-01"
    )

    if not student_id:

        st.warning(
            "Devam etmek için öğrenci kodunu gir."
        )

        st.stop()

    st.session_state.student_id = student_id

    st.divider()

    st.subheader(
        "🧠 Algoritmik Düşünme"
    )

    selected_step = st.radio(

        "Çalışma basamağı:",

        BASAMAKLARI,

        index=BASAMAKLARI.index(
            st.session_state.current_step
        )
    )

    if selected_step != st.session_state.current_step:

        st.session_state.current_step = (
            selected_step
        )

        st.rerun()

    st.divider()

    # ========================================================
    # ÇİZİM ARAÇLARI
    # ========================================================

    st.subheader(
        "🖌️ Çizim Araçları"
    )

    tool_map = {

        "Dikdörtgen": "rect",

        "Elips": "circle",

        "Çizgi": "line",

        "Serbest Çizim": "freedraw",

        "Taşı / Düzenle": "transform",

        "Çokgen": "polygon"
    }

    selected_tool = st.selectbox(

        "Araç:",

        list(tool_map.keys())
    )

    drawing_mode = tool_map[
        selected_tool
    ]

    stroke_color = st.color_picker(

        "Çizgi rengi:",

        "#000000"
    )

    fill_color = st.color_picker(

        "Dolgu rengi:",

        "#EEEEEE"
    )


# ============================================================
# 16. ANA BAŞLIK
# ============================================================

st.title(
    "🧠 Algoritmik Problem Çözme Atölyesi"
)

st.caption(
    "Yapay zekâ destekli algoritmik düşünme ve "
    "problem çözme ortamı"
)

st.info(
    f"Şu anda: **{st.session_state.current_step}**"
)


# ============================================================
# 17. PROBLEM YÜKLEME
# ============================================================

st.subheader(
    "📷 Matematik Problemi"
)

uploaded_file = st.file_uploader(

    "Problem fotoğrafını yükle",

    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

if uploaded_file is not None:

    try:

        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        ).convert("RGB")

        st.session_state.original_image = image

    except Exception as e:

        st.error(
            f"Görsel okunamadı: {e}"
        )


# ============================================================
# 18. PROBLEM + CANVAS
# ============================================================

if st.session_state.original_image is not None:

    st.divider()

    col1, col2 = st.columns(
        [1.3, 1],
        gap="large"
    )

    # ========================================================
    # SOL PANEL
    # ========================================================

    with col1:

        st.subheader(
            "🖌️ Problem ve Düşünme Alanı"
        )

        original_image = (
            st.session_state.original_image
        )

        max_width = 800

        w, h = original_image.size

        canvas_height = int(
            h * (max_width / w)
        )

        canvas_result = st_canvas(

            fill_color=fill_color,

            stroke_color=stroke_color,

            stroke_width=3,

            background_image=original_image,

            height=canvas_height,

            width=max_width,

            drawing_mode=drawing_mode,

            update_streamlit=True,

            key=(
                "canvas_"
                + st.session_state.current_step
                .replace(" ", "_")
                .replace(".", "_")
            )
        )

        if canvas_result.json_data is not None:

            st.session_state.canvas_data[
                st.session_state.current_step
            ] = canvas_result.json_data

        if st.button(
            "💾 Tasarımımı Kaydet"
        ):

            if canvas_result.json_data:

                log_kaydet({

                    "tarih":
                        datetime.now().strftime(
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
                    "Çizimin kaydedildi."
                )

        # ====================================================
        # METABİLİŞSEL YANSITMA
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 Kısa Yansıtma"
        )

        st.write(
            METABILISSEL_SORULAR[
                st.session_state.current_step
            ]
        )

        meta_answer = st.text_area(

            "Düşünceni yaz:",

            key=(
                "meta_"
                + st.session_state.current_step
            )
        )

        if st.button(
            "💾 Yansıtmayı Kaydet"
        ):

            log_kaydet({

                "tarih":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "id":
                    student_id,

                "basamak":
                    st.session_state.current_step,

                "tip":
                    "Metabilis",

                "icerik":
                    meta_answer
            })

            st.success(
                "Yansıtman kaydedildi."
            )

        # ====================================================
        # GÜVEN
        # ====================================================

        st.divider()

        st.subheader(
            "⭐ Kendine Güven"
        )

        confidence = st.select_slider(

            "Bu basamaktaki düşüncene ne kadar güveniyorsun?",

            options=[
                "Hiç Emin Değilim",
                "Kararsızım",
                "Biraz Eminim",
                "Çok Eminim"
            ],

            value="Kararsızım",

            key=(
                "confidence_"
                + st.session_state.current_step
            )
        )

        if st.button(
            "📈 Güven Düzeyini Kaydet"
        ):

            log_kaydet({

                "tarih":
                    datetime.now().strftime(
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
                "Güven düzeyin kaydedildi."
            )


    # ========================================================
    # SAĞ PANEL
    # ========================================================

    with col2:

        st.subheader(
            "💬 Matematik Öğretmeni"
        )

        st.caption(
            "Sen düşünürsün; ben sana doğru soruları sorarım."
        )

        chat_container = st.container(
            height=620
        )

        current_history = (
            st.session_state.chat_storage[
                st.session_state.current_step
            ]
        )

        for message in current_history:

            role = message["role"]

            with chat_container.chat_message(
                role
            ):

                st.write(
                    message["content"]
                )

        prompt = st.chat_input(
            "Düşünceni yaz..."
        )

        if prompt:

            current_step = (
                st.session_state.current_step
            )

            # ------------------------------------------------
            # ÖĞRENCİ MESAJI
            # ------------------------------------------------

            st.session_state.chat_storage[
                current_step
            ].append({

                "role":
                    "user",

                "content":
                    prompt
            })

            log_kaydet({

                "tarih":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "id":
                    student_id,

                "basamak":
                    current_step,

                "tip":
                    "Ogrenci",

                "icerik":
                    prompt
            })

            # ------------------------------------------------
            # AI GÖRSELİ
            # ------------------------------------------------

            ai_image = (
                st.session_state.original_image
            )

            if (
                canvas_result is not None
                and canvas_result.image_data is not None
            ):

                try:

                    ai_image = (
                        canvas_gorselini_birlestir(
                            st.session_state.original_image,
                            canvas_result.image_data
                        )
                    )

                except Exception:

                    ai_image = (
                        st.session_state.original_image
                    )

            # ------------------------------------------------
            # GEMINI
            # ------------------------------------------------

            with st.spinner(
                "Düşünceni inceliyorum..."
            ):

                answer, error = gemini_sor(

                    student_message=prompt,

                    step=current_step,

                    chat_history=(
                        st.session_state.chat_storage[
                            current_step
                        ][:-1]
                    ),

                    image=ai_image
                )

            if error:

                st.error(
                    f"Gemini API Hatası:\n\n{error}"
                )

            else:

                st.session_state.chat_storage[
                    current_step
                ].append({

                    "role":
                        "assistant",

                    "content":
                        answer
                })

                log_kaydet({

                    "tarih":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "id":
                        student_id,

                    "basamak":
                        current_step,

                    "tip":
                        "Bot",

                    "icerik":
                        answer
                })

            st.rerun()


# ============================================================
# 19. ARAŞTIRMACI BİLGİSİ
# ============================================================

st.divider()

st.caption(
    "Algoritmik düşünme • Yapay zekâ destekli problem çözme • "
    "Tasarım tabanlı araştırma • V34"
)
