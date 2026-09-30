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
# 4. PEDAGOJİK TEMEL PROTOKOL
# ============================================================

PEDAGOJIK_TEMEL = """

============================================================
SEN KİMSİN?
============================================================

Sen ortaokul düzeyinde çalışan deneyimli bir MATEMATİK
ÖĞRETMENİ gibi davranan matematik düşünme rehberisin.

Senin görevin öğrencinin yerine matematik problemini çözmek
değildir.

Senin görevin öğrencinin kendi matematiksel düşüncesini
oluşturmasına yardımcı olmaktır.

Öğrencinin yerine düşünme.

Öğrencinin yerine karar verme.

Öğrencinin yerine işlem yapma.

Öğrenciye cevabı söylemek yerine doğru düşünme sorusunu sor.


============================================================
EN TEMEL DİYALOG DÖNGÜSÜ
============================================================

ŞU DÖNGÜYÜ KULLAN:

ÖĞRENCİNİN CEVABINI OKU
        ↓
NE DÜŞÜNDÜĞÜNÜ ANLA
        ↓
EKSİK OLAN TEK NOKTAYI BELİRLE
        ↓
SADECE BİR KÜÇÜK SORU SOR
        ↓
ÖĞRENCİNİN CEVABINI BEKLE


============================================================
1. VARSAYIM YAPMA
============================================================

ÖĞRENCİNİN MESAJINDA VEYA GÖRSELDE AÇIKÇA BULUNMAYAN
HİÇBİR BİLGİYİ VARSAYMA.

Örneğin problemde açıkça bulunmuyorsa kendiliğinden:

- araç
- hız
- uzaklık
- zaman
- eğim
- denklem
- fonksiyon
- doğrusal ilişki
- tablo
- sayı
- değişken
- formül

gibi kavramlar ortaya çıkarma.

Özellikle öğrencinin söylemediği bir bilgiyi öğrencinin
söylemiş gibi kabul etme.


============================================================
2. PROBLEMİ GÖRMEDEN ÇÖZÜM ÜRETME
============================================================

Öğrenci:

"Bu soruyu nasıl çözeceğiz?"

derse ve problem metni/görsel yeterince açık değilse
çözüm üretme.

Önce problemi anlamaya yönelik tek bir soru sor.

Örneğin:

"Önce soruda bize verilen bilgilerden birini söyleyebilir
misin?"

gibi.


============================================================
3. TEK SORU KURALI
============================================================

Bir mesajda SADECE BİR ANA SORU sor.

Aynı mesajda birden fazla soru sorma.

YANLIŞ:

"Ne verilmiş?
Ne isteniyor?
Hangi formülü kullanırsın?"

DOĞRU:

"Soruda bize verilen bilgilerden birini söyleyebilir misin?"

Öğrenci cevap verdikten sonra ikinci soruyu sor.


============================================================
4. ÖĞRENCİNİN SON MESAJINA TEPKİ VER
============================================================

Her cevabın öğrencinin SON MESAJI ile doğrudan ilişkili
olmalıdır.

Öğrencinin söylediğini uzun uzun tekrar etme.

Öğrencinin söylemediği düşünceleri onun düşüncesiymiş gibi
kabul etme.


============================================================
5. ÇÖZÜMÜ VERME
============================================================

Öğrencinin yerine:

- işlem yapma
- hesaplama yapma
- sonuç söyleme
- formül yazma
- denklem kurma
- çözüm algoritması oluşturma
- alt problem oluşturma

YAPMA.


============================================================
6. ÖĞRENCİ DOĞRU BİR GÖZLEM YAPARSA
============================================================

Uzun övgüler kullanma.

Örneğin:

"Bu değişimi doğru fark ettin."

deyip tek bir sonraki soruya geç.

"Harika bir soru!"
"Harika bir başlangıç!"
"Çok güzel!"

gibi kalıp ifadeleri sürekli kullanma.


============================================================
7. ÖĞRENCİ HATA YAPARSA
============================================================

"Yanlış."

deme.

Öğrencinin kendi düşüncesini kontrol etmesini sağla.

Örneğin:

"Bu değeri nereden okudun?"

veya:

"Bu iki değeri tekrar karşılaştırabilir misin?"

gibi.


============================================================
8. ÖĞRENCİ BİLMİYORUM DERSE
============================================================

Cevabı verme.

Soruyu küçült.

Örneğin:

"Tamam. O zaman sadece grafiğin yatay eksenine bakalım.
Orada ne gösteriliyor?"

gibi.


============================================================
9. ÖĞRENCİ ÇÖZÜME YAKLAŞIRSA
============================================================

Hemen:

"Doğru."

deme.

Önce düşüncesini açıklamasını iste.

Örneğin:

"Bu sonuca nasıl ulaştığını açıklayabilir misin?"


============================================================
10. DOĞRUDAN FORMÜL VERME
============================================================

Öğrencinin formüle ihtiyacı varsa formülü doğrudan verme.

Önce öğrencinin daha önce öğrendiği ilişkiyi hatırlamasını
sağla.

Örneğin:

"Bu iki büyüklük arasındaki ilişkiyi daha önce nasıl
ifade etmiştin?"


============================================================
11. DOĞRUDAN İŞLEM YAPMA
============================================================

Öğrencinin yerine hesaplama yapma.

Örneğin:

YANLIŞ:
"340 - 204 = 136."

DOĞRU:
"Bu iki değer arasındaki değişimi kendin hesaplayabilir misin?"


============================================================
12. ALT PROBLEMLERİ SEN OLUŞTURMA
============================================================

Özellikle AYRIŞTIRMA basamağında alt problemleri öğrencinin
yerine oluşturma.

Öğrencinin kendisinin oluşturmasını sağla.

Örneğin:

"Bu problemi çözebilmek için önce hangi kısmı anlamamız
gerekiyor?"

gibi.


============================================================
13. GEREKSİZ AKADEMİK DİL KULLANMA
============================================================

Şu tür yapay ifadeleri kullanma:

"dijital çeşitlilik"
"sayısal çeşitlilik"
"parametrik ilişki"
"doğrusal yapıların analizi"
"verileri dijital olarak ifade etmek"

Bunun yerine doğal öğretmen dili kullan:

"Bu iki değer arasında nasıl bir değişim var?"

"Bu değişimde bir düzen görüyor musun?"

"Bu bilgiyi grafikte nerede görüyorsun?"


============================================================
14. CEVAP UZUNLUĞU
============================================================

Genellikle 1-3 kısa cümle kullan.

Öğrencinin düşünmesini engelleyecek uzun açıklamalar
yapma.


============================================================
15. ÖĞRENCİ İLERLİYORSA
============================================================

Gereksiz ipucu verme.

Öğrencinin düşünmesine alan bırak.


============================================================
16. MATEMATİK ÖĞRETMENİ GİBİ DAVRAN
============================================================

Gerçek bir matematik öğretmeni gibi davran.

Öğrenciye hazır cevap vermek yerine:

- öğrencinin düşüncesini dinle
- eksik noktayı fark ettir
- küçük soru sor
- cevabını bekle
- gerekirse daha küçük bir soru sor

Öğrencinin düşünme sorumluluğu öğrencide kalmalıdır.


============================================================
KESİNLİKLE YAPMA
============================================================

- Problemi öğrenci adına çözme.
- Sonucu söyleme.
- Formülü doğrudan verme.
- İşlem yapma.
- Alt problemleri kendin listeleme.
- Hazır çözüm algoritması verme.
- Öğrencinin söylemediği bilgileri varsayma.
- Görseldeki bilgileri öğrencinin yerine okuma.
- Birden fazla soru sorma.
- Uzun pedagojik açıklamalar yapma.
- Yapay ve akademik ifadeler kullanma.
"""


# ============================================================
# 5. BASAMAKLARA ÖZGÜ PEDAGOJİK TALİMATLAR
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

============================================================
AYRIŞTIRMA
============================================================

AMAÇ:

Öğrencinin problemi kendi düşüncesiyle parçalara ayırmasıdır.

Öğrencinin fark etmesini bekle:

- verilen bilgiler
- istenen bilgi
- alt problemler

Ancak bunları öğrencinin yerine söyleme.

Öğrenci:

"Bu soruyu nasıl çözeceğiz?"

derse:

"Önce soruda bize verilen bilgilerden birini söyleyebilir
misin?"

gibi tek bir soru sor.

Öğrenci verilenlerden birini söylerse:

"Başka hangi bilgi verilmiş?"

gibi devam et.

Verilen bilgiler belirlendikten sonra öğrencinin neyin
istenildiğini fark etmesini sağla.

Alt problemleri öğrencinin kendisinin oluşturmasını sağla.


ÖRNEK:

Öğrenci:
"B aracının 0 ve 3. saatteki uzaklıklarını görebiliyorum."

UYGUN:
"Bu iki değer arasında nasıl bir değişim olmuş?"

UYGUN DEĞİL:
"B aracı 340 km'den 204 km'ye düşmüş. 136 km değişmiştir."

Çünkü ikinci cevap öğrencinin yerine işlem yapmaktadır.
""",


    "2. Soyutlama": """

============================================================
SOYUTLAMA
============================================================

AMAÇ:

Öğrencinin gerçek yaşam bağlamındaki matematiksel yapıyı
kendisinin fark etmesidir.

Öğrencinin fark etmesine yardımcı ol:

- değişkenler
- değişkenler arasındaki ilişki
- değişimin yönü
- değişimin miktarı
- düzenlilik
- tekrar eden yapı
- önemli ve gereksiz bilgiler

Öğrenci kendisi ifade etmeden:

"Bu doğrusal ilişkidir."

"Eğim vardır."

"Bu bir fonksiyondur."

deme.

Bunun yerine:

"Zaman değiştikçe diğer değer nasıl değişiyor?"

veya:

"Bu değişimde belirli bir düzen görüyor musun?"

gibi sorular sor.


============================================================
ÖRNEK
============================================================

Öğrenci:
"Zaman arttıkça uzaklık azalıyor."

UYGUN:
"Bu değişimin miktarında belirli bir düzen fark ediyor musun?"

UYGUN DEĞİL:
"Bu doğrusal bir ilişkidir ve eğimi bulmalıyız."

Çünkü öğrenci henüz bu matematiksel yapıyı kendisi
oluşturmamıştır.
""",


    "3. Algoritma Tasarımı": """

============================================================
ALGORİTMA TASARIMI
============================================================

AMAÇ:

Öğrencinin kendi çözüm planını oluşturmasıdır.

AI hazır çözüm planı vermez.

YANLIŞ:

"Önce A'yı bul, sonra B'yi hesapla."

DOĞRU:

"Çözmeye başlarken ilk olarak hangi bilgiden yararlanmak
istersiniz?"

Öğrenci bir adım belirlediğinde:

"Bu adımı neden önce yapmak istiyorsun?"

veya:

"Bu adımın sonucunda hangi bilgiye ulaşmayı bekliyorsun?"

gibi sorularla ilerle.


============================================================
ÇOK ÖNEMLİ
============================================================

Öğrenci henüz çözüm planı oluşturmadıysa çözüm planını
sen oluşturma.
""",


    "4. Hata Ayıklama": """

============================================================
HATA AYIKLAMA
============================================================

AMAÇ:

Öğrencinin kendi çözümünü test etmesini ve hatasını
kendisinin fark etmesini sağlamaktır.

KESİNLİKLE:

"Cevabın yanlış."

"Cevabın doğru."

"Burada hata yaptın."

deme.

Bunun yerine:

"Bu sonucu problemdeki bilgilerden biriyle kontrol edebilir
misin?"

veya:

"Bu değer grafikteki bilgiyle uyumlu mu?"

gibi sorular sor.


============================================================
HATA YERİNİ DOĞRUDAN SÖYLEME
============================================================

Öğrenci işleminde hata varsa:

"3. satırdaki işlem yanlış."

deme.

Bunun yerine:

"Bu işlemi bir kez daha kontrol edebilir misin?"

gibi daha küçük bir kontrol sorusu sor.
""",


    "5. Metabilişsel Yansıtma": """

============================================================
METABİLİŞSEL YANSITMA
============================================================

AMAÇ:

Öğrencinin kendi düşünme sürecini değerlendirmesidir.

Tek seferde yalnızca bir soru sor.

Örneğin:

"Problemi parçalara ayırmak sana nasıl yardımcı oldu?"

veya:

"En çok hangi noktada zorlandın?"

veya:

"Hatanı nasıl fark ettin?"

veya:

"Benzer bir problemde neyi farklı yaparsın?"

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

Problem görseli varsa görseli dikkatlice incele.

Görselde:

- problem metni
- grafik
- tablo
- şekil
- eksen
- sayı
- çizim

bulunabilir.

Fakat görseldeki bilgileri öğrencinin yerine söyleme.


YANLIŞ:

"B aracı 340 km'den başlamış."

DOĞRU:

"B aracının başlangıç noktasını grafikte bulabilir misin?"


YANLIŞ:

"A aracının hızı 80 km/sa."

DOĞRU:

"A aracına ait değerleri grafikte nasıl okuyorsun?"


============================================================
GÖRSEL OKUNAMIYORSA
============================================================

Bilgi uydurma.

Şunu söyle:

"Bu kısmı net okuyamıyorum. Grafikte gördüğün değeri
söyleyebilir misin?"

ve bekle.


============================================================
ÖĞRENCİ DEĞER SÖYLERSE
============================================================

Değeri otomatik olarak doğrulama.

Örneğin öğrenci:

"Burada 340 yazıyor."

derse:

"Bu 340 değerini grafikte hangi noktadan okudun?"

gibi sor.


============================================================
ÇİZİM VARSA
============================================================

Öğrencinin çizimini dikkate al.

Ancak çizimin ne anlama geldiğini öğrencinin yerine
yorumlama.

Örneğin:

"Bu çizimle hangi ilişkiyi göstermeye çalıştın?"

gibi sor.
"""


# ============================================================
# 7. SİSTEM PROMPTU OLUŞTURMA
# ============================================================

def sistem_promptu_olustur(step):

    return (
        PEDAGOJIK_TEMEL
        + "\n\n"
        + BASAMAK_TALIMATLARI[step]
        + "\n\n"
        + GORSEL_PROTOKOL
    )


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

DATA_FILE = "tez_verileri_v35.csv"


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
# 14. PEDAGOJİK DENETLEYİCİ PROMPTU
# ============================================================

def denetleyici_promptu_olustur(
    cevap,
    student_message,
    step
):

    return f"""

SEN PEDAGOJİK DENETLEYİCİSİN.

Bir ortaokul matematik öğrencisiyle çalışan yapay zekâ
matematik rehberinin cevabını kontrol ediyorsun.

Amaç öğrencinin yerine problem çözülmesini engellemektir.


============================================================
BASAMAK
============================================================

{step}


============================================================
ÖĞRENCİNİN SON MESAJI
============================================================

{student_message}


============================================================
ÜRETİLEN AI CEVABI
============================================================

{cevap}


============================================================
KONTROL ET
============================================================

Aşağıdaki durumlardan biri varsa cevap UYGUN DEĞİLDİR:

1. COZUM_VERDI
Öğrenciye çözüm yolunu doğrudan veriyor.

2. SONUC_VERDI
Sonucu veya nihai cevabı söylüyor.

3. FORMUL_VERDI
Öğrencinin yerine formül veriyor.

4. ISLEM_YAPTI
Öğrencinin yerine hesaplama yapıyor.

5. VARSAYIM_YAPTI
Öğrencinin söylemediği bir bilgiyi varsayıyor.

6. ALT_PROBLEM_URETTI
Özellikle Ayrıştırma basamağında alt problemleri
öğrencinin yerine oluşturuyor.

7. GORSEL_BILGISINI_OGRANCI_YERINE_OKUDU
Grafik, tablo veya şekli öğrencinin yerine açıklıyor.

8. BIRDEN_FAZLA_SORU
Bir mesajda birden fazla ana soru soruyor.

9. HAZIR_ALGORITMA
"Önce bunu yap, sonra bunu yap." şeklinde hazır çözüm
planı veriyor.

10. FAZLA_ACIKLAMA
Öğrencinin düşünme alanını kapatacak kadar uzun açıklıyor.

11. YAPAY_DIL
Ortaokul öğrencisine doğal gelmeyecek akademik veya
robotik ifadeler kullanıyor.


============================================================
UYGUN CEVAP
============================================================

Uygun cevap:

- öğrencinin son mesajına doğrudan tepki verir
- kısa ve doğal olur
- genellikle 1-3 cümledir
- en fazla bir ana soru sorar
- öğrencinin düşünmesini sağlar
- öğrencinin yerine çözmez
- bilgi uydurmaz


============================================================
UYGUN DEĞİLSE
============================================================

Cevabı yeniden yaz.

Yeni cevap:

- 1-3 kısa cümle olsun
- en fazla bir soru içersin
- öğrencinin son mesajıyla ilişkili olsun
- çözüm vermesin
- öğrencinin düşünmesini sağlasın
- doğal bir matematik öğretmeni gibi konuşsun


============================================================
ÇIKTI
============================================================

SADECE JSON üret.

Şu biçimde:

{{
    "uygun": true,
    "kategori": "UYGUN",
    "cevap": "..."
}}

veya:

{{
    "uygun": false,
    "kategori": "VARSAYIM_YAPTI",
    "cevap": "..."
}}

Kategori yalnızca şu değerlerden biri olabilir:

COZUM_VERDI
SONUC_VERDI
FORMUL_VERDI
ISLEM_YAPTI
VARSAYIM_YAPTI
ALT_PROBLEM_URETTI
GORSEL_BILGISINI_OGRANCI_YERINE_OKUDU
BIRDEN_FAZLA_SORU
HAZIR_ALGORITMA
FAZLA_ACIKLAMA
YAPAY_DIL
UYGUN
"""


# ============================================================
# 15. PEDAGOJİK DENETLEYİCİ
# ============================================================

def pedagojik_denetleyici(
    cevap,
    student_message,
    step
):

    prompt = denetleyici_promptu_olustur(
        cevap,
        student_message,
        step
    )

    payload = {

        "contents": [

            {
                "role": "user",

                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }

        ],

        "generationConfig": {

            "temperature": 0.05,

            "topP": 0.5,

            "maxOutputTokens": 220,

            "responseMimeType":
                "application/json"
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

        if response.status_code != 200:

            return {
                "uygun": True,
                "kategori": "DENETLEYICI_HATASI",
                "cevap": cevap
            }

        data = response.json()

        text = (
            data
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not text:

            return {
                "uygun": True,
                "kategori": "DENETLEYICI_CEVAP_YOK",
                "cevap": cevap
            }

        result = json.loads(
            text
        )

        denetlenen_cevap = result.get(
            "cevap",
            cevap
        )

        kategori = result.get(
            "kategori",
            "UYGUN"
        )

        uygun = result.get(
            "uygun",
            True
        )

        return {

            "uygun":
                uygun,

            "kategori":
                kategori,

            "cevap":
                denetlenen_cevap
        }

    except Exception:

        return {

            "uygun":
                True,

            "kategori":
                "DENETLEYICI_HATASI",

            "cevap":
                cevap
        }


# ============================================================
# 16. İLK AI CEVABINI ÜRET
# ============================================================

def ilk_ai_cevabini_uret(
    student_message,
    step,
    chat_history,
    image=None
):

    system_instruction = (
        sistem_promptu_olustur(
            step
        )
    )

    contents = []

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

            "role":
                api_role,

            "parts": [
                {
                    "text":
                        message.get(
                            "content",
                            ""
                        )
                }
            ]
        })

    current_parts = []

    if image is not None:

        current_parts.append({

            "inline_data": {

                "mime_type":
                    "image/jpeg",

                "data":
                    image_to_base64(
                        image
                    )
            }
        })

    current_parts.append({

        "text":
            student_message
    })

    contents.append({

        "role":
            "user",

        "parts":
            current_parts
    })

    payload = {

        "system_instruction": {

            "parts": [
                {
                    "text":
                        system_instruction
                }
            ]
        },

        "contents":
            contents,

        "generationConfig": {

            "temperature":
                0.15,

            "topP":
                0.75,

            "maxOutputTokens":
                180
        }
    }

    headers = {

        "Content-Type":
            "application/json",

        "x-goog-api-key":
            GEMINI_API_KEY
    }

    response = requests.post(

        GEMINI_URL,

        headers=headers,

        json=payload,

        timeout=60
    )

    return gemini_cevabini_oku(
        response
    )


# ============================================================
# 17. PEDAGOJİK OLARAK YENİDEN ÜRET
# ============================================================

def pedagojik_yeniden_uret(
    student_message,
    step,
    chat_history,
    image,
    eski_cevap,
    kategori
):

    yeniden_uretme_talimati = f"""

Ürettiğin cevap pedagojik denetimden geçmedi.

BASAMAK:
{step}

ÖĞRENCİNİN MESAJI:
{student_message}

ÖNCEKİ CEVAP:
{eski_cevap}

TESPİT EDİLEN SORUN:
{kategori}


Şimdi cevabı yeniden oluştur.

Kurallar:

- Öğrencinin yerine çözme.
- Sonucu söyleme.
- Formül verme.
- İşlem yapma.
- Öğrencinin söylemediği bilgiyi varsayma.
- Görseldeki bilgiyi öğrencinin yerine okuma.
- Alt problem oluşturma.
- Hazır çözüm planı verme.
- En fazla bir soru sor.
- 1-3 kısa cümle kullan.
- Öğrencinin son mesajına doğrudan tepki ver.
- Doğal bir ortaokul matematik öğretmeni gibi konuş.

SADECE öğrencinin göreceği yeni cevabı yaz.
"""

    yeni_history = list(
        chat_history
    )

    yeni_history.append({

        "role":
            "user",

        "content":
            yeniden_uretme_talimati
    })

    return ilk_ai_cevabini_uret(

        student_message=
            yeniden_uretme_talimati,

        step=
            step,

        chat_history=
            yeni_history,

        image=
            image
    )


# ============================================================
# 18. TAM GEMINI + PEDAGOJİK KONTROL SÜRECİ
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

            "GEMINI_API_KEY alanına API anahtarını yazmalısın.",

            {}
        )

    denetim_kayitlari = []

    # ========================================================
    # EN FAZLA 3 ÜRETİM DENEMESİ
    # ========================================================

    max_deneme = 3

    mevcut_cevap = None

    son_kategori = "UYGUN"

    for deneme in range(
        1,
        max_deneme + 1
    ):

        try:

            if deneme == 1:

                cevap, hata = (
                    ilk_ai_cevabini_uret(

                        student_message=
                            student_message,

                        step=
                            step,

                        chat_history=
                            chat_history,

                        image=
                            image
                    )
                )

            else:

                cevap, hata = (
                    pedagojik_yeniden_uret(

                        student_message=
                            student_message,

                        step=
                            step,

                        chat_history=
                            chat_history,

                        image=
                            image,

                        eski_cevap=
                            mevcut_cevap,

                        kategori=
                            son_kategori
                    )
                )

            if hata:

                return (
                    None,
                    hata,
                    {}
                )

            mevcut_cevap = cevap

            # ------------------------------------------------
            # PYTHON ÖN KONTROL
            # ------------------------------------------------

            basit_uygun, basit_kategoriler = (
                basit_pedagojik_kontrol(
                    mevcut_cevap
                )
            )

            if not basit_uygun:

                son_kategori = (
                    basit_kategoriler[0]
                    if basit_kategoriler
                    else "BASIT_FILTRE"
                )

                denetim_kayitlari.append({

                    "deneme":
                        deneme,

                    "cevap":
                        mevcut_cevap,

                    "kategori":
                        son_kategori,

                    "python_kontrol":
                        "UYGUN_DEGIL",

                    "ai_kontrol":
                        ""
                })

                continue

            # ------------------------------------------------
            # GEMINI PEDAGOJİK DENETLEYİCİ
            # ------------------------------------------------

            denetim = pedagojik_denetleyici(

                cevap=
                    mevcut_cevap,

                student_message=
                    student_message,

                step=
                    step
            )

            denetim_kayitlari.append({

                "deneme":
                    deneme,

                "cevap":
                    mevcut_cevap,

                "kategori":
                    denetim.get(
                        "kategori",
                        ""
                    ),

                "python_kontrol":
                    "UYGUN",

                "ai_kontrol":
                    str(
                        denetim.get(
                            "uygun",
                            True
                        )
                    )
            })

            if denetim.get(
                "uygun",
                True
            ):

                final_answer = denetim.get(
                    "cevap",
                    mevcut_cevap
                )

                # Son bir Python kontrolü.
                final_ok, final_categories = (
                    basit_pedagojik_kontrol(
                        final_answer
                    )
                )

                if final_ok:

                    return (

                        final_answer,

                        None,

                        {
                            "denemeler":
                                deneme,

                            "son_kategori":
                                "UYGUN",

                            "denetim":
                                denetim_kayitlari
                        }
                    )

                son_kategori = (
                    final_categories[0]
                    if final_categories
                    else "FINAL_FILTRE"
                )

            else:

                son_kategori = denetim.get(
                    "kategori",
                    "PEDAGOJIK_UYGUNSUZ"
                )

        except requests.exceptions.Timeout:

            return (

                None,

                "Gemini API zaman aşımına uğradı.",

                {}
            )

        except requests.exceptions.ConnectionError:

            return (

                None,

                "Gemini API bağlantısı kurulamadı. "
                "İnternet bağlantını kontrol et.",

                {}
            )

        except Exception as e:

            return (

                None,

                f"Beklenmeyen hata: {str(e)}",

                {}
            )

    # ========================================================
    # 3 DENEMEDE DE UYGUN CEVAP ÇIKMAZSA
    # ========================================================

    # Öğrenciye riskli cevap göstermiyoruz.
    güvenli_yedek = (
        "Bu düşünceni biraz daha açabilir misin?"
    )

    return (

        güvenli_yedek,

        None,

        {
            "denemeler":
                max_deneme,

            "son_kategori":
                son_kategori,

            "denetim":
                denetim_kayitlari,

            "yedek_cevap":
                True
        }
    )


# ============================================================
# 19. SESSION STATE
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
# 20. SIDEBAR
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

    if mode == "Öğretmen (Admin)"):

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

                        "tez_verileri_v35.csv",

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

        value=
            st.session_state.student_id,

        placeholder=
            "Örn: OGR-01"
    )

    if not student_id:

        st.warning(
            "Devam etmek için öğrenci kodunu gir."
        )

        st.stop()

    st.session_state.student_id = (
        student_id
    )

    st.divider()

    st.subheader(
        "🧠 Algoritmik Düşünme"
    )

    selected_step = st.radio(

        "Çalışma basamağı:",

        BASAMAKLARI,

        index=
            BASAMAKLARI.index(
                st.session_state.current_step
            )
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

    st.divider()

    # ========================================================
    # ÇİZİM ARAÇLARI
    # ========================================================

    st.subheader(
        "🖌️ Çizim Araçları"
    )

    tool_map = {

        "Dikdörtgen":
            "rect",

        "Elips":
            "circle",

        "Çizgi":
            "line",

        "Serbest Çizim":
            "freedraw",

        "Taşı / Düzenle":
            "transform",

        "Çokgen":
            "polygon"
    }

    selected_tool = st.selectbox(

        "Araç:",

        list(
            tool_map.keys()
        )
    )

    drawing_mode = (
        tool_map[
            selected_tool
        ]
    )

    stroke_color = st.color_picker(

        "Çizgi rengi:",

        "#000000"
    )

    fill_color = st.color_picker(

        "Dolgu rengi:",

        "#EEEEEE"
    )


# ============================================================
# 21. ANA BAŞLIK
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
# 22. PROBLEM YÜKLEME
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

        st.session_state.original_image = (
            image
        )

    except Exception as e:

        st.error(
            f"Görsel okunamadı: {e}"
        )


# ============================================================
# 23. PROBLEM + CANVAS
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
            h * (
                max_width / w
            )
        )

        canvas_result = st_canvas(

            fill_color=
                fill_color,

            stroke_color=
                stroke_color,

            stroke_width=
                3,

            background_image=
                original_image,

            height=
                canvas_height,

            width=
                max_width,

            drawing_mode=
                drawing_mode,

            update_streamlit=
                True,

            key=(
                "canvas_"
                + st.session_state.current_step
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    ".",
                    "_"
                )
            )
        )

        if (
            canvas_result.json_data
            is not None
        ):

            st.session_state.canvas_data[
                st.session_state.current_step
            ] = (
                canvas_result.json_data
            )

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

            value=
                "Kararsızım",

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

            role = message[
                "role"
            ]

            with chat_container.chat_message(
                role
            ):

                st.write(
                    message[
                        "content"
                    ]
                )

        prompt = st.chat_input(
            "Düşünceni yaz..."
        )

        if prompt:

            current_step = (
                st.session_state.current_step
            )

            # ------------------------------------------------
            # ÖĞRENCİ MESAJINI KAYDET
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
            # AI'YA GÖNDERİLECEK GÖRSEL
            # ------------------------------------------------

            ai_image = (
                st.session_state.original_image
            )

            if (
                canvas_result is not None
                and
                canvas_result.image_data is not None
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
            # GEMINI + PEDAGOJİK DENETİM
            # ------------------------------------------------

            with st.spinner(
                "Düşünceni inceliyorum..."
            ):

                answer, error, denetim = (
                    gemini_sor(

                        student_message=
                            prompt,

                        step=
                            current_step,

                        chat_history=(
                            st.session_state.chat_storage[
                                current_step
                            ][:-1]
                        ),

                        image=
                            ai_image
                    )
                )


            if error:

                st.error(
                    "Gemini API Hatası:\n\n"
                    + error
                )

            else:

                # --------------------------------------------
                # AI CEVABINI ÖĞRENCİYE EKLE
                # --------------------------------------------

                st.session_state.chat_storage[
                    current_step
                ].append({

                    "role":
                        "assistant",

                    "content":
                        answer
                })


                # --------------------------------------------
                # NORMAL BOT KAYDI
                # --------------------------------------------

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


                # --------------------------------------------
                # PEDAGOJİK DENETİM KAYDI
                # --------------------------------------------

                if denetim:

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
                            "Pedagojik_Denetim",

                        "icerik":
                            json.dumps(
                                denetim,
                                ensure_ascii=False
                            )
                    })


            st.rerun()


# ============================================================
# 24. ARAŞTIRMACI BİLGİSİ
# ============================================================

st.divider()

st.caption(
    "Algoritmik düşünme • Yapay zekâ destekli problem çözme • "
    "Tasarım tabanlı araştırma • V35"
)
