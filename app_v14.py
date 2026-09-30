import streamlit as st
import requests
import pandas as pd
import os
import io
import time
import base64
import json

from PIL import Image
from datetime import datetime
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 1. UYGULAMA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Algoritmik Düşünme Atölyesi",
    layout="wide"
)


# ============================================================
# 2. GEMINI API — STREAMLIT SECRETS
# ============================================================

if "GEMINI_API_KEY" not in st.secrets:

    st.error(
        "Gemini API anahtarı bulunamadı. "
        "Streamlit Cloud > Settings > Secrets bölümünü kontrol edin."
    )

    st.stop()


GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

MODEL_NAME = "gemini-2.5-flash"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL_NAME}:generateContent"
    f"?key={GEMINI_API_KEY}"
)

DATA_FILE = "tez_verileri_final.csv"


# ============================================================
# 3. ALGORİTMİK DÜŞÜNME BASAMAKLARI
# ============================================================

BASAMAKLARI = [
    "1. Ayrıştırma",
    "2. Soyutlama",
    "3. Algoritma Tasarımı",
    "4. Hata Ayıklama"
]


# ============================================================
# 4. ALT BECERİLER
# ============================================================

ALT_BECERILER = {

    "1. Ayrıştırma": {
        "A1": "Alt amaç belirleme",
        "A2": "Problemi parçalara ayırma",
        "A3": "Parçalar arasındaki ilişkiyi kurma"
    },

    "2. Soyutlama": {
        "S1": "İlgili bilgiyi seçme",
        "S2": "Gereksiz bilgiyi ayırt etme",
        "S3": "Matematiksel ilişkileri belirleme",
        "S4": "Uygun matematiksel temsil oluşturma",
        "S5": "Örüntü veya temel yapıyı fark etme"
    },

    "3. Algoritma Tasarımı": {
        "AT1": "Gerekli işlemi belirleme",
        "AT2": "İşlemleri sıralama",
        "AT3": "Ara sonuçları sonraki adımlarda kullanma",
        "AT4": "Çözüm prosedürü oluşturma",
        "AT5": "Yöntemin benzer problemlere uygulanabilirliğini düşünme"
    },

    "4. Hata Ayıklama": {
        "HA1": "Çözümü test etme",
        "HA2": "Sonucu problem koşullarıyla karşılaştırma",
        "HA3": "Tutarsızlığı fark etme",
        "HA4": "Hatanın bulunduğu adımı belirleme",
        "HA5": "Hatalı adımı düzeltme",
        "HA6": "Düzeltilmiş çözümü yeniden test etme"
    }
}


# ============================================================
# 5. BASAMAK TALİMATLARI
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

AMAÇ:

Öğrencinin problemi daha küçük parçalara, alt amaçlara veya
ulaşılması gereken küçük hedeflere ayırmasını sağlamak.

ALT DAVRANIŞLAR:

A1 - Alt amaç belirleme:
Öğrencinin çözümde ulaşılması gereken küçük hedefi kendisinin
belirlemesini sağla.

A2 - Problemi parçalara ayırma:
Öğrencinin problemi daha küçük ve yönetilebilir parçalara
ayırmasını sağla.

A3 - Parçalar arasındaki ilişkiyi kurma:
Öğrencinin belirlediği parçaların birbirleriyle nasıl ilişkili
olduğunu fark etmesini sağla.

ÖNEMLİ:

- Verilenleri öğrencinin yerine listeleme.
- İstenen bilgiyi öğrencinin yerine söyleme.
- Alt problemleri öğrencinin yerine oluşturma.
- Çözüm yolunu verme.
- İşlem yaptırma.
- Sonuca yaklaştıracak hazır bir yol gösterme.

Öğrencinin kendi ayrıştırmasını oluşturmasını sağla.
""",

    "2. Soyutlama": """

AMAÇ:

Öğrencinin problemin çözümü için gerekli bilgileri seçmesini,
gereksiz bilgileri ayırt etmesini ve problemin temel matematiksel
yapısını ortaya çıkarmasını sağlamak.

ALT DAVRANIŞLAR:

S1 - İlgili bilgiyi seçme
S2 - Gereksiz bilgiyi ayırt etme
S3 - Matematiksel ilişkileri belirleme
S4 - Uygun matematiksel temsil oluşturma
S5 - Örüntü veya temel yapıyı fark etme

ÖNEMLİ:

- Problemin hangi matematik konusu olduğunu doğrudan söyleme.
- Örneğin "Bu bir EKOK problemidir." deme.
- Kullanılacak formülü söyleme.
- Kullanılacak işlemi doğrudan söyleme.
- Öğrencinin yerine matematiksel model oluşturma.

Öğrencinin önemli bilgiyi ve matematiksel yapıyı kendisinin
ortaya çıkarmasını sağla.

Örüntüyü zorunlu tutma. Problem uygunsa destekle.
""",

    "3. Algoritma Tasarımı": """

AMAÇ:

Öğrencinin çözüm için gerekli işlemleri belirlemesini,
bunları mantıklı bir sıraya koymasını ve çözüm prosedürünü
kendisinin oluşturmasını sağlamak.

ALT DAVRANIŞLAR:

AT1 - Gerekli işlemi belirleme
AT2 - İşlemleri sıralama
AT3 - Ara sonuçları sonraki adımlarda kullanma
AT4 - Çözüm prosedürü oluşturma
AT5 - Yöntemin benzer problemlere uygulanabilirliğini düşünme

ÖNEMLİ:

- Çözüm adımlarını öğrenci adına verme.
- "Önce bunu yap, sonra bunu yap." şeklinde çözüm sunma.
- İşlem sonucu hesaplama.
- Formül verme.
- Öğrencinin yerine algoritma oluşturma.

Öğrencinin kendi çözüm prosedürünü oluşturmasını sağla.
""",

    "4. Hata Ayıklama": """

AMAÇ:

Öğrencinin kendi çözümünü test etmesini, problem koşullarıyla
karşılaştırmasını, olası tutarsızlığı fark etmesini, hatanın
yerini belirlemesini ve çözümünü düzeltmesini sağlamak.

ALT DAVRANIŞLAR:

HA1 - Çözümü test etme
HA2 - Sonucu problem koşullarıyla karşılaştırma
HA3 - Tutarsızlığı fark etme
HA4 - Hatanın bulunduğu adımı belirleme
HA5 - Hatalı adımı düzeltme
HA6 - Düzeltilmiş çözümü yeniden test etme

ÖNEMLİ:

"Hata yaptın."
"Bu işlem yanlış."
"Doğru cevap şu."
gibi ifadeler kullanma.

Hatanın yerini öğrencinin kendisinin bulmasını sağla.
Düzeltmeyi öğrencinin kendisinin yapmasını sağla.
"""
}


# ============================================================
# 6. METABİLİŞSEL YANSITMA
# ============================================================

METABILISSEL_SORULAR = {

    "1. Ayrıştırma":
        "Problemi parçalara ayırırken hangi küçük hedefi belirlediğini ve neden onu önce ele aldığını düşün.",

    "2. Soyutlama":
        "Problemde hangi bilgilerin ve ilişkilerin çözüm için önemli olduğunu nasıl belirlediğini düşün.",

    "3. Algoritma Tasarımı":
        "Çözüm adımlarını oluştururken işlemleri hangi düşünceyle sıraladığını düşün.",

    "4. Hata Ayıklama":
        "Çözümünü kontrol ederken sonucunun güvenilir olduğuna nasıl karar verdiğini düşün."
}


# ============================================================
# 7. ANA SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

ROLÜN:

Ortaokul matematik öğrencisinin algoritmik düşünme becerilerini
destekleyen pedagojik bir rehber yapay zekâsın.

TEMEL AMAÇ:

Öğrencinin matematik problemini kendi düşünmesi, çözümünü
kendisinin oluşturması ve kendi çözümünü kontrol etmesidir.

SENİN GÖREVİN:

Öğrenciye çözümü anlatmak değil, öğrencinin düşünmesini
sağlayan kısa ve hedeflenmiş sorular sormaktır.

KESİN KURALLAR:

1. Öğrencinin yerine problemi çözme.

2. Sonucu söyleme.

3. İşlemi öğrencinin yerine yapma.

4. Formülü öğrencinin yerine yazma.

5. Alt problemi öğrencinin yerine oluşturma.

6. Çözüm algoritmasını öğrencinin yerine oluşturma.

7. Hatanın yerini doğrudan söyleme.

8. "Yanlış", "hata yaptın", "doğru cevap" gibi ifadeler kullanma.

9. Öğrenci cevabını doğrudan tamamlamaya çalışma.

10. Her yanıtta YALNIZCA BİR ANA SORU sor.

11. Bir yanıtta birden fazla soru sorma.

12. Numaralı veya maddeli çözüm adımları verme.

13. Uzun açıklamalar yapma.

14. Öğrenci cevap vermeden sonraki bilişsel adıma geçme.

15. Yanıt 1-3 kısa cümle olsun.

16. Ortaokul öğrencisinin anlayabileceği doğal Türkçe kullan.

17. Öğrencinin söylediği şeyi gereksiz yere tekrar etme.

18. Öğrenci mevcut bilişsel hedefi yeterince gerçekleştirdiyse
aynı davranışı tekrar tekrar sordurma.

19. Öğrenci cevabı eksikse çözümü açıklamak yerine daha dar,
daha anlaşılır ve tek hedefli bir soru sor.

20. Öğrenci doğrudan cevap veya sonuç isterse sonucu verme.
İlgili algoritmik düşünme basamağındaki bilişsel davranışı
tetikleyen tek bir soru sor.

21. Öğrencinin kullandığı matematiksel kavramı kendisi söylemişse
onu kabul edebilirsin; ancak bir sonraki işlemi veya sonucu
sen söyleme.

22. Öğrencinin cevabındaki doğru veya uygun kısmı uzun uzun
övgüyle anlatma. Kısa ve doğal ol.

23. "Harika bir başlangıç noktası!", "Çok güzel düşünmüşsün!"
gibi kalıp övgüler kullanma.

24. Öğrenci bir soru sorarsa, doğrudan öğretim yapmadan,
mevcut basamağın amacına uygun tek bir rehber soru ile
düşünmesini sürdür.

25. Önceki konuşmada öğrencinin zaten verdiği bilgiyi yeniden
istemekten kaçın.

26. Önceki rehber sorusuna öğrencinin verdiği cevabı dikkate al.
Aynı soruyu farklı kelimelerle tekrar etme.

27. Öğrenci bir alt beceriyi gerçekleştirmişse onu tekrar
ölçmeye çalışma; sıradaki eksik bilişsel davranışa geç.

28. Öğrenci mevcut basamağın temel hedeflerini yeterince
gerçekleştirmişse basamağın tamamlandığını bildir.

29. Basamak tamamlanmadıysa yalnızca mevcut basamakta kal.

30. Bir sonraki basamağın çözümünü veya işlemini önceden
öğrenciye verme.

TEMEL İLKE:

Sen bir çözüm anlatıcısı değilsin.
Sen bir bilişsel rehbersin.

Öğrencinin düşünmesi gereken şeyi öğrencinin yerine düşünme.
"""


# ============================================================
# 8. SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

if "uploaded_mime_type" not in st.session_state:
    st.session_state.uploaded_mime_type = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {
        s: [] for s in BASAMAKLARI
    }

if "current_step" not in st.session_state:
    st.session_state.current_step = BASAMAKLARI[0]

if "stage_ready" not in st.session_state:
    st.session_state.stage_ready = False

if "completed_stages" not in st.session_state:
    st.session_state.completed_stages = []

if "process_finished" not in st.session_state:
    st.session_state.process_finished = False

if "last_ai_result" not in st.session_state:
    st.session_state.last_ai_result = None


# ============================================================
# 9. VERİ KAYIT FONKSİYONU
# ============================================================

def log_kaydet(data):

    try:

        df = pd.DataFrame([data])

        dosya_var = os.path.isfile(DATA_FILE)

        df.to_csv(
            DATA_FILE,
            mode="a",
            index=False,
            header=not dosya_var,
            encoding="utf-8-sig"
        )

    except Exception as e:

        st.error(f"Veri kayıt hatası: {e}")


# ============================================================
# 10. GÜVENLİ AI YANITI KONTROLÜ
# ============================================================

def ai_yaniti_guvenli_mi(text):

    if not text:
        return False

    temiz = text.strip()

    if not temiz:
        return False

    soru_sayisi = temiz.count("?")

    if soru_sayisi != 1:
        return False

    kelime_sayisi = len(temiz.split())

    if kelime_sayisi > 45:
        return False

    yasakli_ifadeler = [
        "doğru cevap",
        "cevap:",
        "cevap ",
        "sonuç ",
        "sonuç:",
        "yanlış yaptın",
        "hata yaptın",
        "şunu yapmalısın",
        "önce bunu yap",
        "sonra bunu yap",
        "şimdi bunu hesapla"
    ]

    kucuk = temiz.lower()

    for ifade in yasakli_ifadeler:

        if ifade in kucuk:
            return False

    return True


# ============================================================
# 11. TEK SORULUK YEDEK SORULAR
# ============================================================

def guvenli_yedek_soru(stage):

    yedek_sorular = {

        "1. Ayrıştırma":
            "Bu problemi çözebilmek için ulaşman gereken ilk küçük hedef sence nedir?",

        "2. Soyutlama":
            "Problemin çözümü için önemli olduğunu düşündüğün bilgiler arasında nasıl bir ilişki var?",

        "3. Algoritma Tasarımı":
            "Çözüm yolunda bundan sonraki adımının ne olması gerektiğini nasıl düşünüyorsun?",

        "4. Hata Ayıklama":
            "Çözümünün problemdeki koşulları sağladığını nasıl kontrol edebilirsin?"
    }

    return yedek_sorular[stage]


# ============================================================
# 12. GERÇEK GEMINI KONUŞMA GEÇMİŞİNİ OLUŞTUR
# ============================================================

def gemini_contents_olustur(history):

    contents = []

    for mesaj in history:

        role = mesaj.get("role")
        content = mesaj.get("content", "").strip()

        if not content:
            continue

        if role == "user":

            contents.append({
                "role": "user",
                "parts": [
                    {
                        "text": content
                    }
                ]
            })

        elif role == "assistant":

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
# 13. GEMINI REHBER FONKSİYONU
# ============================================================

def gemini_rehber_cevabi(
    stage,
    student_message,
    history,
    image_data=None,
    mime_type=None
):

    stage_instruction = BASAMAK_TALIMATLARI[stage]

    alt_beceriler = ALT_BECERILER.get(stage, {})

    alt_beceri_text = "\n".join(
        [
            f"{k}: {v}"
            for k, v in alt_beceriler.items()
        ]
    )


    # --------------------------------------------------------
    # GERÇEK KONUŞMA GEÇMİŞİ
    # --------------------------------------------------------

    contents = gemini_contents_olustur(history)


    # --------------------------------------------------------
    # KONTROL PROMPTU
    # --------------------------------------------------------

    control_prompt = f"""

Şu anda öğrencinin bulunduğu algoritmik düşünme basamağı:

{stage}

BU BASAMAĞIN PEDAGOJİK TALİMATI:

{stage_instruction}

BU BASAMAKTAKİ ALT BECERİLER:

{alt_beceri_text}


ÖĞRENCİNİN SON MESAJI ZATEN KONUŞMA GEÇMİŞİNDE BULUNUYOR.

KONUŞMA GEÇMİŞİNİ DİKKATLİCE İNCELE.

Özellikle son öğrencinin cevabını, hemen önce sorduğun soruyu
ve öğrencinin daha önce verdiği bilgileri birlikte değerlendir.


GÖREV:

1. Öğrencinin son cevabında hangi alt becerinin ortaya çıktığını
belirle.

2. Öğrencinin bu basamağın temel bilişsel hedefini ne ölçüde
gerçekleştirdiğini değerlendir.

3. Basamak henüz tamamlanmadıysa yalnızca eksik veya yeterince
ortaya çıkmamış bilişsel davranışı düşündürecek TEK BİR SORU sor.

4. Öğrencinin zaten gerçekleştirdiği bir davranışı yeniden
sordurma.

5. Önceki sorunu aynı veya benzer ifadelerle tekrar etme.

6. Öğrencinin cevabını açıklayarak tamamlamaya çalışma.

7. Öğrenci sonuç isterse sonucu verme; yalnızca düşündürücü
tek bir soru sor.

8. Basamak gerçekten tamamlandıysa "basamak_tamamlandi": true
yap.

9. Basamak tamamlandıysa soru alanında yeni basamağın çözümünü
başlatacak soru sorma.

YANIT KURALLARI:

- Yalnızca bir ana soru.
- En fazla bir soru işareti.
- 1-3 kısa cümle.
- Doğal Türkçe.
- Çözüm yok.
- Sonuç yok.
- Formül yok.
- Hazır işlem yok.
- Hazır alt problem yok.
- Hazır algoritma yok.
- "Yanlış" yok.
- "Hata yaptın" yok.
- Kalıp övgü yok.
- Öğrencinin zaten söylediği bilgiyi tekrar isteme.

ÇIKTI SADECE JSON OLMALI.
"""


    # --------------------------------------------------------
    # İLK MESAJDA SYSTEM PROMPT + KONTROL PROMPTU
    # --------------------------------------------------------

    if contents:

        contents[0]["parts"].insert(
            0,
            {
                "text": control_prompt
            }
        )

    else:

        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": control_prompt
                    }
                ]
            }
        ]


    # --------------------------------------------------------
    # GÖRSEL EKLE
    # --------------------------------------------------------

    if image_data:

        kullanilacak_mime = mime_type

        if not kullanilacak_mime:
            kullanilacak_mime = "image/jpeg"

        # Görseli son kullanıcı mesajına ekle
        for content_item in reversed(contents):

            if content_item["role"] == "user":

                content_item["parts"].append(
                    {
                        "inline_data": {
                            "mime_type": kullanilacak_mime,
                            "data": image_data
                        }
                    }
                )

                break


    # --------------------------------------------------------
    # API PAYLOAD
    # --------------------------------------------------------

    payload = {

        "system_instruction": {

            "parts": [
                {
                    "text": SYSTEM_PROMPT
                }
            ]
        },

        "contents": contents,

        "generationConfig": {

            "temperature": 0.2,

            "topP": 0.8,

            "responseMimeType": "application/json",

            "responseSchema": {

                "type": "OBJECT",

                "properties": {

                    "alt_beceri": {
                        "type": "STRING"
                    },

                    "basamak_tamamlandi": {
                        "type": "BOOLEAN"
                    },

                    "soru": {
                        "type": "STRING"
                    }

                },

                "required": [
                    "alt_beceri",
                    "basamak_tamamlandi",
                    "soru"
                ]
            }
        }
    }


    headers = {
        "Content-Type": "application/json"
    }


    son_hata = None


    # ========================================================
    # API İSTEĞİ
    # ========================================================

    for deneme in range(3):

        try:

            response = requests.post(
                GEMINI_URL,
                json=payload,
                headers=headers,
                timeout=45
            )

            try:
                result = response.json()

            except Exception:
                result = {}


            # ------------------------------------------------
            # YOĞUNLUK / GEÇİCİ HATA
            # ------------------------------------------------

            if response.status_code in [
                429,
                500,
                502,
                503
            ]:

                son_hata = result

                time.sleep(
                    2 * (deneme + 1)
                )

                continue


            # ------------------------------------------------
            # API HATASI
            # ------------------------------------------------

            if response.status_code != 200:

                return {

                    "alt_beceri":
                        "API_HATASI",

                    "basamak_tamamlandi":
                        False,

                    "soru":
                        guvenli_yedek_soru(stage),

                    "api_hatasi":
                        result
                }


            # ------------------------------------------------
            # CANDIDATES KONTROLÜ
            # ------------------------------------------------

            if "candidates" not in result:

                return {

                    "alt_beceri":
                        "API_HATASI",

                    "basamak_tamamlandi":
                        False,

                    "soru":
                        guvenli_yedek_soru(stage),

                    "api_hatasi":
                        result
                }


            # ------------------------------------------------
            # GEMINI METNİ
            # ------------------------------------------------

            raw_text = (
                result["candidates"][0]
                ["content"]["parts"][0]
                ["text"]
            )


            # ------------------------------------------------
            # JSON ÇÖZ
            # ------------------------------------------------

            try:

                ai_result = json.loads(
                    raw_text
                )

            except Exception:

                return {

                    "alt_beceri":
                        "JSON_HATASI",

                    "basamak_tamamlandi":
                        False,

                    "soru":
                        guvenli_yedek_soru(stage)
                }


            # ------------------------------------------------
            # ALANLAR
            # ------------------------------------------------

            alt_beceri = ai_result.get(
                "alt_beceri",
                "BELİRSİZ"
            )

            tamamlandi = ai_result.get(
                "basamak_tamamlandi",
                False
            )

            soru = ai_result.get(
                "soru",
                ""
            )


            # ------------------------------------------------
            # GÜVENLİK KONTROLÜ
            # ------------------------------------------------

            if not ai_yaniti_guvenli_mi(soru):

                soru = guvenli_yedek_soru(
                    stage
                )

                tamamlandi = False


            return {

                "alt_beceri":
                    alt_beceri,

                "basamak_tamamlandi":
                    bool(tamamlandi),

                "soru":
                    soru
            }


        except Exception as e:

            son_hata = str(e)

            time.sleep(
                2 * (deneme + 1)
            )


    # ========================================================
    # TÜM DENEMELER BAŞARISIZ
    # ========================================================

    return {

        "alt_beceri":
            "API_HATASI",

        "basamak_tamamlandi":
            False,

        "soru":
            guvenli_yedek_soru(stage),

        "api_hatasi":
            son_hata
    }


# ============================================================
# 14. FİNAL SÜREÇ ÖZETİ
# ============================================================

def final_ozet_olustur():

    butun_surec = ""

    for stage in BASAMAKLARI:

        butun_surec += (
            f"\n\n===== {stage} =====\n"
        )

        for mesaj in st.session_state.chat_storage.get(
            stage,
            []
        ):

            role = mesaj.get(
                "role",
                ""
            )

            if role == "user":

                butun_surec += (
                    f"ÖĞRENCİ: "
                    f"{mesaj['content']}\n"
                )

            elif role == "assistant":

                butun_surec += (
                    f"REHBER: "
                    f"{mesaj['content']}\n"
                )


    final_prompt = f"""

Aşağıdaki süreç bir ortaokul öğrencisinin matematiksel problem
çözme sürecidir.

Öğrencinin çözümünü yeniden çözme.

Sadece süreçte öğrencinin sergilediği düşünme davranışlarını
özetle.

Özette özellikle:

- problemi nasıl ayrıştırdığı,
- hangi bilgileri ve ilişkileri fark ettiği,
- çözüm algoritmasını nasıl oluşturduğu,
- çözümünü nasıl kontrol ettiği,
- hata ayıklama sürecinde ne yaptığı

üzerinde dur.

Öğrencinin henüz gerçekleştirmediği bir davranışı
gerçekleştirmiş gibi yazma.

Sonucu veya matematiksel çözümü tekrar açıklama.

Öğrenciye yönelik kısa, doğal ve teşvik edici bir değerlendirme
yaz.

SÜREÇ:

{butun_surec}
"""


    payload = {

        "system_instruction": {

            "parts": [

                {
                    "text": """
Sen ortaokul matematik öğrencisinin problem çözme
sürecini özetleyen pedagojik bir rehbersin.

Çözümü yeniden üretme.
Öğrencinin düşünme sürecini özetle.
Eksik davranışları olmuş gibi gösterme.
"""
                }

            ]
        },

        "contents": [

            {
                "role": "user",

                "parts": [

                    {
                        "text": final_prompt
                    }

                ]
            }

        ],

        "generationConfig": {

            "temperature": 0.3,

            "topP": 0.8
        }
    }


    try:

        response = requests.post(

            GEMINI_URL,

            json=payload,

            headers={
                "Content-Type":
                    "application/json"
            },

            timeout=45
        )


        result = response.json()


        if "candidates" in result:

            return (
                result["candidates"][0]
                ["content"]["parts"][0]
                ["text"]
            )


        return (
            "Süreç özeti oluşturulamadı."
        )


    except Exception as e:

        return (
            f"Süreç özeti oluşturulamadı: {e}"
        )


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

        sifre = st.text_input(
            "Şifre:",
            type="password"
        )


        if sifre == "tez2024":

            st.success(
                "Admin Paneli Aktif"
            )


            if os.path.isfile(DATA_FILE):

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


    # ========================================================
    # ÖĞRENCİ
    # ========================================================

    student_id = st.text_input(

        "Öğrenci No:",

        placeholder="Örn: Hakan"
    )


    if not student_id:

        st.warning(
            "Devam etmek için giriş yapın."
        )

        st.stop()


    st.divider()


    # ========================================================
    # SÜREÇ DURUMU
    # ========================================================

    st.write(
        "### 🧠 Algoritmik Düşünme Süreci"
    )


    for i, stage in enumerate(
        BASAMAKLARI
    ):

        if stage in st.session_state.completed_stages:

            st.success(
                f"✓ {stage}"
            )

        elif stage == st.session_state.current_step:

            st.info(
                f"▶ {stage}"
            )

        else:

            st.write(
                f"○ {stage}"
            )


    st.divider()


    # ========================================================
    # ÇİZİM ARAÇLARI
    # ========================================================

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


    secilen_etiket = st.selectbox(

        "Araç Seçin:",

        list(
            tool_map.keys()
        )
    )


    drawing_mode = tool_map[
        secilen_etiket
    ]


    stroke_color = st.color_picker(

        "Çizgi Rengi:",

        "#000000"
    )


    fill_color = st.color_picker(

        "Kutu Rengi:",

        "#EEEEEE"
    )


# ============================================================
# 16. ANA SAYFA
# ============================================================

st.title(
    "🎯 Algoritmik Problem Çözme Rehberi"
)


st.write(

    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)


# ============================================================
# 17. GÖRSEL YÜKLEME
# ============================================================

uploaded_file = st.file_uploader(

    "Soru Fotoğrafını Yükle",

    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


bg_url = None

canvas_height = 450

display_width = 800


if uploaded_file is not None:

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )


        raw_img = Image.open(
            io.BytesIO(file_bytes)
        )


        w, h = raw_img.size


        if w > 0:

            canvas_height = int(

                h * (
                    display_width / w
                )
            )


        canvas_height = min(
            canvas_height,
            900
        )


        resized_img = raw_img.resize(

            (
                display_width,
                canvas_height
            )
        )


        buffered = io.BytesIO()


        resized_img.save(

            buffered,

            format="PNG"
        )


        img_base64 = (
            base64.b64encode(
                buffered.getvalue()
            ).decode()
        )


        bg_url = (
            "data:image/png;base64,"
            f"{img_base64}"
        )


        st.session_state.uploaded_file_data = (

            base64.b64encode(
                file_bytes
            ).decode()
        )


        st.session_state.uploaded_mime_type = (
            uploaded_file.type
        )


        st.session_state.uploaded_file_name = (
            uploaded_file.name
        )


        st.markdown(

            f"""
            <style>

            iframe[title="streamlit_drawable_canvas.st_canvas"] {{
                background-image: url("{bg_url}") !important;
                background-size: contain !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
            }}

            </style>
            """,

            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"Görsel yükleme hatası: {e}"
        )


# ============================================================
# 18. İKİ SÜTUN
# ============================================================

st.divider()


col1, col2 = st.columns(

    [1.3, 1],

    gap="large"
)


# ============================================================
# 19. SOL PANEL
# ============================================================

with col1:

    st.write(
        "🖼️ **Tasarım ve Planlama Alanı**"
    )


    st.caption(
        "Problem üzerinde çizim yapabilir veya akış şeması oluşturabilirsin."
    )


    canvas_result = st_canvas(

        fill_color=fill_color,

        stroke_color=stroke_color,

        stroke_width=3,

        background_image=None,

        background_color="rgba(0,0,0,0)",

        height=canvas_height,

        width=display_width,

        drawing_mode=drawing_mode,

        update_streamlit=True,

        key=(
            f"canvas_"
            f"{st.session_state.current_step}"
            .replace(
                " ",
                "_"
            )
        )
    )


    # ========================================================
    # ÇİZİM KAYDET
    # ========================================================

    if st.button(
        "🖼️ Tasarımı Kaydet"
    ):

        if canvas_result.json_data:

            log_kaydet({

                "tarih":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "ogrenci_id":
                    student_id,

                "basamak":
                    st.session_state.current_step,

                "alt_beceri":
                    "",

                "basamak_tamamlandi":
                    "",

                "tip":
                    "Cizim",

                "ogrenci_mesaji":
                    "",

                "ai_mesaji":
                    "",

                "icerik":
                    str(
                        canvas_result.json_data
                    )
            })


            st.success(
                "Tasarım kaydedildi."
            )


    # ========================================================
    # METABİLİŞSEL YANSITMA
    # ========================================================

    st.write("---")


    st.subheader(
        "🧠 Metabilişsel Yansıtma"
    )


    st.info(
        METABILISSEL_SORULAR[
            st.session_state.current_step
        ]
    )


    m_cevap = st.text_area(

        "Düşünceni buraya yaz...",

        key=(
            "meta_area_"
            + str(
                BASAMAKLARI.index(
                    st.session_state.current_step
                )
            )
        )
    )


    if st.button(
        "💾 Düşüncemi Kaydet"
    ):

        if m_cevap.strip():

            log_kaydet({

                "tarih":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "ogrenci_id":
                    student_id,

                "basamak":
                    st.session_state.current_step,

                "alt_beceri":
                    "",

                "basamak_tamamlandi":
                    "",

                "tip":
                    "Metabiliş",

                "ogrenci_mesaji":
                    m_cevap,

                "ai_mesaji":
                    "",

                "icerik":
                    m_cevap
            })


            st.success(
                "Düşüncen kaydedildi."
            )


    # ========================================================
    # EMİNLİK
    # ========================================================

    st.write("---")


    st.write(
        "⭐ **Bu adımdaki çözümünden ne kadar eminsin?**"
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
            f"slider_"
            f"{st.session_state.current_step}"
        )
    )


    if st.button(
        "📈 Eminlik Derecesini Kaydet"
    ):

        log_kaydet({

            "tarih":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "ogrenci_id":
                student_id,

            "basamak":
                st.session_state.current_step,

            "alt_beceri":
                "",

            "basamak_tamamlandi":
                "",

            "tip":
                "Eminlik",

            "ogrenci_mesaji":
                "",

            "ai_mesaji":
                "",

            "icerik":
                confidence
        })


        st.success(
            f"Eminlik: {confidence}"
        )


    # ========================================================
    # BASAMAK GEÇİŞİ
    # ========================================================

    st.write("---")


    if st.session_state.stage_ready:

        st.success(
            "Bu basamağın temel düşünme hedefinin "
            "tamamlandığı değerlendirildi."
        )


        current_index = BASAMAKLARI.index(

            st.session_state.current_step
        )


        if current_index < len(BASAMAKLARI) - 1:

            if st.button(

                "➡️ Sonraki Basamağa Geç",

                use_container_width=True
            ):

                current_stage = (
                    st.session_state.current_step
                )


                if current_stage not in \
                        st.session_state.completed_stages:

                    st.session_state.completed_stages.append(
                        current_stage
                    )


                st.session_state.current_step = (

                    BASAMAKLARI[
                        current_index + 1
                    ]
                )


                st.session_state.stage_ready = False

                st.session_state.last_ai_result = None

                st.rerun()


        else:

            st.success(
                "Dört algoritmik düşünme basamağı tamamlandı."
            )


            if st.button(

                "🏁 Çözümü Bitir ve Süreç Özetini Al",

                use_container_width=True
            ):

                with st.spinner(
                    "Süreç analiz ediliyor..."
                ):

                    summary_text = (
                        final_ozet_olustur()
                    )


                st.session_state.process_finished = True


                log_kaydet({

                    "tarih":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "ogrenci_id":
                        student_id,

                    "basamak":
                        "FİNAL",

                    "alt_beceri":
                        "",

                    "basamak_tamamlandi":
                        "",

                    "tip":
                        "Final Özeti",

                    "ogrenci_mesaji":
                        "",

                    "ai_mesaji":
                        summary_text,

                    "icerik":
                        summary_text
                })


                st.info(
                    summary_text
                )


    else:

        st.caption(
            "Basamak geçişi, mevcut basamağın hedefi "
            "yeterince gerçekleştirildikten sonra açılır."
        )


# ============================================================
# 20. SAĞ PANEL - REHBER BOT
# ============================================================

with col2:

    st.write(
        "💬 **Rehber Bot — Algoritmik Düşünme Desteği**"
    )


    st.caption(
        "Bot çözümü söylemez. Düşünmeni sağlayacak tek bir soru sorar."
    )


    # --------------------------------------------------------
    # SOHBET GEÇMİŞİ
    # --------------------------------------------------------

    chat_container = st.container(
        height=550
    )


    with chat_container:

        for m in st.session_state.chat_storage[
            st.session_state.current_step
        ]:

            if m["role"] == "user":

                st.chat_message(
                    "user"
                ).write(
                    m["content"]
                )

            else:

                st.chat_message(
                    "assistant"
                ).write(
                    m["content"]
                )


    # --------------------------------------------------------
    # ÖĞRENCİ MESAJI
    # --------------------------------------------------------

    p = st.chat_input(
        "Düşünceni buraya yaz..."
    )


    if p:

        current_stage = (
            st.session_state.current_step
        )


        # ====================================================
        # ÖĞRENCİ MESAJINI KAYDET
        # ====================================================

        st.session_state.chat_storage[
            current_stage
        ].append({

            "role":
                "user",

            "content":
                p
        })


        log_kaydet({

            "tarih":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "ogrenci_id":
                student_id,

            "basamak":
                current_stage,

            "alt_beceri":
                "",

            "basamak_tamamlandi":
                "",

            "tip":
                "Öğrenci",

            "ogrenci_mesaji":
                p,

            "ai_mesaji":
                "",

            "icerik":
                p
        })


        # ====================================================
        # GEMINI
        # ====================================================

        with st.spinner(
            "Düşünceni inceliyor..."
        ):

            result = gemini_rehber_cevabi(

                stage=current_stage,

                student_message=p,

                history=(
                    st.session_state
                    .chat_storage[
                        current_stage
                    ]
                ),

                image_data=(
                    st.session_state
                    .uploaded_file_data
                ),

                mime_type=(
                    st.session_state
                    .uploaded_mime_type
                )
            )


        # ====================================================
        # AI SONUCU
        # ====================================================

        ans = result.get(

            "soru",

            guvenli_yedek_soru(
                current_stage
            )
        )


        alt_beceri = result.get(

            "alt_beceri",

            "BELİRSİZ"
        )


        basamak_tamamlandi = result.get(

            "basamak_tamamlandi",

            False
        )


        # ====================================================
        # AI MESAJINI SOHBETE EKLE
        # ====================================================

        st.session_state.chat_storage[
            current_stage
        ].append({

            "role":
                "assistant",

            "content":
                ans
        })


        # ====================================================
        # BASAMAK TAMAMLANDI MI?
        # ====================================================

        if basamak_tamamlandi:

            st.session_state.stage_ready = True


        # ====================================================
        # AI KAYDI
        # ====================================================

        log_kaydet({

            "tarih":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "ogrenci_id":
                student_id,

            "basamak":
                current_stage,

            "alt_beceri":
                alt_beceri,

            "basamak_tamamlandi":
                basamak_tamamlandi,

            "tip":
                "Bot",

            "ogrenci_mesaji":
                p,

            "ai_mesaji":
                ans,

            "icerik":
                ans
        })


        # ====================================================
        # EKRANI YENİLE
        # ====================================================

        st.rerun()
