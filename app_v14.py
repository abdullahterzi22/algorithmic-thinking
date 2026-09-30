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
    "4. Hata Ayıklama"
]


# ============================================================
# 4. DOĞRUSAL İLİŞKİLERE ÖZGÜ ALT BECERİLER
# ============================================================

ALT_BECERILER = {

    "1. Ayrıştırma": [
        ("A1", "Problemin amacını ve istenen durumu belirleme"),
        ("A2", "Problemi anlamlı parçalara ayırma"),
        ("A3", "Değişkenleri ve verilen bilgileri ayırt etme"),
        ("A4", "Parçalar arasındaki ilişkiyi kurma")
    ],

    "2. Soyutlama": [
        ("S1", "İlişkiyi açıklamak için gerekli bilgileri seçme"),
        ("S2", "Gereksiz bilgileri ayırt etme"),
        ("S3", "Değişkenler arasındaki matematiksel ilişkiyi fark etme"),
        ("S4", "Bağımsız ve bağımlı değişkeni belirleme"),
        ("S5", "Uygun matematiksel temsil oluşturma"),
        ("S6", "Değişim miktarını ve başlangıç değerini fark etme"),
        ("S7", "Örüntü veya doğrusal yapıyı fark etme")
    ],

    "3. Algoritma Tasarımı": [
        ("AT1", "Çözüm için gerekli bilgileri kullanma"),
        ("AT2", "İşlemleri uygun sıraya koyma"),
        ("AT3", "Bir değişkendeki değişimi diğer değişkenle ilişkilendirme"),
        ("AT4", "Çözüm prosedürü oluşturma"),
        ("AT5", "Oluşturduğu yöntemi benzer doğrusal ilişkilere aktarma")
    ],

    "4. Hata Ayıklama": [
        ("HA1", "Çözümü veya modeli test etme"),
        ("HA2", "Tablo, grafik ve matematiksel ilişkiyi karşılaştırma"),
        ("HA3", "Sonucu problem koşullarıyla karşılaştırma"),
        ("HA4", "Tutarsızlığı fark etme"),
        ("HA5", "Hatanın bulunduğu adımı belirleme"),
        ("HA6", "Hatalı adımı düzeltme"),
        ("HA7", "Düzeltilmiş çözümü yeniden test etme")
    ]
}


# ============================================================
# 5. BASAMAK TALİMATLARI
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

AMAÇ:
Öğrencinin doğrusal ilişki içeren problemi çözmeden önce
problemi kendi düşüncesiyle anlamlı parçalara ayırmasını sağlamak.

Öğrencinin kendisinin:

- problemin neyi istediğini fark etmesini,
- verilen bilgileri ayırt etmesini,
- değişkenleri fark etmesini,
- problemi anlamlı parçalara ayırmasını,
- parçalar arasındaki ilişkiyi kurmasını

sağla.

Öğrenciye hazır bir parça listesi verme.

Örneğin:
"Önce x'i bul, sonra y'yi bul."
gibi yönlendirmeler yapma.

Öğrenci bir bilgiyi zaten söylediyse tekrar sorma.

Öğrencinin son cevabından hareket ederek bir sonraki düşünme adımını hedefle.
""",

    "2. Soyutlama": """

AMAÇ:
Öğrencinin doğrusal ilişkinin temel matematiksel yapısını
kendisinin fark etmesini sağlamak.

Öğrencinin:

- gerekli bilgileri seçmesini,
- gereksiz bilgileri ayırt etmesini,
- değişkenleri belirlemesini,
- değişkenler arasındaki ilişkiyi fark etmesini,
- bağımsız ve bağımlı değişkenleri düşünmesini,
- tablo, grafik veya cebirsel ifade gibi uygun temsili fark etmesini,
- başlangıç değerini ve değişim miktarını fark etmesini,
- örüntü veya doğrusal yapıyı önceki bilgileriyle ilişkilendirmesini

sağla.

Öğrenci söylemeden "doğrusal ilişki",
"doğrunun eğimi", "y = ax + b" gibi kavramları doğrudan verme.

Öğrencinin yerine matematiksel modeli kurma.
""",

    "3. Algoritma Tasarımı": """

AMAÇ:
Öğrencinin doğrusal ilişkiyi çözmek veya temsil etmek için
kendi işlem ve düşünme sırasını oluşturmasını sağlamak.

Öğrencinin:

- hangi bilgiden yararlanacağını,
- hangi işlemleri yapacağını,
- işlemleri hangi sırayla yapacağını,
- bir adımdaki sonucu sonraki adımda nasıl kullanacağını,
- çözüm prosedürünü

kendisi oluşturmasını sağla.

"Önce bunu hesapla, sonra şunu yap."
şeklinde hazır çözüm planı verme.

Öğrencinin kendi planını oluşturmasını sağlayacak
tek bir soru sor.
""",

    "4. Hata Ayıklama": """

AMAÇ:
Öğrencinin doğrusal ilişkiye yönelik oluşturduğu çözümü,
tablosunu, grafiğini veya matematiksel ifadesini
kendisinin test etmesini sağlamak.

Öğrencinin:

- çözümünü test etmesini,
- tablo ve grafik arasında tutarlılık aramasını,
- matematiksel ifadeyi problem koşullarıyla karşılaştırmasını,
- tutarsızlığı fark etmesini,
- hatalı adımı belirlemesini,
- düzeltmesini,
- düzeltilmiş çözümü tekrar test etmesini

sağla.

"Yanlış yaptın." deme.

Doğru sonucu söyleme.

Hatanın bulunduğu yeri söyleme.

Sadece öğrencinin kendi kontrolünü yapmasını sağlayacak
tek bir soru sor.
"""
}


# ============================================================
# 6. SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

ROLÜN:
Ortaokul matematik dersinde öğrencinin problem çözme sürecine
rehberlik eden pedagojik bir yapay zekâsın.

SEN ÇÖZÜCÜ DEĞİLSİN.

Öğrencinin yerine düşünme.
Öğrencinin yerine işlem yapma.
Öğrenciye cevabı verme.

Görevin öğrencinin düşünmesini sağlayacak kısa,
tek adımlı sorular sormaktır.


==================================================
KESİN DİYALOG KURALLARI
==================================================

1. HER YANITINDA YALNIZCA BİR ANA SORU SOR.

2. Öğrenci cevap vermeden yeni bir düşünme adımına geçme.

3. Önceki konuşmanın tamamını dikkate al.

4. Öğrencinin daha önce verdiği bilgiyi tekrar sorma.

5. Önceki sorunu tekrar sorma.

6. Öğrencinin SON CEVABINDAN hareket et.

7. Öğrenci bir düşünme adımını tamamladıysa
   aynı beceriyi farklı cümlelerle tekrar sordurma.

8. Öğrenci kendiliğinden bir sonraki düşünme adımına
   geçtiyse onu geriye döndürme.

9. Hazır alt problem listesi verme.

10. Öğrenci adına çözüm planı oluşturma.

11. İşlem sonucu verme.

12. Formül verme.

13. Öğrenci söylemeden matematiksel kavramı doğrudan söyleme.

14. "Doğru", "yanlış", "cevap budur", "sonuç budur"
    gibi ifadeler kullanma.

15. "Harika", "çok güzel", "tebrikler" gibi
    otomatik övgüler kullanma.

16. "Hadi birlikte çözelim" gibi ifadeler kullanma.

17. Yanıt mümkün olduğunca 1-3 kısa cümle olsun.

18. Yanıtta yalnızca BİR soru işareti bulunabilir.

19. Liste halinde soru sorma.

20. Bir önceki AI sorusunu kelimeleri değiştirerek
    tekrar sorma.


==================================================
DOĞRUSAL İLİŞKİLER
==================================================

Problem doğrusal ilişkilerle ilgili olabilir.

Ancak öğrencinin henüz fark etmediği bir kavramı
doğrudan söyleme.

Öğrenci değişkenleri kendisi fark etsin.

Öğrenci iki değişken arasındaki ilişkiyi kendisi fark etsin.

Öğrenci uygun matematiksel gösterimi kendisi oluştursun.

Öğrencinin yerine grafik, tablo veya cebirsel ifade oluşturma.


==================================================
ÖĞRENCİ CEVABINI DEĞERLENDİRME
==================================================

Öğrencinin son cevabını önceki konuşmayla birlikte değerlendir.

Eğer öğrenci sorulan düşünme adımını henüz göstermediyse,
aynı hedefi biraz daha anlaşılır hale getiren tek bir soru sor.

Eğer öğrenci hedeflenen düşünme adımını gösterdiyse,
aynı şeyi tekrar sorma.

Bir sonraki eksik düşünme adımına geç.

Eğer öğrenci başka bir alt beceriyi kendiliğinden gösterdiyse,
onu görmezden gelme.

Öğrencinin mevcut düşüncesinden devam et.

Eğer öğrenci "bilmiyorum" derse,
cevabı verme.

Daha küçük ve düşündürücü tek bir soru sor.

Eğer öğrenci doğrudan sonucu söylerse,
sonucu onaylama.

Kendisinin kontrol etmesini sağlayan tek bir soru sor.


==================================================
EN ÖNEMLİ KURAL
==================================================

HER TURDA ÖĞRENCİNİN SON SÖYLEDİĞİ ŞEYE CEVAP VER.

KONUŞMAYI İLK TURDAYMIŞ GİBİ BAŞLATMA.

Örneğin öğrenci:

"Grafikte yatay eksende zaman,
dikey eksende ücret var."

dediyse:

"Soruda hangi bilgiler veriliyor?"

sorusunu tekrar sorma.

Bunun yerine öğrencinin verdiği bu bilgiden
hareket ederek bir sonraki düşünme adımını hedefle.


==================================================
YANIT BİÇİMİ
==================================================

Yalnızca öğrenciye gönderilecek kısa mesajı üret.

Açıklama yapma.

Başlık oluşturma.

Liste oluşturma.

Alt beceri kodu yazma.

Birden fazla soru sorma.
"""


# ============================================================
# 7. METABİLİŞSEL SORULAR
# ============================================================

METABILISSEL_SORULAR = {

    "1. Ayrıştırma":
        "Problemi parçalara ayırırken hangi bilgileri birlikte düşünmenin önemli olduğunu nasıl fark ettin?",

    "2. Soyutlama":
        "Değişkenler arasındaki temel ilişkiyi fark ederken hangi bilgilerin önemli olduğuna nasıl karar verdin?",

    "3. Algoritma Tasarımı":
        "Çözüm adımlarının sırasına nasıl karar verdin?",

    "4. Hata Ayıklama":
        "Çözümünü kontrol ederken modelinin problem koşullarına uyduğundan nasıl emin oldun?"
}


# ============================================================
# 8. GÜVENLİK
# ============================================================

YASAK_IFADELER = [
    "doğru cevap",
    "cevap:",
    "cevap ",
    "sonuç:",
    "sonuç ",
    "yanlış yaptın",
    "yanlış cevap",
    "şunu yap",
    "önce bunu yap",
    "sonra bunu yap",
    "işlemi yap",
    "formülü kullan",
    "çözüm şu",
    "cevap şudur"
]


def ai_yaniti_guvenli_mi(metin):

    if not metin:
        return False

    metin = metin.strip()

    if len(metin.split()) > 55:
        return False

    if metin.count("?") != 1:
        return False

    kucuk = metin.lower()

    for ifade in YASAK_IFADELER:
        if ifade in kucuk:
            return False

    if re.search(r"\n\s*[-•]\s*", metin):
        return False

    if re.search(r"\n\s*\d+[.)]\s*", metin):
        return False

    return True


# ============================================================
# 9. FALLBACK SORULAR
# ============================================================

FALLBACK_SORULAR = {

    "1. Ayrıştırma":
        "Problemi anlamak için hangi iki bilgi arasındaki ilişkiye özellikle dikkat etmek istersin?",

    "2. Soyutlama":
        "Değişkenlerden hangisinin diğerindeki değişime bağlı olduğunu nasıl anlayabilirsin?",

    "3. Algoritma Tasarımı":
        "Çözüm yolunu oluştururken hangi adımın önce gelmesi gerektiğini nasıl belirleyebilirsin?",

    "4. Hata Ayıklama":
        "Bulduğun ilişkinin problemde verilen başka bir bilgiyle uyumlu olup olmadığını nasıl kontrol edebilirsin?"
}


# ============================================================
# 10. SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

if "uploaded_file_mime" not in st.session_state:
    st.session_state.uploaded_file_mime = None

if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = {
        basamak: [] for basamak in BASAMAKLARI
    }

if "current_step" not in st.session_state:
    st.session_state.current_step = BASAMAKLARI[0]

if "completed_stages" not in st.session_state:
    st.session_state.completed_stages = set()

if "stage_ready" not in st.session_state:
    st.session_state.stage_ready = {
        basamak: False for basamak in BASAMAKLARI
    }

if "student_id" not in st.session_state:
    st.session_state.student_id = ""


# ============================================================
# 11. CSV KAYDI
# ============================================================

def log_kaydet(data):

    dosya = "tez_verileri_final.csv"

    df = pd.DataFrame([data])

    df.to_csv(
        dosya,
        mode="a",
        index=False,
        header=not os.path.isfile(dosya),
        encoding="utf-8-sig"
    )


# ============================================================
# 12. KONUŞMA GEÇMİŞİ
# ============================================================

def konusma_gecmisini_olustur(basamak):

    gecmis = st.session_state.chat_storage[basamak]

    if not gecmis:
        return "Bu basamakta henüz konuşma yapılmadı."

    metinler = []

    for i, mesaj in enumerate(gecmis, start=1):

        if mesaj["role"] == "user":
            rol = "ÖĞRENCİ"
        else:
            rol = "REHBER"

        metinler.append(
            f"{i}. {rol}: {mesaj['content']}"
        )

    return "\n".join(metinler)


# ============================================================
# 13. GEMINI İÇİN GERÇEK KONUŞMA GEÇMİŞİ
# ============================================================

def gemini_history_olustur(basamak):

    gecmis = st.session_state.chat_storage[basamak]

    contents = []

    for mesaj in gecmis:

        role = (
            "user"
            if mesaj["role"] == "user"
            else "model"
        )

        contents.append({
            "role": role,
            "parts": [
                {
                    "text": mesaj["content"]
                }
            ]
        })

    return contents


# ============================================================
# 14. ÖNCEKİ BOT SORULARI
# ============================================================

def onceki_bot_sorulari(basamak):

    sorular = []

    for mesaj in st.session_state.chat_storage[basamak]:

        if mesaj["role"] == "assistant":
            sorular.append(mesaj["content"])

    return sorular


# ============================================================
# 15. GEMINI YANITI
# ============================================================

def gemini_yanit_uret(basamak, ogrenci_mesaji):

    history = gemini_history_olustur(basamak)

    onceki_sorular = onceki_bot_sorulari(basamak)

    if onceki_sorular:
        onceki_sorular_metni = "\n".join(
            f"- {soru}"
            for soru in onceki_sorular[-8:]
        )
    else:
        onceki_sorular_metni = "Henüz önceki rehber sorusu yok."

    kontrol_mesaji = f"""
MEVCUT BASAMAK:
{basamak}

BU BASAMAĞIN PEDAGOJİK AMACI:
{BASAMAK_TALIMATLARI[basamak]}

ALT BECERİLER:
{alt_beceri_listesi(basamak)}

ÖNCEKİ REHBER SORULARI:
{onceki_sorular_metni}

ÖNEMLİ:
Önceki rehber sorularından hiçbirini tekrar etme.

Öğrencinin son cevabını mutlaka değerlendir.

Öğrenci son soruya cevap verdiyse aynı düşünme hedefini
yeniden sorma.

Bir sonraki anlamlı düşünme adımına geç.

YANITINDA SADECE BİR SORU BULUNSUN.

Yalnızca öğrenciye gönderilecek doğal Türkçe mesajı üret.
"""

    # Gerçek konuşma geçmişinin sonuna kontrol mesajı ekleniyor.
    contents = list(history)

    contents.append({
        "role": "user",
        "parts": [
            {
                "text": kontrol_mesaji
            }
        ]
    })

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
            "topP": 0.7,
            "maxOutputTokens": 150
        }
    }


    # ========================================================
    # GÖRSEL
    # ========================================================

    if (
        st.session_state.uploaded_file_data
        and st.session_state.uploaded_file_mime
    ):

        # Görseli son kullanıcı mesajına ekle
        contents[-1]["parts"].append({
            "inline_data": {
                "mime_type":
                    st.session_state.uploaded_file_mime,

                "data":
                    st.session_state.uploaded_file_data
            }
        })


    # ========================================================
    # API
    # ========================================================

    son_hata = None

    for deneme in range(3):

        try:

            response = requests.post(
                GEMINI_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json"
                },
                timeout=45
            )

            try:
                sonuc = response.json()
            except Exception:
                sonuc = {}

            if response.status_code == 503:

                son_hata = sonuc
                time.sleep(2 * (deneme + 1))
                continue

            if response.status_code == 429:

                son_hata = sonuc
                time.sleep(3 * (deneme + 1))
                continue

            if "candidates" not in sonuc:

                son_hata = sonuc
                break

            metin = (
                sonuc["candidates"][0]
                ["content"]
                ["parts"][0]
                ["text"]
            ).strip()


            # =================================================
            # GÜVENLİK KONTROLÜ
            # =================================================

            if not ai_yaniti_guvenli_mi(metin):

                return (
                    FALLBACK_SORULAR[basamak],
                    "FALLBACK"
                )


            # =================================================
            # AYNI SORUYU TEKRARLAMA KONTROLÜ
            # =================================================

            onceki = onceki_bot_sorulari(basamak)

            if metin in onceki:

                return (
                    FALLBACK_SORULAR[basamak],
                    "TEKRAR_KONTROL"
                )


            return metin, "GEMINI"


        except Exception as e:

            son_hata = str(e)

            time.sleep(2)


    return None, son_hata


# ============================================================
# 16. ALT BECERİ LİSTESİ
# ============================================================

def alt_beceri_listesi(basamak):

    return "\n".join(
        f"- {kod}: {aciklama}"
        for kod, aciklama in ALT_BECERILER[basamak]
    )


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👨‍🏫 Araştırma Paneli")

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

            st.success("Admin Paneli Aktif")

            dosya_yolu = "tez_verileri_final.csv"

            if os.path.isfile(dosya_yolu):

                try:

                    df_csv = pd.read_csv(
                        dosya_yolu,
                        sep=None,
                        engine="python",
                        on_bad_lines="skip"
                    )

                    st.write("### 📊 Veri Kayıtları")

                    st.dataframe(
                        df_csv.tail(50),
                        use_container_width=True
                    )

                    csv_data = (
                        df_csv
                        .to_csv(index=False)
                        .encode("utf-8-sig")
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
        placeholder="Örn: Hakan",
        value=st.session_state.student_id
    )

    if not student_id:

        st.warning(
            "Devam etmek için öğrenci numarasını girin."
        )

        st.stop()

    st.session_state.student_id = student_id


    st.divider()

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
        list(tool_map.keys())
    )

    drawing_mode = tool_map[secilen_etiket]


    stroke_color = st.color_picker(
        "Çizgi Rengi:",
        "#000000"
    )


    fill_color = st.color_picker(
        "Kutu Rengi:",
        "#EEEEEE"
    )


    st.divider()


    # ========================================================
    # AŞAMA SEÇİMİ
    # ========================================================

    secilebilir_basamaklar = BASAMAKLARI[:1]

    for i in range(1, len(BASAMAKLARI)):

        onceki = BASAMAKLARI[i - 1]

        if (
            onceki in st.session_state.completed_stages
            or st.session_state.stage_ready.get(
                onceki,
                False
            )
        ):

            secilebilir_basamaklar.append(
                BASAMAKLARI[i]
            )


    secilen_basamak = st.radio(
        "Aşamayı Seçin:",
        secilebilir_basamaklar,
        index=(
            secilebilir_basamaklar.index(
                st.session_state.current_step
            )
        )
    )


    if (
        secilen_basamak
        != st.session_state.current_step
    ):

        st.session_state.current_step = (
            secilen_basamak
        )

        st.rerun()


# ============================================================
# 18. ANA BAŞLIK
# ============================================================

st.title(
    "🎯 Doğrusal İlişkiler - Algoritmik Problem Çözme Rehberi"
)

st.write(
    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)


# ============================================================
# 19. GÖRSEL YÜKLEME
# ============================================================

uploaded_file = st.file_uploader(
    "Soru Fotoğrafını Yükle",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


bg_url = None

canvas_height = 450

display_width = 800


if uploaded_file is not None:

    try:

        file_bytes = uploaded_file.getvalue()

        raw_img = Image.open(
            io.BytesIO(file_bytes)
        )

        mime_type = uploaded_file.type

        if not mime_type:

            if uploaded_file.name.lower().endswith(
                ".png"
            ):

                mime_type = "image/png"

            else:

                mime_type = "image/jpeg"


        st.session_state.uploaded_file_data = (
            base64.b64encode(
                file_bytes
            ).decode()
        )

        st.session_state.uploaded_file_mime = (
            mime_type
        )


        w, h = raw_img.size

        canvas_height = int(
            h * (
                display_width / w
            )
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


        img_base64 = base64.b64encode(
            buffered.getvalue()
        ).decode()


        bg_url = (
            "data:image/png;base64,"
            + img_base64
        )


        st.markdown(
            f"""
            <style>

            iframe[title="streamlit_drawable_canvas.st_canvas"]
            {{
                background-image:
                    url("{bg_url}") !important;

                background-size:
                    contain !important;

                background-repeat:
                    no-repeat !important;

                background-position:
                    center !important;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"Görsel yükleme hatası: {e}"
        )


st.divider()


# ============================================================
# 20. İKİ SÜTUN
# ============================================================

col1, col2 = st.columns(
    [1.3, 1],
    gap="large"
)


# ============================================================
# 21. SOL PANEL
# ============================================================

with col1:

    st.write(
        "🖼️ **Tasarım ve Planlama Alanı**"
    )

    st.caption(
        "Problem üzerinde çizim yapabilir veya "
        "akış şeması oluşturabilirsin."
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
            "canvas_"
            + st.session_state.current_step
            .replace(" ", "_")
        )
    )


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

                "tip":
                    "Cizim",

                "alt_beceri":
                    "",

                "basamak_tamamlandi":
                    "",

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

    st.info(
        "🧠 **Öz-Yansıtma:** "
        + METABILISSEL_SORULAR[
            st.session_state.current_step
        ]
    )


    m_cevap = st.text_area(
        "Düşünceni buraya yaz...",
        key=(
            "meta_area_"
            + st.session_state.current_step[0]
        )
    )


    if st.button(
        "💾 Düşüncemi Kaydet"
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

            "tip":
                "Metabiliş",

            "alt_beceri":
                "",

            "basamak_tamamlandi":
                "",

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
            "slider_"
            + st.session_state.current_step
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

            "tip":
                "Eminlik",

            "alt_beceri":
                "",

            "basamak_tamamlandi":
                "",

            "ogrenci_mesaji":
                confidence,

            "ai_mesaji":
                "",

            "icerik":
                confidence
        })


        st.success(
            f"Eminlik: {confidence}"
        )


    # ========================================================
    # FİNAL ÖZET
    # ========================================================

    st.write("---")


    if st.button(
        "🏁 Çözümü Bitir ve Özetini Al"
    ):

        with st.spinner(
            "Süreç analiz ediliyor..."
        ):

            try:

                butun_surec = ""

                for basamak in BASAMAKLARI:

                    butun_surec += (
                        f"\n\n===== "
                        f"{basamak} =====\n"
                    )

                    butun_surec += (
                        konusma_gecmisini_olustur(
                            basamak
                        )
                    )


                final_payload = {

                    "system_instruction": {

                        "parts": [{

                            "text": """

Sen ortaokul matematik öğretmenisin.

Öğrencinin doğrusal ilişkiler içeren problem
çözme sürecini pedagojik açıdan değerlendir.

Öğrencinin yerine çözüm yapma.

Öğrencinin:

- problemi ayrıştırmasını,
- değişkenleri ve ilişkileri fark etmesini,
- doğrusal yapıyı soyutlamasını,
- çözüm algoritması oluşturmasını,
- çözümünü kontrol etmesini

süreç kayıtlarına dayanarak özetle.

Eksik yönleri yargılayıcı olmayan bir dille belirt.

Doğru çözümü veya sonucu verme.

Kısa bir süreç değerlendirmesi oluştur.
"""
                        }]
                    },

                    "contents": [{

                        "parts": [{

                            "text":
                                "Öğrencinin süreç kayıtları:\n"
                                + butun_surec
                        }]
                    }],

                    "generationConfig": {

                        "temperature": 0.3,

                        "maxOutputTokens": 500
                    }
                }


                response = requests.post(

                    GEMINI_URL,

                    json=final_payload,

                    headers={
                        "Content-Type":
                            "application/json"
                    },

                    timeout=45
                )


                res_final = response.json()


                if "candidates" in res_final:

                    summary_text = (
                        res_final[
                            "candidates"
                        ][0][
                            "content"
                        ][
                            "parts"
                        ][0][
                            "text"
                        ]
                    )


                    st.info(
                        summary_text
                    )


                    log_kaydet({

                        "tarih":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "ogrenci_id":
                            student_id,

                        "basamak":
                            "FİNAL",

                        "tip":
                            "Final Özeti",

                        "alt_beceri":
                            "",

                        "basamak_tamamlandi":
                            "",

                        "ogrenci_mesaji":
                            "",

                        "ai_mesaji":
                            summary_text,

                        "icerik":
                            summary_text
                    })


                else:

                    st.error(
                        f"Özet oluşturma hatası: "
                        f"{res_final}"
                    )


            except Exception as e:

                st.error(
                    f"Özet hatası: {e}"
                )


# ============================================================
# 22. SAĞ PANEL – REHBER BOT
# ============================================================

with col2:

    st.write(
        "💬 **Rehber Bot**"
    )


    mevcut_basamak = (
        st.session_state.current_step
    )


    if (
        mevcut_basamak
        in st.session_state.completed_stages
    ):

        st.success(
            "✓ Bu basamak tamamlandı."
        )

    else:

        st.caption(
            "Bot yalnızca bu basamağın "
            "düşünme sürecine rehberlik eder."
        )


    # ========================================================
    # KONUŞMA
    # ========================================================

    chat_container = st.container(
        height=550
    )


    for mesaj in st.session_state.chat_storage[
        mevcut_basamak
    ]:

        with chat_container.chat_message(
            mesaj["role"]
        ):

            st.write(
                mesaj["content"]
            )


    # ========================================================
    # İLK MESAJ
    # ========================================================

    if not st.session_state.chat_storage[
        mevcut_basamak
    ]:

        baslangic_sorulari = {

            "1. Ayrıştırma":
                "Bu problemde hangi iki değişken arasındaki ilişkiyi anlaman gerektiğini düşünüyorsun?",

            "2. Soyutlama":
                "Verilen bilgilerden hangileri değişkenler arasındaki ilişkiyi ortaya çıkarmana yardımcı olabilir?",

            "3. Algoritma Tasarımı":
                "Kurmak istediğin çözüm yolunda hangi adımdan başlamanın uygun olduğunu düşünüyorsun?",

            "4. Hata Ayıklama":
                "Oluşturduğun çözümün problemdeki bilgilerle uyumlu olduğunu nasıl kontrol edebilirsin?"
        }


        ilk_soru = baslangic_sorulari[
            mevcut_basamak
        ]


        if st.button(
            "▶ Rehberi Başlat",
            key="baslat_" + mevcut_basamak
        ):

            st.session_state.chat_storage[
                mevcut_basamak
            ].append({

                "role":
                    "assistant",

                "content":
                    ilk_soru
            })


            log_kaydet({

                "tarih":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "ogrenci_id":
                    student_id,

                "basamak":
                    mevcut_basamak,

                "tip":
                    "Bot",

                "alt_beceri":
                    "",

                "basamak_tamamlandi":
                    False,

                "ogrenci_mesaji":
                    "",

                "ai_mesaji":
                    ilk_soru,

                "icerik":
                    ilk_soru
            })


            st.rerun()


    # ========================================================
    # CHAT INPUT
    # ========================================================

    p = st.chat_input(
        "Düşünceni buraya yaz..."
    )


    if p:

        # ----------------------------------------------------
        # ÖĞRENCİ MESAJI
        # ----------------------------------------------------

        st.session_state.chat_storage[
            mevcut_basamak
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
                mevcut_basamak,

            "tip":
                "Öğrenci",

            "alt_beceri":
                "",

            "basamak_tamamlandi":
                False,

            "ogrenci_mesaji":
                p,

            "ai_mesaji":
                "",

            "icerik":
                p
        })


        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        with st.spinner(
            "Rehber düşünüyor..."
        ):

            ans, kaynak = gemini_yanit_uret(
                mevcut_basamak,
                p
            )


        if ans:

            # ------------------------------------------------
            # BOT CEVABI
            # ------------------------------------------------

            st.session_state.chat_storage[
                mevcut_basamak
            ].append({

                "role":
                    "assistant",

                "content":
                    ans
            })


            log_kaydet({

                "tarih":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "ogrenci_id":
                    student_id,

                "basamak":
                    mevcut_basamak,

                "tip":
                    "Bot",

                "alt_beceri":
                    "",

                "basamak_tamamlandi":
                    False,

                "ogrenci_mesaji":
                    p,

                "ai_mesaji":
                    ans,

                "icerik":
                    ans,

                "kaynak":
                    kaynak
            })


            st.rerun()


        else:

            st.error(
                "Gemini yanıt oluşturamadı."
            )

            st.error(
                str(kaynak)
            )
