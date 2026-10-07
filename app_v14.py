import streamlit as st
import requests
import pandas as pd
import os
import base64
from datetime import datetime
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
Öğrencinin doğrusal ilişki içeren problemi anlamlı
parçalara ayırmasını sağlamak.

Öğrencinin:
- Hangi bilgilerin birbiriyle ilişkili olduğunu incelemesini,
- Problemi anlamlı parçalara ayırmasını,
- Problemdeki değişen veya birbirine bağlı nicelikleri
  fark etmesini sağla.

Yasak:

- Çözüm yolu söylemek.
- İşlem önermek.
- Formül vermek.
- Denklem vermek.
- Doğrusal ilişkiyi doğrudan söylemek.
- Cevabı söylemek.
- Öğrencinin yerine problemi parçalara ayırmak.
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
        "Kurduğun ilişkinin ve bulduğun sonucun problem koşullarıyla uyumlu olduğundan nasıl emin oldun?"
}


# ============================================================
# ANA SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

Sen ortaokul matematik öğrencisine rehberlik eden
bir matematik öğretmenisin.

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

def gemini_generate(
    parts,
    temperature=0.2,
    max_tokens=250,
    include_problem_image=False
):

    final_parts = []

    # Önce metin/gönderilen içerikler
    final_parts.extend(parts)

    # Gerektiğinde gerçek problem görselini ekle
    if include_problem_image:

        image_part = get_problem_image_part()

        if image_part is not None:

            final_parts.append(
                image_part
            )

    payload = {

        "contents": [
            {
                "role": "user",
                "parts": final_parts
            }
        ],

        "generationConfig": {

            "temperature":
                temperature,

            "topP":
                0.8,

            "maxOutputTokens":
                max_tokens
        }
    }

    response = requests.post(

        GEMINI_URL,

        params={
            "key":
                GEMINI_API_KEY
        },

        headers={
            "Content-Type":
                "application/json"
        },

        json=payload,

        timeout=60
    )

    if response.status_code != 200:

        try:

            detail = response.json()

        except Exception:

            detail = response.text

        raise Exception(
            f"Gemini API Hatası "
            f"({response.status_code}): "
            f"{detail}"
        )

    data = response.json()

    try:

        return (
            data["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
            .strip()
        )

    except Exception:

        raise Exception(
            f"Gemini beklenmeyen yanıt verdi: "
            f"{data}"
        )


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

def ilk_soruyu_olustur(
    current_step,
    problem_analysis
):

    prompt = f"""

{SYSTEM_PROMPT}

MEVCUT BASAMAK:

{current_step}


BASAMAĞIN AKADEMİK AMACI:

{BASAMAK_TALIMATLARI[current_step]}


PROBLEM ANALİZİ:

{problem_analysis}


ÖNEMLİ:

Yüklenen gerçek problem görselini de incele.

Öğrenci henüz bu basamakta cevap vermedi.

İlk soru,
BU PROBLEMDEKİ GERÇEK BİLGİLERE dayanmalıdır.

Genel ve her probleme uyabilecek
"Problemde bize hangi bilgiler verilmiş?"
gibi mekanik bir soru sormaktan kaçın.

Yalnızca bu basamağı başlatacak
TEK bir kısa yönlendirici soru sor.

Çözüm verme.

Açıklama yapma.

Birden fazla soru sorma.

Formül verme.

Denklem verme.

İşlem önerme.

"""


    return gemini_generate(

        [
            {
                "text":
                    prompt
            }
        ],

        temperature=0.2,

        max_tokens=100,

        include_problem_image=True
    )


# ============================================================
# ÖĞRENCİ CEVABINA YANIT VER
# ============================================================

def ogrenciye_cevap_ver(
    current_step,
    problem_analysis,
    chat_history,
    student_message
):

    conversation = ""


    for message in chat_history:

        if message["role"] == "user":

            conversation += (
                f"ÖĞRENCİ: "
                f"{message['content']}\n"
            )

        else:

            conversation += (
                f"REHBER: "
                f"{message['content']}\n"
            )


    prompt = f"""

{SYSTEM_PROMPT}

============================================================
MEVCUT BASAMAK
============================================================

{current_step}


============================================================
BASAMAK PROTOKOLÜ
============================================================

{BASAMAK_TALIMATLARI[current_step]}


============================================================
PROBLEM ANALİZİ
============================================================

{problem_analysis}


============================================================
ÖNCEKİ DİYALOG
============================================================

{conversation}


============================================================
ÖĞRENCİNİN SON MESAJI
============================================================

{student_message}


============================================================
GÖREV
============================================================

Yüklenen gerçek problem görselini incele.

Öğrencinin son mesajını dikkate al.

Öğrencinin söylediği şeyden hareketle
bir sonraki düşünme adımını destekle.

Sorun mutlaka mevcut probleme bağlansın.

Ancak problemi sen çözme.

Öğrencinin yerine düşünme.

Öğrencinin söylemediği bir matematiksel sonucu
onun adına çıkarma.

Yalnızca TEK bir soru sor.

1-2 kısa cümleden fazla yazma.

Birden fazla soru sorma.

Çözüm verme.

Formül verme.

Denklem verme.

İşlem sonucu verme.

Eğim söyleme.

Doğrusal ilişkiyi öğrencinin yerine söyleme.

Cevabı söyleme.

Öğrencinin hatasını doğrudan söyleme.

"""


    return gemini_generate(

        [
            {
                "text":
                    prompt
            }
        ],

        temperature=0.2,

        max_tokens=120,

        include_problem_image=True
    )


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
# SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:

    st.session_state.uploaded_file_data = None


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
    "🎯 Algoritmik Problem Çözme Rehberi"
)


st.write(

    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)


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


        st.session_state.chat_storage = {

            stage: []

            for stage
            in BASAMAK_TALIMATLARI
        }


        st.rerun()


# ============================================================
# YÜKLENMİŞ PROBLEM
# ============================================================

else:

    st.image(

        st.session_state.uploaded_file_data,

        width=800
    )


    if st.button(
        "❌ Soruyu Değiştir"
    ):

        st.session_state.uploaded_file_data = (
            None
        )

        st.session_state.problem_analysis = (
            None
        )


        st.session_state.chat_storage = {

            stage: []

            for stage
            in BASAMAK_TALIMATLARI
        }


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

        key=(
            "canvas_"
            +
            st.session_state.current_step
            .replace(" ", "_")
        )
    )


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

                first_question = (
                    ilk_soruyu_olustur(

                        current_stage,

                        st.session_state
                        .problem_analysis
                    )
                )


                st.session_state \
                    .chat_storage[
                        current_stage
                    ].append({

                        "role":
                            "assistant",

                        "content":
                            first_question
                    })


                log_kaydet({

                    "tarih":
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "id":
                        student_id,

                    "basamak":
                        current_stage,

                    "tip":
                        "Bot",

                    "icerik":
                        first_question
                })


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

            "basamak":
                current_stage,

            "tip":
                "Öğrenci",

            "icerik":
                student_message
        })


        # ----------------------------------------------------
        # GEMINI YANITI
        # ----------------------------------------------------

        with st.spinner(
            "Rehber Bot düşünüyor..."
        ):

            try:

                answer = (
                    ogrenciye_cevap_ver(

                        current_stage,

                        st.session_state
                        .problem_analysis,

                        st.session_state
                        .chat_storage[
                            current_stage
                        ][:-1],

                        student_message
                    )
                )


                # --------------------------------------------
                # BOT YANITINI KAYDET
                # --------------------------------------------

                st.session_state \
                    .chat_storage[
                        current_stage
                    ].append({

                        "role":
                            "assistant",

                        "content":
                            answer
                    })


                log_kaydet({

                    "tarih":
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "id":
                        student_id,

                    "basamak":
                        current_stage,

                    "tip":
                        "Bot",

                    "icerik":
                        answer
                })


                st.rerun()


            except Exception as e:

                st.error(
                    "Gemini yanıt oluşturamadı."
                )

                st.code(
                    str(e)
                )
