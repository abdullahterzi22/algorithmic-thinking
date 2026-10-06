import streamlit as st
import requests
import pandas as pd
import os
import base64
from datetime import datetime
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 1. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Algoritmik Düşünme Atölyesi",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# 2. GEMINI AYARLARI
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


# ============================================================
# 3. VERİ DOSYASI
# ============================================================

DATA_FILE = "tez_verileri_final.csv"


# ============================================================
# 4. BASAMAKLAR
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """
Amaç: Öğrencinin problemi alt bileşenlerine ayırmasını sağlamak.

Öğrencinin:
- Verilenleri fark etmesini,
- İstenen bilgiyi belirlemesini,
- Problemi anlamlı parçalara ayırmasını,
- Bilgiler arasındaki ilişkileri fark etmesini sağla.

Yasak:
- Çözüm yolu söylemek.
- İşlem önermek.
- Formül vermek.
- Cevabı söylemek.
- Sonuca götüren doğrudan ipucu vermek.
""",

    "2. Soyutlama": """
Amaç: Öğrencinin problemin matematiksel yapısını fark etmesini sağlamak.

Öğrencinin:
- Gerekli bilgileri ayırt etmesini,
- Gereksiz bilgileri fark etmesini,
- Önceki bilgileri hatırlamasını,
- Benzer problem yapılarıyla ilişki kurmasını,
- Uygunsa örüntü veya genel yapı fark etmesini sağla.

Yasak:
- EBOB'u doğrudan söylemek.
- EKOK'u doğrudan söylemek.
- Kullanılacak yöntemi söylemek.
- Formül vermek.
- Çözüm vermek.
""",

    "3. Algoritma Tasarımı": """
Amaç: Öğrencinin kendi çözüm planını oluşturmasını sağlamak.

Öğrencinin:
- İlk adımı belirlemesini,
- Sonraki adımları planlamasını,
- Adımların sırasını belirlemesini,
- Neden bu yöntemi seçtiğini açıklamasını,
- Gerekirse alternatif yol düşünmesini sağla.

Yasak:
- Algoritmayı öğrencinin yerine oluşturmak.
- İşlem sırasını söylemek.
- Formül vermek.
- Sonucu söylemek.
""",

    "4. Hata Ayıklama": """
Amaç: Öğrencinin kendi çözümünü kontrol etmesini sağlamak.

Öğrencinin:
- Çözüm adımlarını kontrol etmesini,
- Sonucu problem koşullarıyla karşılaştırmasını,
- Mantıksal tutarlılığı incelemesini,
- Alternatif kontrol yolu düşünmesini sağla.

Yasak:
- "Doğru" demek.
- "Yanlış" demek.
- "Hata yaptın" demek.
- Doğru cevabı ima etmek.
"""
}


# ============================================================
# 5. METABİLİŞSEL SORULAR
# ============================================================

METABILISSEL_SORULAR = {

    "1. Ayrıştırma":
        "Bu problemi parçalara ayırırken en çok hangi bilgi dikkatini çekti?",

    "2. Soyutlama":
        "Bu soruda özellikle dikkat etmen gereken noktalar nelerdi? Gereksiz olduğunu düşündüğün bilgiler oldu mu?",

    "3. Algoritma Tasarımı":
        "Çözüm adımlarını planlarken nasıl bir yol izledin?",

    "4. Hata Ayıklama":
        "Bulduğun sonucun mantıklı olduğundan nasıl emin oldun? Farklı bir kontrol yolu düşündün?"
}


# ============================================================
# 6. ANA PROMPT
# ============================================================

SYSTEM_PROMPT = """

Sen ortaokul matematik öğrencisine rehberlik eden
bir matematik öğretmenisin.

TEMEL İLKE:

ÖĞRENCİ ÇÖZER.
SEN YALNIZCA DÜŞÜNME SÜRECİNİ YÖNLENDİRİRSİN.

ÇOK ÖNEMLİ:

Sana ayrıca problemin görseli gönderilmektedir.

Problemin görselini DİKKATLİCE İNCELE.

Öğrencinin cevabını mutlaka bu görseldeki gerçek
problem bağlamında değerlendir.

Problem görselinde bulunan sayıları, ifadeleri,
tabloyu, grafiği veya şekli kendin değiştirme.

Görsel ile problem analizi arasında farklılık varsa,
görseli esas al.

Görselde okunamayan bir bilgi varsa bunu uydurma.
Öğrenciden ilgili bilgiyi açıklamasını iste.

KESİN KURALLAR:

1. Asla doğrudan çözüm verme.
2. Asla cevap verme.
3. Asla işlem sonucunu söyleme.
4. Asla formülü doğrudan verme.
5. Öğrencinin yerine hesaplama yapma.
6. Öğrencinin yerine algoritma oluşturma.
7. Öğrencinin hatasını doğrudan söyleme.
8. "Doğru", "yanlış", "hata yaptın" ifadelerini kullanma.
9. Aynı anda birden fazla soru sorma.
10. Her mesajda yalnızca BİR yönlendirici soru sor.
11. Öğrencinin cevabını bekle.
12. Uzun açıklamalar yapma.
13. Liste halinde çözüm verme.
14. Öğrencinin düşüncesini kendin tamamlamaya çalışma.
15. Genel ve problemden bağımsız soru sorma.
16. Daha önce sorulan soruyu tekrar etme.
17. Öğrencinin verdiği cevaptan hareketle bir sonraki
    düşünme adımına yönlendir.
18. Problemdeki belirli bilgileri gerektiğinde soru içinde
    kullanabilirsin; ancak cevabı söyleme.
19. EBOB, EKOK veya başka bir yöntemi öğrenci adına seçme.
20. Öğrencinin yerine alt problem oluşturma.
21. Öğrenciye çözümü düşündürecek ipucu verebilirsin,
    ancak cevabın kendisini verme.

YANIT:

- Türkçe.
- 1-2 kısa cümle.
- Yalnızca bir soru.
- Öğrencinin seviyesine uygun.
- Hazır övgü kalıpları kullanma.

ÖRNEK SORULAR:

"Problemde bize hangi bilgiler verilmiş?"

"Problemde senden ne isteniyor?"

"Bu bilgileri nasıl gruplandırabilirsin?"

"Bu iki bilgi arasında nasıl bir ilişki görüyorsun?"

"İlk adımının ne olması gerektiğine nasıl karar verdin?"

"Bu adımı neden seçtin?"

"Bu sonucu nasıl kontrol edebilirsin?"

ASLA:

"Cevap 40."

"EKOK kullanmalısın."

"EBOB'u bulmalısın."

"Önce 6 ve 8'in..."

"Formülü kullanalım."

"Burada hata yaptın."

"Doğru cevap bu."
"""


# ============================================================
# 7. CSV KAYIT
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
# 8. PROBLEM GÖRSELİNİ HAZIRLAMA
# ============================================================

def get_problem_image_part():

    """
    Yüklenmiş problem görselini Gemini'nin anlayabileceği
    inline_data biçimine dönüştürür.

    KRİTİK:
    Bu fonksiyon yalnızca problem ilk yüklendiğinde değil,
    Gemini ile yapılan HER rehberlik çağrısında kullanılır.
    """

    uploaded_file = (
        st.session_state.get(
            "uploaded_file_data"
        )
    )

    if uploaded_file is None:
        return None

    try:

        image_bytes = uploaded_file.getvalue()

        image_data = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        mime_type = uploaded_file.type

        if not mime_type:
            mime_type = "image/jpeg"

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": image_data
            }
        }

    except Exception:

        return None


# ============================================================
# 9. GEMINI API
# ============================================================

def gemini_generate(
    parts,
    temperature=0.2,
    max_tokens=250,
    include_problem_image=False
):

    # --------------------------------------------------------
    # ÖNEMLİ:
    #
    # include_problem_image=True olduğunda problem görseli
    # aynı Gemini isteğinin içine eklenir.
    # --------------------------------------------------------

    final_parts = []

    if include_problem_image:

        image_part = get_problem_image_part()

        if image_part is not None:

            final_parts.append(
                image_part
            )

    # Metin parçalarını ekle

    final_parts.extend(parts)

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

        params={
            "key": GEMINI_API_KEY
        },

        headers={
            "Content-Type": "application/json"
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
            ["content"]["parts"][0]
            ["text"]
            .strip()
        )

    except Exception:

        raise Exception(
            f"Gemini beklenmeyen yanıt verdi: {data}"
        )


# ============================================================
# 10. GÖRSEL ANALİZİ
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

Yalnızca problemi rehber öğretmenin anlayabilmesi
için analiz et.

Şunları belirle:

1. Problem metni.
2. Verilen bilgiler.
3. İstenen bilgi.
4. Tablo, grafik veya şekil varsa bunların içeriği.
5. Önemli matematiksel ilişkiler.
6. Okunamayan veya belirsiz bölümler.

Çözüm yapma.

İşlem yapma.

Cevap bulma.

Kullanılacak yöntemi söyleme.

Bu analiz daha sonra öğrencinin düşünmesini
yönlendirmek için kullanılacaktır.
"""


    parts = [

        {
            "text": prompt
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
# 11. YENİ AŞAMA İÇİN İLK SORU
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

PROBLEMİN ÖN ANALİZİ:

{problem_analysis}

ÇOK ÖNEMLİ:

Problem görseli de bu isteğin içinde gönderilmiştir.

Önce görseldeki gerçek problemi incele.

Problem analizini görsel ile karşılaştır.

Ardından yalnızca mevcut basamağa uygun
TEK bir kısa yönlendirici soru sor.

Sorun genel bir matematik sorusu olmamalıdır.

Soru doğrudan öğrencinin üzerinde çalıştığı
bu probleme ilişkin olmalıdır.

Öğrenci henüz bu basamakta cevap vermedi.

Çözüm verme.
Açıklama yapma.
Birden fazla soru sorma.
"""


    return gemini_generate(

        [
            {
                "text": prompt
            }
        ],

        temperature=0.2,

        max_tokens=100,

        # ====================================================
        # KRİTİK DÜZELTME
        # Problem görseli ilk soruda da Gemini'ye gidiyor.
        # ====================================================

        include_problem_image=True
    )


# ============================================================
# 12. ÖĞRENCİ CEVABINA YANIT
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

MEVCUT BASAMAK:

{current_step}

BASAMAK PROTOKOLÜ:

{BASAMAK_TALIMATLARI[current_step]}

PROBLEMİN ÖN ANALİZİ:

{problem_analysis}

ÖNCEKİ DİYALOG:

{conversation}

ÖĞRENCİNİN SON MESAJI:

{student_message}

GÖREV:

Öğrencinin son mesajını dikkate al.

Fakat yalnızca öğrencinin mesajına bakarak
genel bir soru üretme.

Önce sana gönderilen PROBLEM GÖRSELİNİ incele.

Öğrencinin söylediği şeyin problemdeki hangi
bilgi, ilişki veya koşulla bağlantılı olduğunu düşün.

Ardından öğrencinin mevcut düşünmesini bir sonraki
küçük adıma taşıyacak TEK bir soru sor.

Sorunun öğrencinin üzerinde çalıştığı gerçek
problemle bağlantılı olması zorunludur.

Öğrencinin yerine düşünme.

Problemi çözme.

1-2 kısa cümleden fazla yazma.

Birden fazla soru sorma.

Çözüm verme.

Formül verme.

İşlem sonucu verme.

Cevabı söyleme.

Öğrencinin hatasını doğrudan söyleme.

Daha önce sorulan soruyu tekrar etme.

Öğrencinin cevabında zaten ortaya koyduğu bilgiyi
tekrar sordurma.

Sadece bir sonraki düşünme adımına yönlendir.
"""


    return gemini_generate(

        [
            {
                "text": prompt
            }
        ],

        temperature=0.2,

        max_tokens=120,

        # ====================================================
        # EN ÖNEMLİ DÜZELTME
        #
        # Öğrenci her cevap verdiğinde problem görseli
        # Gemini'ye TEKRAR gönderiliyor.
        # ====================================================

        include_problem_image=True
    )


# ============================================================
# 13. FİNAL ÖZET
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

Aşağıdaki öğrencinin algoritmik düşünme sürecini
kısa biçimde değerlendir.

Problemi yeniden çözme.

Cevabı söyleme.

Yeni çözüm yolu verme.

Şu dört boyutu değerlendir:

1. Ayrıştırma
2. Soyutlama
3. Algoritma Tasarımı
4. Hata Ayıklama

Her boyut için öğrencinin süreçteki yaklaşımını
kısaca belirt.

Sonunda genel bir süreç değerlendirmesi yap.

Öğrenci:
{student_id}

SÜREÇ:

{process}
"""


    return gemini_generate(

        [
            {
                "text": prompt
            }
        ],

        temperature=0.3,

        max_tokens=500,

        include_problem_image=False
    )


# ============================================================
# 14. SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None


if "problem_analysis" not in st.session_state:
    st.session_state.problem_analysis = None


if "chat_storage" not in st.session_state:

    st.session_state.chat_storage = {

        stage: []

        for stage in BASAMAK_TALIMATLARI
    }


if "current_step" not in st.session_state:

    st.session_state.current_step = (
        "1. Ayrıştırma"
    )


# ============================================================
# 15. SIDEBAR
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


    drawing_mode = tool_map[selected_tool]


    stroke_color = st.color_picker(

        "Çizgi Rengi:",

        "#000000"
    )


    fill_color = st.color_picker(

        "Kutu Rengi:",

        "#EEEEEE"
    )


    st.divider()


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


    if selected_step != (
        st.session_state.current_step
    ):

        st.session_state.current_step = (
            selected_step
        )

        st.rerun()


# ============================================================
# 16. ANA EKRAN
# ============================================================

st.title(
    "🎯 Algoritmik Problem Çözme Rehberi"
)


st.write(

    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)


# ============================================================
# 17. SORU FOTOĞRAFI
# ============================================================

if st.session_state.uploaded_file_data is None:

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

            for stage in BASAMAK_TALIMATLARI
        }


        st.rerun()


else:

    st.image(

        st.session_state.uploaded_file_data,

        width=500
    )


    if st.button(
        "❌ Soruyu Değiştir"
    ):

        st.session_state.uploaded_file_data = None

        st.session_state.problem_analysis = None

        st.session_state.chat_storage = {

            stage: []

            for stage in BASAMAK_TALIMATLARI
        }


        st.rerun()


# ============================================================
# 18. PROBLEM ANALİZİ
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
# 19. SÜTUNLAR
# ============================================================

col1, col2 = st.columns(

    [1.3, 1],

    gap="large"
)


# ============================================================
# 20. SOL TARAF
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


    st.info(

        "🧠 **Öz-Yansıtma:** "
        + METABILISSEL_SORULAR[
            st.session_state.current_step
        ]
    )


    meta_key = (

        "meta_"
        + st.session_state.current_step
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
            + st.session_state.current_step
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


    if st.button(
        "🏁 Çözümü Bitir ve Özetini Al"
    ):

        with st.spinner(
            "Süreç analiz ediliyor..."
        ):

            try:

                final_text = final_ozet_olustur(

                    student_id,

                    st.session_state.chat_storage
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
# 21. SAĞ TARAF – REHBER BOT
# ============================================================

with col2:

    st.write(
        "💬 **Rehber Bot**"
    )


    current_stage = (
        st.session_state.current_step
    )


    # ========================================================
    # İLK SORU
    # ========================================================

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


    # ========================================================
    # SOHBET
    # ========================================================

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


    # ========================================================
    # ÖĞRENCİ MESAJI
    # ========================================================

    student_message = st.chat_input(

        "Düşünceni veya cevabını yaz..."
    )


    if student_message:

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


        with st.spinner(
            "Rehber Bot düşünüyor..."
        ):

            try:

                answer = ogrenciye_cevap_ver(

                    current_stage,

                    st.session_state
                    .problem_analysis,

                    st.session_state
                    .chat_storage[
                        current_stage
                    ][:-1],

                    student_message
                )


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
