import streamlit as st
import pandas as pd
import os
import json
import base64
from datetime import datetime
from google import genai
from google.genai import types
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 1. STREAMLIT AYARLARI
# ============================================================

st.set_page_config(
    page_title="Algoritmik Düşünme Atölyesi",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# 2. GEMINI API AYARLARI
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error(
        """
        ❌ GEMINI_API_KEY bulunamadı.

        Streamlit Secrets bölümüne şu satırı ekleyin:

        GEMINI_API_KEY = "API_KEY_BURAYA"
        """
    )
    st.stop()


# Güncel model
MODEL_NAME = "gemini-3.5-flash-lite"


# Gemini istemcisi
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 3. VERİ DOSYASI
# ============================================================

DATA_FILE = "tez_verileri_final.csv"


# ============================================================
# 4. AKADEMİK BASAMAK TALİMATLARI
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

AMAÇ:
Öğrencinin problemi alt bileşenlerine ayırmasını sağlamak.

ÖĞRENCİNİN YAPMASI BEKLENENLER:
- Verilen bilgileri fark etmek.
- İstenen bilgiyi belirlemek.
- Problemi anlamlı parçalara ayırmak.
- Bilgiler arasındaki ilişkileri fark etmek.
- Problemi kendi cümleleriyle ifade etmek.

REHBERİN YAPMASI GEREKENLER:
- Verilenleri fark ettirecek sorular sor.
- İstenen bilgiyi öğrencinin kendisinin söylemesini sağla.
- Problemdeki ilişkileri öğrencinin fark etmesini sağla.

YASAK:
- Çözüm yolu söyleme.
- İşlem önerme.
- Formül verme.
- Cevap verme.
- Sonuca götüren doğrudan ipucu verme.
""",


    "2. Soyutlama": """

AMAÇ:
Öğrencinin problemin altında yatan matematiksel yapıyı
fark etmesini sağlamak.

ÖĞRENCİNİN YAPMASI BEKLENENLER:
- Problem için gerekli bilgileri ayırt etmek.
- Gereksiz bilgileri fark etmek.
- Önceki matematiksel bilgileri hatırlamak.
- Benzer problem yapılarıyla ilişki kurmak.
- Örüntü veya genel yapı fark etmek.
- Problemi matematiksel bir temsil ile ifade etmek.

REHBERİN YAPMASI GEREKENLER:
- "Bu bilgilerden hangileri önemli?" gibi sorular sor.
- Önceki bilgileri hatırlatacak sorular sor.
- Benzer bir problemle bağlantı kurdur.
- Öğrencinin kendisinin matematiksel yapıyı keşfetmesini sağla.

YASAK:
- EBOB olduğunu söyleme.
- EKOK olduğunu söyleme.
- Üslü sayı olduğunu söyleme.
- Kullanılacak yöntemi söyleme.
- Formül verme.
- Çözüm verme.
""",


    "3. Algoritma Tasarımı": """

AMAÇ:
Öğrencinin kendi çözüm algoritmasını oluşturmasını sağlamak.

ÖĞRENCİNİN YAPMASI BEKLENENLER:
- Çözümün ilk adımını belirlemek.
- Sonraki adımları planlamak.
- Adımların sırasını belirlemek.
- Ara sonuçları düşünmek.
- Neden bu yöntemi seçtiğini açıklamak.
- Gerekirse alternatif bir yol düşünmek.
- Çözümünü akış şeması ile ifade etmek.

REHBERİN YAPMASI GEREKENLER:
- "İlk olarak ne yapmayı düşünüyorsun?"
- "Bundan sonra hangi adımı izlersin?"
- "Bu adımı neden seçtin?"
- "Bu adımın sonucunu nasıl kullanacaksın?"
gibi sorularla öğrenciyi yönlendir.

YASAK:
- Algoritmayı öğrencinin yerine oluşturma.
- İşlem sırasını söyleme.
- Formül verme.
- Sonucu söyleme.
""",


    "4. Hata Ayıklama": """

AMAÇ:
Öğrencinin kendi çözümünü test etmesini ve varsa
uyumsuzlukları kendisinin fark etmesini sağlamak.

ÖĞRENCİNİN YAPMASI BEKLENENLER:
- Çözüm adımlarını kontrol etmek.
- Sonucu problem koşullarıyla karşılaştırmak.
- Mantıksal tutarlılığı incelemek.
- Alternatif bir kontrol yolu düşünmek.
- Gerekirse bir adımı yeniden düzenlemek.

REHBERİN YAPMASI GEREKENLER:
- Çözümü doğrudan değerlendirme.
- Öğrencinin kendi çözümünü kontrol etmesini sağla.
- Problem koşullarıyla karşılaştırma yaptır.
- Alternatif kontrol yolu düşündür.

YASAK:
- "Doğru."
- "Yanlış."
- "Cevabın doğru."
- "Burada hata yaptın."
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
# 6. ANA PEDAGOJİK SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

SEN BİR ORTAOKUL MATEMATİK ÖĞRETMENİ VE REHBERİSİN.

Öğrencinin matematik problemini kendisinin çözmesini
sağlamakla görevlisin.

TEMEL İLKE:

ÖĞRENCİ ÇÖZER.
SEN YALNIZCA DÜŞÜNME SÜRECİNİ YÖNLENDİRİRSİN.

============================================================
KESİN KURALLAR
============================================================

1. Asla doğrudan çözüm verme.

2. Asla cevap verme.

3. Asla işlem sonucunu söyleme.

4. Asla formülü doğrudan yazma.

5. Öğrencinin yerine hesaplama yapma.

6. Öğrencinin yerine algoritma oluşturma.

7. Öğrencinin yapması gereken işlemi doğrudan söyleme.

8. Öğrencinin hatasını doğrudan söyleme.

9. "Doğru", "yanlış", "hata yaptın" gibi
   kesin değerlendirmeler kullanma.

10. Öğrenciye aynı anda birden fazla soru sorma.

11. Her mesajda yalnızca BİR yönlendirici soru sor.

12. Öğrencinin cevabını bekle.

13. Uzun açıklama yapma.

14. Liste halinde çözüm verme.

15. Öğrencinin düşüncesini kendin tamamlamaya çalışma.

============================================================
YANIT UZUNLUĞU
============================================================

Her yanıt:

- En fazla 2 kısa cümle.
- En fazla 1 soru.
- Doğal Türkçe.
- Öğrencinin yaş seviyesine uygun.

============================================================
SORU SORMA KURALI
============================================================

Öğrencinin son mesajına göre yalnızca bir sonraki
düşünme adımını hedefle.

Örneğin öğrenci verilenleri söylüyorsa:

"Bu bilgilerden hangilerinin problemde kullanılacağını
nasıl belirleyebilirsin?"

Öğrenci bir çözüm adımı söylüyorsa:

"Bu adımı seçmenin sebebi nedir?"

Öğrenci sonuç söylüyorsa:

"Bu sonucu problemdeki koşullardan hangisiyle
karşılaştırabilirsin?"

============================================================
YASAK ÖRNEKLER
============================================================

"Önce EKOK'u bulmalısın."

"6 ve 8'in EKOK'u 24'tür."

"Bu soruda EKOK kullanılır."

"Formülü yazalım."

"Cevap 40 TL."

"İlk olarak 6'yı 8'e böl."

"Burada hata yaptın."

"Doğru cevap bu."

============================================================
İZİN VERİLEN YAKLAŞIM
============================================================

"Problemde bize hangi bilgiler verilmiş?"

"Problemde senden ne isteniyor?"

"Bu bilgileri nasıl gruplandırabilirsin?"

"Bu iki bilgi arasında nasıl bir ilişki görüyorsun?"

"Benzer bir problemi daha önce nasıl düşünmüştün?"

"İlk adımının ne olması gerektiğine nasıl karar verdin?"

"Bu adımın sonucunu nasıl kontrol edebilirsin?"

============================================================
ÖNEMLİ
============================================================

Öğrenci bir şeyi eksik veya belirsiz söylediğinde
cevabı tamamlamaya çalışma.

Öğrencinin kendisinin fark etmesini sağlayacak
tek bir soru sor.

"""


# ============================================================
# 7. VERİ KAYIT FONKSİYONU
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
# 8. GÖRSELİ BASE64'E ÇEVİR
# ============================================================

def image_to_base64(uploaded_file):

    return base64.b64encode(
        uploaded_file.getvalue()
    ).decode("utf-8")


# ============================================================
# 9. GÖRSEL PROBLEMİ ANALİZ ET
# ============================================================

def analyze_problem_image(uploaded_file):

    if uploaded_file is None:
        return None

    try:

        image_data = uploaded_file.getvalue()

        mime_type = uploaded_file.type

        prompt = """

Bu görselde bir matematik problemi bulunmaktadır.

Görseli ÖĞRENCİYE ÇÖZÜM VERMEK için değil,
rehber öğretmenin problem bağlamını anlayabilmesi için analiz et.

Şunları belirle:

1. Problem metni.
2. Verilen sayısal bilgiler.
3. İstenen bilgi.
4. Varsa tablo, grafik veya şekil.
5. Görseldeki önemli matematiksel ilişkiler.
6. Görselde okunamayan veya belirsiz bir bölüm varsa belirt.

ÇÖZÜM YAPMA.

İŞLEM YAPMA.

CEVAP BULMA.

EBOB, EKOK veya başka bir yöntemi çözüm olarak önerme.

Yalnızca problem bağlamını çıkar.
"""

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[

                types.Part.from_bytes(
                    data=image_data,
                    mime_type=mime_type
                ),

                prompt
            ]
        )

        return response.text.strip()

    except Exception as e:

        return f"Görsel analiz edilemedi: {e}"


# ============================================================
# 10. GEMINI İLE YENİ ETKİLEŞİM BAŞLAT
# ============================================================

def yeni_interaction_baslat(
    current_step,
    problem_analysis
):

    stage_instruction = (
        BASAMAK_TALIMATLARI[
            current_step
        ]
    )

    prompt = f"""

{SYSTEM_PROMPT}

============================================================
MEVCUT BASAMAK
============================================================

{current_step}

============================================================
BU BASAMAĞIN AKADEMİK PROTOKOLÜ
============================================================

{stage_instruction}

============================================================
PROBLEM ANALİZİ
============================================================

{problem_analysis}

============================================================
GÖREV
============================================================

Öğrenci henüz bu basamakta herhangi bir cevap vermedi.

Bu nedenle çözüm verme.

Yalnızca öğrencinin bu basamağa başlamasını sağlayacak
TEK bir kısa yönlendirici soru sor.

Başka açıklama yapma.
"""

    try:

        interaction = client.interactions.create(

            model=MODEL_NAME,

            input=prompt,

            store=True
        )

        return interaction

    except Exception as e:

        raise Exception(
            f"Gemini etkileşimi başlatılamadı: {e}"
        )


# ============================================================
# 11. ÖĞRENCİ CEVABINA GEMINI YANITI
# ============================================================

def gemini_ogrenci_cevabi(
    current_step,
    student_message,
    previous_interaction_id,
    problem_analysis
):

    stage_instruction = (
        BASAMAK_TALIMATLARI[
            current_step
        ]
    )

    prompt = f"""

{SYSTEM_PROMPT}

============================================================
MEVCUT BASAMAK
============================================================

{current_step}

============================================================
AKADEMİK PROTOKOL
============================================================

{stage_instruction}

============================================================
PROBLEM BAĞLAMI
============================================================

{problem_analysis}

============================================================
ÖĞRENCİNİN SON MESAJI
============================================================

{student_message}

============================================================
GÖREV
============================================================

Öğrencinin SON mesajını değerlendir.

Ancak çözümü sen yapma.

Öğrencinin yerine düşünme.

Yalnızca öğrencinin bir sonraki düşünme adımını
destekle.

TEK BİR KISA SORU SOR.

Birden fazla soru sorma.

Açıklama yapma.

Çözüm verme.

Formül verme.

İşlem sonucu verme.

Cevabı söyleme.

Öğrencinin hatasını doğrudan söyleme.
"""

    try:

        interaction = client.interactions.create(

            model=MODEL_NAME,

            input=prompt,

            previous_interaction_id=
                previous_interaction_id,

            store=True
        )

        return interaction

    except Exception as e:

        raise Exception(
            f"Gemini yanıt oluşturamadı: {e}"
        )


# ============================================================
# 12. FİNAL ÖZETİ
# ============================================================

def final_ozet_olustur(
    student_id,
    chat_storage
):

    process_text = ""

    for stage, messages in chat_storage.items():

        process_text += (
            f"\n\n===== {stage} =====\n"
        )

        for message in messages:

            process_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )


    prompt = f"""

Sen bir ortaokul matematik öğretmenisin.

Aşağıdaki öğrencinin algoritmik düşünme sürecini
öğretmen gözüyle değerlendir.

ÖNEMLİ:

Problemi yeniden çözme.

Cevabı söyleme.

Yeni bir çözüm yolu verme.

Öğrencinin çözümünü değiştirme.

Bunun yerine şu başlıklarda kısa bir değerlendirme yap:

1. Ayrıştırma
2. Soyutlama
3. Algoritma Tasarımı
4. Hata Ayıklama

Her başlık altında öğrencinin süreçte
ne yaptığını kısaca belirt.

Sonunda öğrencinin algoritmik düşünme süreciyle ilgili
kısa bir genel değerlendirme yap.

Öğrenci:
{student_id}

ÖĞRENCİNİN SÜRECİ:

{process_text}
"""

    try:

        interaction = client.interactions.create(

            model=MODEL_NAME,

            input=prompt,

            store=False
        )

        return interaction.output_text.strip()

    except Exception as e:

        raise Exception(
            f"Final özeti oluşturulamadı: {e}"
        )


# ============================================================
# 13. SESSION STATE
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


if "interaction_ids" not in st.session_state:

    st.session_state.interaction_ids = {

        stage: None

        for stage in BASAMAK_TALIMATLARI
    }


if "current_step" not in st.session_state:

    st.session_state.current_step = (
        "1. Ayrıştırma"
    )


# ============================================================
# 14. SIDEBAR
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
    # ADMIN PANELİ
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
    # ÖĞRENCİ GİRİŞİ
    # ========================================================

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


    # ========================================================
    # AKIŞ ŞEMASI ARAÇLARI
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

        list(tool_map.keys())
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


    st.divider()


    # ========================================================
    # AŞAMA SEÇİMİ
    # ========================================================

    step_list = list(
        BASAMAK_TALIMATLARI.keys()
    )


    current_index = step_list.index(
        st.session_state.current_step
    )


    selected_step = st.radio(

        "Aşamayı Seçin:",

        step_list,

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
# 15. ANA BAŞLIK
# ============================================================

st.title(
    "🎯 Algoritmik Problem Çözme Rehberi"
)


st.write(

    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)


# ============================================================
# 16. PROBLEM FOTOĞRAFI
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


    if uploaded is not None:

        st.session_state.uploaded_file_data = (
            uploaded
        )


        with st.spinner(
            "Problem analiz ediliyor..."
        ):

            analysis = (
                analyze_problem_image(
                    uploaded
                )
            )


        st.session_state.problem_analysis = (
            analysis
        )


        # ----------------------------------------------------
        # Yeni problem için etkileşimleri temizle
        # ----------------------------------------------------

        st.session_state.interaction_ids = {

            stage: None

            for stage in BASAMAK_TALIMATLARI
        }


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

        st.session_state.uploaded_file_data = (
            None
        )

        st.session_state.problem_analysis = (
            None
        )


        st.session_state.interaction_ids = {

            stage: None

            for stage in BASAMAK_TALIMATLARI
        }


        st.session_state.chat_storage = {

            stage: []

            for stage in BASAMAK_TALIMATLARI
        }


        st.rerun()


# ============================================================
# 17. PROBLEM ANALİZİ
# ============================================================

if st.session_state.problem_analysis:

    with st.expander(
        "🔎 Problem analizini göster",
        expanded=False
    ):

        st.write(
            st.session_state.problem_analysis
        )


st.divider()


# ============================================================
# 18. İKİ SÜTUN
# ============================================================

col1, col2 = st.columns(

    [1.3, 1],

    gap="large"
)


# ============================================================
# 19. SOL SÜTUN
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


    # ========================================================
    # TASARIM KAYDET
    # ========================================================

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
                    json.dumps(
                        canvas_result.json_data,
                        ensure_ascii=False
                    )
            })


            st.success(
                "Tasarım veritabanına kaydedildi!"
            )


    st.write("---")


    # ========================================================
    # METABİLİŞ
    # ========================================================

    st.info(

        "🧠 **Öz-Yansıtma:** "
        + METABILISSEL_SORULAR[
            st.session_state.current_step
        ]
    )


    m_cevap = st.text_area(

        "Düşünceni buraya yaz...",

        key=(
            "meta_"
            + st.session_state.current_step
        )
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
            "Düşüncen kaydedildi!"
        )


    st.write("---")


    # ========================================================
    # EMİNLİK
    # ========================================================

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
            f"Eminlik derecen: {confidence}"
        )


    st.write("---")


    # ========================================================
    # FİNAL ÖZETİ
    # ========================================================

    if st.button(
        "🏁 Çözümü Bitir ve Özetini Al"
    ):

        with st.spinner(
            "Çözüm süreci analiz ediliyor..."
        ):

            try:

                final_text = (
                    final_ozet_olustur(

                        student_id,

                        st.session_state
                        .chat_storage
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
                    "Final özeti hazırlanamadı."
                )

                st.code(
                    str(e)
                )


# ============================================================
# 20. SAĞ SÜTUN – REHBER BOT
# ============================================================

with col2:

    st.write(
        "💬 **Rehber Bot**"
    )


    # ========================================================
    # İLK ETKİLEŞİM
    # ========================================================

    current_stage = (
        st.session_state.current_step
    )


    if (
        st.session_state.problem_analysis
        and
        st.session_state
        .interaction_ids[
            current_stage
        ] is None
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

                interaction = (
                    yeni_interaction_baslat(

                        current_stage,

                        st.session_state
                        .problem_analysis
                    )
                )


                answer = (
                    interaction
                    .output_text
                    .strip()
                )


                st.session_state \
                    .interaction_ids[
                        current_stage
                    ] = interaction.id


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
                    "Gemini ilk soruyu oluşturamadı."
                )

                st.code(
                    str(e)
                )


    # ========================================================
    # SOHBET ALANI
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

        "Düşünceni buraya yaz..."
    )


    if student_message:

        # ----------------------------------------------------
        # Öğrenci mesajını kaydet
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
        # GEMINI
        # ----------------------------------------------------

        with st.spinner(
            "Rehber Bot düşünüyor..."
        ):

            try:

                previous_id = (
                    st.session_state
                    .interaction_ids[
                        current_stage
                    ]
                )


                interaction = (
                    gemini_ogrenci_cevabi(

                        current_stage,

                        student_message,

                        previous_id,

                        st.session_state
                        .problem_analysis
                    )
                )


                answer = (
                    interaction
                    .output_text
                    .strip()
                )


                # ------------------------------------------------
                # SONRAKİ ETKİLEŞİM ID
                # ------------------------------------------------

                st.session_state \
                    .interaction_ids[
                        current_stage
                    ] = interaction.id


                # ------------------------------------------------
                # BOT CEVABI
                # ------------------------------------------------

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
