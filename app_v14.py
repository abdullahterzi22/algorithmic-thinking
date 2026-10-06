import streamlit as st
import requests
import pandas as pd
import os
import base64
from datetime import datetime
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
# 2. GEMINI AYARLARI
# ============================================================

# API anahtarı Streamlit Secrets'tan alınır.
# .streamlit/secrets.toml içinde:
#
# GEMINI_API_KEY = "BURAYA_API_KEY"
#

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error(
        "❌ GEMINI_API_KEY bulunamadı.\n\n"
        "Streamlit Secrets bölümüne GEMINI_API_KEY ekleyin."
    )
    st.stop()


# Gemini 2.5 Flash-Lite
MODEL_NAME = "gemini-2.5-flash-lite"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL_NAME}:generateContent"
)


# ============================================================
# 3. AKADEMİK PROTOKOLLER
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """
Amaç:
Öğrencinin matematik problemini alt bileşenlerine ayırmasını sağlamak.

Zorunlu davranışlar:
- Verilenleri fark ettir.
- İstenen bilgiyi açıkça söylet.
- Problemi uygun alt parçalara ayırmasını sağla.
- Sayısal ilişkileri fark ettir.
- Öğrencinin problemi kendi cümleleriyle ifade etmesini destekle.

Yasak:
- Çözüm yolu önermek.
- İşlem yaptırmak.
- Sonuca yaklaşan ipucu vermek.
- Formül vermek.
- Cevabı söylemek.
""",

    "2. Soyutlama": """
Amaç:
Öğrencinin problemin matematiksel yapısını ortaya çıkarmasını sağlamak.

Zorunlu davranışlar:
- Problemde hangi bilgilerin önemli olduğunu düşündür.
- Gereksiz bilgileri ayırt ettir.
- Önceki matematiksel bilgileri hatırlat.
- Benzer problem yapılarıyla bağlantı kurmasını sağla.
- Uygunsa örüntü veya genel yapı düşündür.
- Öğrencinin problemi matematiksel bir temsil ile ifade etmesini destekle.

Yasak:
- EBOB, EKOK veya başka bir kavramı doğrudan cevap olarak söylemek.
- Stratejiyi açıkça belirtmek.
- İşlem yapmak.
- Sonucu söylemek.
""",

    "3. Algoritma Tasarımı": """
Amaç:
Öğrencinin kendi çözüm planını oluşturmasını sağlamak.

Zorunlu davranışlar:
- Adım adım plan oluşturmasını iste.
- Önce ne yapacağını sorgula.
- Sonraki adımı öğrencinin belirlemesini sağla.
- Neden bu yöntemi seçtiğini sorgula.
- Başka bir yöntem mümkün mü diye düşündür.
- Çözümün genellenebilirliğini sorgula.
- Gerekirse akış şeması oluşturmasını destekle.

Yasak:
- İşlem adımlarını doğrudan vermek.
- Hesap sonucu söylemek.
- Algoritmayı öğrencinin yerine oluşturmak.
""",

    "4. Hata Ayıklama": """
Amaç:
Öğrencinin kendi çözümünü kontrol etmesini sağlamak.

Zorunlu davranışlar:
- Sonucun problem koşullarını sağlayıp sağlamadığını sorgula.
- Öğrencinin çözümündeki adımları kontrol etmesini sağla.
- Alternatif doğrulama yolu düşündür.
- Mantıksal tutarlılığı sorgula.
- Gerekirse öğrencinin belirli bir adımı yeniden incelemesini sağla.

Yasak:
- Sonucu doğrudan doğru veya yanlış olarak söylemek.
- Doğru cevabı ima etmek.
- Çözümü öğrencinin yerine düzeltmek.
"""
}


# ============================================================
# 4. METABİLİŞSEL SORULAR
# ============================================================

METABILISSEL_SORULAR = {

    "1. Ayrıştırma":
        "Bu problemi parçalara ayırırken en çok hangi bilgi dikkatini çekti?",

    "2. Soyutlama":
        "Bu soruda özellikle dikkat etmen gereken noktalar nelerdi? Gereksiz olduğunu düşündüğün bilgiler oldu mu?",

    "3. Algoritma Tasarımı":
        "Çözüm adımlarını planlarken nasıl bir yol izledin?",

    "4. Hata Ayıklama":
        "Bulduğun sonucun mantıklı olduğundan nasıl emin oldun? Farklı bir kontrol yolu düşündün mü?"
}


# ============================================================
# 5. GENEL SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

ROLÜN:
Ortaokul matematik öğrencisine rehberlik eden bir öğretmensin.

TEMEL AMAÇ:
Öğrencinin matematik problemini KENDİSİNİN çözmesini sağlamak.

KESİN KURALLAR:

1. Asla doğrudan çözüm verme.
2. Asla işlem sonucunu söyleme.
3. Asla formülü doğrudan verme.
4. Öğrenci yerine problem çözme.
5. Öğrencinin yerine algoritma oluşturma.
6. Öğrenciye cevabı ima eden ipucu verme.
7. Uzun açıklamalar yapma.
8. Aynı anda birden fazla soru sorma.
9. Her yanıtında yalnızca BİR yönlendirici soru sor.
10. Öğrencinin cevabını bekle.

YANIT BİÇİMİ:

- Türkçe yaz.
- 1-3 kısa cümle kullan.
- Yanıtın sonunda yalnızca bir soru bulunmalıdır.
- Önceki öğrencinin söylediğini gereksiz yere tekrar etme.
- Hazır övgü kalıpları kullanma.
- "Harika bir başlangıç noktası!" gibi kalıpları kullanma.

KULLANABİLECEĞİN YAKLAŞIMLAR:

- "Bu bilgiyi neden önemli gördün?"
- "Burada senden ne isteniyor?"
- "Bu bilgileri nasıl gruplandırabilirsin?"
- "Bu iki bilgi arasında nasıl bir ilişki görüyorsun?"
- "İlk adımın ne olmalı?"
- "Bu adımı neden seçtin?"
- "Bulduğun sonucu nasıl kontrol edebilirsin?"

ASLA ŞUNLARI YAPMA:

- "Cevap ..."
- "Sonuç ..."
- "Şu işlemi yap."
- "Burada EKOK kullanmalısın."
- "Burada EBOB kullanmalısın."
- "Önce 6 ve 8'in..."
- "Şimdi şu formülü kullan."
- "Yanlış yaptın."
- "Doğru cevap şu."

ÖĞRENCİNİN HATASINI DOĞRUDAN SÖYLEME.

Bunun yerine öğrencinin kendi hatasını fark etmesini sağlayacak
tek bir kısa soru sor.

"""


# ============================================================
# 6. DOSYAYA KAYIT
# ============================================================

DATA_FILE = "tez_verileri_final.csv"


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

        st.warning(f"Veri kaydedilemedi: {e}")


# ============================================================
# 7. GÖRSELİ GEMINI'YE HAZIRLAMA
# ============================================================

def dosyayi_base64_yap(file):

    return base64.b64encode(
        file.getvalue()
    ).decode("utf-8")


# ============================================================
# 8. GEMINI'YE MESAJ GÖNDERME
# ============================================================

def gemini_cevap_al(
    current_step,
    chat_history,
    image_file=None
):

    stage_instruction = BASAMAK_TALIMATLARI[current_step]

    full_system = f"""
{SYSTEM_PROMPT}

ŞU ANDAKİ BASAMAK:
{current_step}

BU BASAMAĞIN AKADEMİK PROTOKOLÜ:
{stage_instruction}
"""

    contents = []

    # --------------------------------------------------------
    # Sistem + konuşma bağlamı
    # --------------------------------------------------------

    conversation_text = full_system + "\n\n"

    conversation_text += "ÖNCEKİ KONUŞMA:\n"

    for message in chat_history:

        role = message["role"]

        if role == "user":
            conversation_text += f"ÖĞRENCİ: {message['content']}\n"

        elif role == "assistant":
            conversation_text += f"REHBER: {message['content']}\n"

    conversation_text += """

Şimdi öğrencinin son mesajına uygun şekilde cevap ver.

Yalnızca bu basamağa odaklan.

Tek bir kısa yönlendirici soru sor.

Çözüm verme.
"""

    contents.append({
        "text": conversation_text
    })


    # --------------------------------------------------------
    # Görsel varsa Gemini'ye gönder
    # --------------------------------------------------------

    if image_file is not None:

        image_base64 = dosyayi_base64_yap(image_file)

        mime_type = image_file.type

        contents.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": image_base64
            }
        })


    # --------------------------------------------------------
    # Gemini payload
    # --------------------------------------------------------

    payload = {

        "contents": [
            {
                "role": "user",
                "parts": contents
            }
        ],

        "generationConfig": {

            "temperature": 0.2,

            "topP": 0.8,

            "maxOutputTokens": 200
        }
    }


    headers = {
        "Content-Type": "application/json"
    }


    # --------------------------------------------------------
    # API isteği
    # --------------------------------------------------------

    response = requests.post(
        GEMINI_URL,
        headers=headers,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=60
    )


    # --------------------------------------------------------
    # Hata kontrolü
    # --------------------------------------------------------

    if response.status_code != 200:

        try:
            error_detail = response.json()

        except:
            error_detail = response.text

        raise Exception(
            f"Gemini API Hatası ({response.status_code}): "
            f"{error_detail}"
        )


    data = response.json()


    # --------------------------------------------------------
    # Cevabı çıkar
    # --------------------------------------------------------

    try:

        answer = (
            data["candidates"][0]
            ["content"]["parts"][0]["text"]
        )

        return answer.strip()

    except Exception:

        raise Exception(
            f"Gemini yanıtı beklenen formatta değil: {data}"
        )


# ============================================================
# 9. SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None


if "chat_storage" not in st.session_state:

    st.session_state.chat_storage = {
        s: []
        for s in BASAMAK_TALIMATLARI.keys()
    }


if "current_step" not in st.session_state:

    st.session_state.current_step = "1. Ayrıştırma"


if "canvas_data" not in st.session_state:

    st.session_state.canvas_data = {}


# ============================================================
# 10. SIDEBAR
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


    # --------------------------------------------------------
    # ADMIN
    # --------------------------------------------------------

    if mode == "Öğretmen (Admin)":

        sifre = st.text_input(
            "Şifre:",
            type="password"
        )

        if sifre == "tez2024":

            st.success("Admin Paneli Aktif")

            if os.path.isfile(DATA_FILE):

                try:

                    df_csv = pd.read_csv(
                        DATA_FILE,
                        sep=None,
                        engine="python",
                        on_bad_lines="skip"
                    )

                    st.write("### 📊 Veri Kayıtları")

                    st.dataframe(
                        df_csv.tail(20),
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


    # --------------------------------------------------------
    # ÖĞRENCİ
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CANVAS ARAÇLARI
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

        "Çokgen":
            "polygon"
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


    # --------------------------------------------------------
    # AŞAMA SEÇİMİ
    # --------------------------------------------------------

    step_list = list(
        BASAMAK_TALIMATLARI.keys()
    )

    cur_idx = step_list.index(
        st.session_state.current_step
    )


    sel = st.radio(
        "Aşamayı Seçin:",
        step_list,
        index=cur_idx
    )


    if sel != st.session_state.current_step:

        st.session_state.current_step = sel

        st.rerun()


# ============================================================
# 11. ANA EKRAN
# ============================================================

st.title(
    "🎯 Algoritmik Problem Çözme Rehberi"
)

st.write(
    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)


# ============================================================
# 12. SORU FOTOĞRAFI
# ============================================================

if st.session_state.uploaded_file_data is None:

    up = st.file_uploader(
        "📷 Soru Fotoğrafı Yükle",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if up:

        st.session_state.uploaded_file_data = up

        st.rerun()

else:

    st.image(
        st.session_state.uploaded_file_data,
        width=450
    )


    if st.button(
        "❌ Soruyu Değiştir"
    ):

        st.session_state.uploaded_file_data = None

        # Eski sohbetleri temizlemiyoruz.
        # İstersen burada temizleyebiliriz.

        st.rerun()


st.divider()


# ============================================================
# 13. İKİ SÜTUN
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
            "canvas_v40_"
            + st.session_state
            .current_step
            .replace(" ", "_")
        )
    )


    # --------------------------------------------------------
    # TASARIM KAYDET
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
                    st.session_state
                    .current_step,

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
    # METABİLİŞ
    # --------------------------------------------------------

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
            + st.session_state
            .current_step[0]
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
                st.session_state
                .current_step,

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
            "slider_"
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
                st.session_state
                .current_step,

            "tip":
                "Eminlik",

            "icerik":
                confidence
        })

        st.success(
            f"Eminlik: {confidence}"
        )


    st.write("---")


    # ========================================================
    # FİNAL ÖZETİ
    # ========================================================

    if st.button(
        "🏁 Çözümü Bitir ve Özetini Al"
    ):

        with st.spinner(
            "Süreç analiz ediliyor..."
        ):

            try:

                hist_full = ""

                for stage, messages in (
                    st.session_state
                    .chat_storage
                    .items()
                ):

                    hist_full += (
                        f"\n\n### {stage}\n"
                    )

                    for m in messages:

                        hist_full += (
                            f"{m['role']}: "
                            f"{m['content']}\n"
                        )


                final_prompt = f"""

Sen bir ortaokul matematik öğretmenisin.

Aşağıdaki öğrencinin algoritmik problem çözme sürecini
incele.

Öğrencinin çözümünü yeniden çözme.

Cevabı söyleme.

Bunun yerine öğrencinin:
- problemi ayrıştırma,
- matematiksel yapıyı soyutlama,
- çözüm planı oluşturma,
- çözümünü kontrol etme

süreçlerini kısa ve pedagojik biçimde özetle.

Öğrenciyi değerlendiren öğretmen dili kullan.

Öğrencinin güçlü yönlerini ve geliştirebileceği yönleri
kısaca belirt.

Öğrenci:
{student_id}

Süreç:
{hist_full}
"""


                payload = {

                    "contents": [

                        {

                            "role": "user",

                            "parts": [

                                {
                                    "text":
                                        final_prompt
                                }

                            ]
                        }

                    ],

                    "generationConfig": {

                        "temperature": 0.3,

                        "topP": 0.8,

                        "maxOutputTokens": 500
                    }
                }


                r_final = requests.post(

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


                if r_final.status_code != 200:

                    raise Exception(
                        r_final.text
                    )


                final_data = (
                    r_final.json()
                )


                final_text = (
                    final_data
                    ["candidates"][0]
                    ["content"]["parts"][0]
                    ["text"]
                )


                st.info(
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
# 14. SAĞ SÜTUN – REHBER BOT
# ============================================================

with col2:

    st.write(
        "💬 **Rehber Bot**"
    )


    chat_container = st.container(
        height=550
    )


    # --------------------------------------------------------
    # GEÇMİŞ MESAJLAR
    # --------------------------------------------------------

    for m in (
        st.session_state
        .chat_storage[
            st.session_state.current_step
        ]
    ):

        chat_container.chat_message(
            m["role"]
        ).write(
            m["content"]
        )


    # --------------------------------------------------------
    # ÖĞRENCİ MESAJI
    # --------------------------------------------------------

    p = st.chat_input(
        "Düşünceni veya sorunu buraya yaz..."
    )


    if p:

        # ----------------------------------------------------
        # Öğrenciyi kaydet
        # ----------------------------------------------------

        st.session_state \
            .chat_storage[
                st.session_state.current_step
            ].append({

                "role":
                    "user",

                "content":
                    p
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
                st.session_state.current_step,

            "tip":
                "Öğrenci",

            "icerik":
                p
        })


        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        with st.spinner(
            "Rehber Bot düşünüyor..."
        ):

            try:

                ans = gemini_cevap_al(

                    current_step=
                        st.session_state
                        .current_step,

                    chat_history=
                        st.session_state
                        .chat_storage[
                            st.session_state
                            .current_step
                        ],

                    image_file=
                        st.session_state
                        .uploaded_file_data
                )


                # ------------------------------------------------
                # BOT CEVABINI KAYDET
                # ------------------------------------------------

                st.session_state \
                    .chat_storage[
                        st.session_state
                        .current_step
                    ].append({

                        "role":
                            "assistant",

                        "content":
                            ans
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
                        st.session_state.current_step,

                    "tip":
                        "Bot",

                    "icerik":
                        ans
                })


                st.rerun()


            except Exception as e:

                st.error(
                    "Gemini yanıt oluşturamadı."
                )

                st.code(
                    str(e)
                )
