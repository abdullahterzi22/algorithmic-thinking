import streamlit as st
import requests
import pandas as pd
import os
import base64
import io
import json
import re
import difflib
import unicodedata
from datetime import datetime
from PIL import Image
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
APP_VERSION = "v31"

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
Öğrencinin problemi yalnızca düşünme gereksinimleri bakımından
anlamlı parçalara ayırmasını sağlamak.

AYRIŞTIRMA'NIN TEMEL ZİNCİRİ:
İSTENEN → BUNU BULMAK İÇİN GEREKEN BİLGİ →
BU BİLGİYİ BULMAK İÇİN GEREKEN BİLGİ → VERİLEN/TEMEL BİLGİ

Öğrencinin:
- Önce neyin istendiğini belirlemesini,
- İstenen şeyi belirlemek için hangi bilgiye ihtiyaç olduğunu
  kendisinin fark etmesini,
- Gerekli bir bilgi eksikse o bilgiye ulaşmak için hangi
  bilgiye ihtiyaç olduğunu kendisinin düşünmesini,
- Elindeki verilen bilgilerin bu zincirde hangi ihtiyacı
  karşıladığını fark etmesini sağla.

ÇOK ÖNEMLİ:
Ayrıştırma sırasında öğrencinin henüz söylemediği bir veri
kaynağını, niceliği, değeri, süreyi, hızı, işlemi, matematiksel
ilişkiyi, grafik özelliğini veya çözüm adımını sen ortaya çıkarma.
Öğrenciyi B aracına, hıza, süreye, belirli bir grafik noktasına,
eksen kesişimine veya herhangi bir hesaplamaya kendin yönlendirme.

Yasak:
- Çözüm yolu söylemek.
- İşlem önermek.
- Formül vermek.
- Denklem vermek.
- Doğrusal ilişkiyi doğrudan söylemek.
- Cevabı söylemek.
- Hız, süre, değişim miktarı veya başka bir çözüm niceliğini
  öğrencinin yerine ortaya çıkarmak.
- Öğrencinin yerine problemi parçalara ayırmak.
