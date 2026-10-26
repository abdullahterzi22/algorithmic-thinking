import streamlit as st
import requests
import pandas as pd
import os
import base64
import io
import json
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

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)


DATA_FILE = "tez_verileri_final.csv"


# ============================================================
# TEK KAYNAK: DOĞRUSAL İLİŞKİLER İÇİN BASAMAK TALİMATLARI
# ============================================================

BASAMAK_TALIMATLARI = {

    "1. Ayrıştırma": """

Amaç:
Problemi oluşturan bileşenleri, verilenleri, koşulları ve alt problemleri
belirleme; problemi yönetilebilir parçalara ayırma.

Öğrencinin:
- Problemde ne istendiğini belirlemesini,
- Verilen bilgileri, koşulları ve gerekli bileşenleri fark etmesini,
- Problemi anlamlı ve yönetilebilir parçalara ayırmasını,
- Bu parçaların birbirleriyle nasıl bağlantılı olduğunu görmeye başlamasını sağla.
- Öğrenci takıldığında aşamaya özgü küçük bir ipucu verebilirsin.

Bu aşamada ayrıştırma, problemi çözmek değildir. Öğrencinin problemi oluşturan
bileşenleri görünür hâle getirmesine yardımcı ol.

İLERLEME PROTOKOLÜ:
- İstenen → istenene ulaşmak için gereken bilgi → bu bilgiyi bulmak için gereken bilgi → verilen/temel bilgi zincirini geriye doğru izle.
- Öğrencinin açıkça belirttiği bilgi veya niceliği belirlenmiş kabul et; aynı bilgiyi farklı sözlerle yeniden sorma.
- Öğrenci bir bilginin eksik olduğunu söylüyorsa bunu fark etmemiş gibi davranma ve görselden onun yerine tamamlamaya çalışma.
- Öğrenci gerekli bir bilgiyi zaten belirttiyse bir sonraki adımda o bilginin rolünü/yeterliliğini ya da zincirde geriye doğru hangi bilgiye ihtiyaç olduğunu düşündür; çözüm yöntemine geçme.
- Öğrencinin fark ettiği eksen, nokta, çizgi, renk, kesişim veya grafik özelliğini tekrar tarif etme.
- İlk mesajda da bu aşamanın sınırını koru; çözüm, hesaplama veya matematiksel ilişki aramaya geçme.

Yasak:
- Çözüm yolu söylemek.
- İşlem önermek.
- Formül veya denklem vermek.
- Doğrusal ilişkiyi doğrudan söylemek.
- Cevabı söylemek.
- Problemi öğrencinin yerine parçalara ayırmak.
- Öğrenciyi çözüm yöntemine yönlendirmek.

""",

    "2. Soyutlama": """

Amaç:
Ayrıştırma aşamasında belirlenen bilgiler arasından çözüm açısından belirleyici
olanları seçmek; gereksiz ayrıntıları elemek ve problemdeki temel ilişki,
yapı veya örüntüyü öğrencinin kendisinin fark etmesine yardımcı olmak.

Öğrencinin:
- Hangi bilgilerin çözüm açısından önemli olduğunu ve neden önemli olduğunu
  fark etmesini,
- Verilenler ve koşullar arasındaki belirleyici ilişkiyi veya yapıyı fark etmesini,
- Bir bilginin başka bir bilgiyi anlamaya ya da elde etmeye nasıl katkı
  sağlayabileceğini görmesini,
- Gereksiz veya ikincil bilgileri ayırt etmesini,
- Uygun olduğunda örüntü, tekrar eden yapı veya daha genel bir temsil
  oluşturmasını sağla.
- Öğrenci takıldığında aşamaya özgü, küçük ve giderek açılan ipuçları verebilirsin.

İLERLEME REHBERİ:
- Önce öğrencinin belirlediği bilgileri dikkate al; daha önce açıkça söylediği
  bilgileri yeniden buldurmaya çalışma.
- Öğrencinin "hangi bilgi önemli?" sorusundan "neden önemli?" sorusuna
  geçmesine yardımcı ol.
- Önemli bir bilgi ile problemdeki bir koşul arasında bağlantı kurulup
  kurulamayacağını düşündür.
- İki veya daha fazla bilgi arasında bir ilişki, karşılaştırma, bağımlılık,
  düzen, örüntü, değişim ya da ortak yapı olup olmadığını fark ettirmeye çalış.
- Bir bilginin başka bir bilgiyi açıklamak, belirlemek veya elde etmek için
  nasıl kullanılabileceğini öğrencinin kendisinin keşfetmesine rehberlik et.
- Problemde gereksiz, ikincil veya çözüm açısından etkisiz görünen bilgiler
  varsa bunları ayırt etmesini sağla; öğrencinin gerekçesini önemse.
- Uygun olduğunda farklı değerler, durumlar veya gösterimler arasında aynı
  yapının korunup korunmadığını düşündür.
- Öğrenci bir örüntü veya genel yapı fark ettiğinde bunu tekrar tekrar
  buldurma; bu yapının problemdeki diğer bilgilerle nasıl bağlantılı olduğunu
  düşünmesine yardımcı ol.
- Öğrenci bir ilişkiyi kendisi ifade ettiğinde ilişkiyi onun yerine yeniden
  adlandırma veya genişletme; yalnızca bir sonraki soyutlama adımına
  ilerlemesini destekle.
- Öğrenci hesaplama veya işlem sırası kurmaya başladıysa onu işlemi sürdürmeye
  değil, kullandığı bilginin ve ilişkinin neden gerekli olduğunu açıklamaya
  yönlendir.
- Bu aşamada amaç **hangi işlemi hangi sırayla yapacağını belirlemek değil,
  hangi bilginin neden işe yaradığını ve bilgiler arasındaki temel yapının
  ne olduğunu anlamaktır**. Sayısal hesaplamaları, işlem sırasını ve çözüm
  planını **Algoritma Tasarımı** aşamasına bırak.
- Soyutlama rehberini belirli bir problem türüne ait ilişkiye bağlama.
  Rehber; grafik, tablo, şekil, oran, sayı, ölçüm, örüntü, hareket veya başka
  bir matematiksel bağlamdaki farklı problemlerde uygulanabilir olmalıdır.

Yasak:
- "Bu doğrusal ilişkidir." demek.
- Kilit ilişkiyi, yapıyı veya örüntüyü öğrencinin yerine doğrudan adlandırmak.
- Öğrencinin fark etmediği bir bağı hazır olarak söylemek.
- Sayısal hesaplama yapmak veya işlem sonucu vermek.
- Denklem veya formül vermek.
- Eğim ya da sabit değişimi doğrudan söylemek.
- Kullanılacak yöntemi, işlemi veya işlem sırasını söylemek.
- Çözüm planını öğrencinin yerine oluşturmak.
- Çözüm vermek.
- Öğrencinin yerine ilişki veya örüntü oluşturmak.

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

""",


}


# ============================================================
# ALGORİTMİK DÜŞÜNME ALT BECERİLERİ
# ============================================================

ALT_BECERILER = {
    "1. Ayrıştırma": [
        ("A1", "Alt amacı belirleme"),
        ("A2", "Parçalara ayırma"),
        ("A3", "İlişkileri belirleme"),
    ],
    "2. Soyutlama": [
        ("S1", "İlgili bilgiyi belirleme"),
        ("S2", "Gereksiz bilgiyi ayırt etme"),
        ("S3", "Nicelikler arası ilişkileri belirleme"),
        ("S4", "Uygun temsil oluşturma"),
        ("S5", "Örüntü ve genel yapı oluşturma"),
    ],
    "3. Algoritma Tasarımı": [
        ("AT1", "İşlem adımlarını belirleme"),
        ("AT2", "Adımları sıralama"),
        ("AT3", "Ara sonuçları kullanma"),
        ("AT4", "Prosedür oluşturma"),
        ("AT5", "Uygun alternatif yolu değerlendirme"),
    ],
    "4. Hata Ayıklama": [
        ("HA1", "Çözümü test etme"),
        ("HA2", "Koşullarla karşılaştırma"),
        ("HA3", "Tutarsızlığı fark etme"),
        ("HA4", "Sorunlu adımı belirleme"),
        ("HA5", "Düzeltme yapma"),
        ("HA6", "Yeniden test etme"),
    ],
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
YÖNLENDİRİRSİN. Öğrenci takıldığında bir öğretmen gibi ona ipucu verebilirsin. Bu ipucu bulunulan aşamaya yönelik olsun. Mesela ayrıştırma aşamasında öğrenciyi algoritma tasarımına götürmesin.

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

16. Her yanıtta en fazla bir yönlendirici soru kullan. Soru sormak zorunda değilsin; gerektiğinde tek bir kısa yönlendirici cümle kur.

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

25. ÖĞRENCİ TAKILDIĞINDA DESTEK VER:
    Öğrenci açıkça takıldığını belirtiyorsa mevcut basamağın talimatlarında belirtilen
    sınırlar içinde kısa ve somut bir ipucu ver. Gerekirse ardından tek bir soru sor.
    İpucu cevap, sonuç, formül, denklem, yöntem veya çözüm adımını öğrencinin yerine
    vermemeli; öğrencinin daha önce söylediği bilgiyi de gereksiz yere tekrarlamamalıdır.

26. BASAMAK SINIRLARI:
    Her basamağın amacı, izin verilen rehberlik ve yasakları yalnızca BASAMAK TALİMATLARI
    bölümündeki ilgili basamak protokolünden al. Bu genel kuralları, basamağa özgü
    talimatların yerine geçecek şekilde yorumlama.

============================================================
YANIT BİÇİMİ
============================================================

- Türkçe yaz.
- 1 veya 2 kısa cümle kullan.
- Soru cümlesi olmak zorunda değil kurduğun cümle.
- Eğer soru soracaksan YALNIZCA BİR soru sor  
- Soruyu öğrencinin son cevabına göre oluştur.
- Öğrencinin verdiği her cevabı dikkate al.
- Öğrencinin önceki verdiği cevapları unutma.
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

def image_part_from_canvas(image_data):

    if image_data is None:
        return None

    try:
        if hasattr(image_data, "save"):
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            raw = buffer.getvalue()
        else:
            image = Image.fromarray(image_data)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()

        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(raw).decode("utf-8")
            }
        }
    except Exception:
        return None


def safe_get_canvas_image_data(canvas_result):

    if canvas_result is None:
        return None

    try:
        return canvas_result.image_data
    except Exception:
        return None


def gemini_generate(
    parts,
    temperature=0.2,
    max_tokens=250,
    include_problem_image=False,
    include_flowchart_image=False,
    include_annotation_image=False
):

    final_parts = []
    final_parts.extend(parts)

    if include_problem_image:
        image_part = get_problem_image_part()
        if image_part is not None:
            final_parts.append(image_part)

    # Öğrencinin problem üzerinde yaptığı işaretlemeyi de Gemini'ye gönder.
    # Bu görsel, problem arka planı + öğrencinin çizimlerini birlikte içerir.
    if include_annotation_image:
        annotation_image = st.session_state.get("annotation_image_data")
        image_part = image_part_from_canvas(annotation_image)
        if image_part is not None:
            final_parts.append(image_part)

    if include_flowchart_image:
        flowchart_image = st.session_state.get("flowchart_image_data")
        image_part = image_part_from_canvas(flowchart_image)
        if image_part is not None:
            final_parts.append(image_part)

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
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise Exception(
            f"Gemini API Hatası ({response.status_code}): {detail}"
        )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise Exception(f"Gemini beklenmeyen yanıt verdi: {data}")


def parse_guidance_response(raw_text, current_step):
    """Gemini'nin yapılandırılmış yanıtını güvenli biçimde çözer."""

    fallback_skill = ALT_BECERILER[current_step][0][0]
    fallback_question = raw_text.strip()

    # Markdown JSON çevresini temizle
    cleaned = fallback_question.replace("```json", "").replace("```", "").strip()

    # Model bazen JSON'un önüne/arkasına kısa bir ifade ekleyebilir.
    # Bu durumda yalnızca ilk JSON nesnesini almaya çalış.
    if not cleaned.startswith("{"):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

    try:
        data = json.loads(cleaned)
        question = str(data.get("soru", "")).strip()
        alt_beceri = str(data.get("alt_beceri", fallback_skill)).strip()
        completed = bool(data.get("basamak_tamamlandi", False))

        if not question:
            raise ValueError("soru alanı boş")

        valid_codes = {code for code, _ in ALT_BECERILER[current_step]}
        if alt_beceri not in valid_codes:
            alt_beceri = fallback_skill

        return {
            "soru": question,
            "alt_beceri": alt_beceri,
            "basamak_tamamlandi": completed
        }
    except Exception:
        return {
            "soru": fallback_question,
            "alt_beceri": fallback_skill,
            "basamak_tamamlandi": False
        }


# ============================================================
# PROBLEM GÖRSELİNİ ANALİZ ET
# ============================================================

def analyze_problem_image(uploaded_file):
    """Öğretmenin test amacıyla, problemi öğrenci gibi çözümleyip çözme sürecini görünür kılar."""

    if uploaded_file is None:
        return ""

    image_data = base64.b64encode(
        uploaded_file.getvalue()
    ).decode("utf-8")

    prompt = """
Sen ortaokul düzeyinde bir matematik öğrencisisin. Sana bir matematik
probleminin görseli veriliyor. Bu görevde öğretmen için test amacıyla,
problemi bir öğrenci gibi kendin çözmeni ve nasıl düşündüğünü görünür
biçimde yazmanı istiyorum. Bu çıktı öğrenci rehber botunun yanıtı değil;
öğretmenin botun olası hatalarını incelemesi için hazırlanmış ayrı bir
problem çözme örneğidir.

TEMEL GÖREVİN
- Problemi dikkatlice oku ve görseldeki metni, sayıları, birimleri,
  tabloyu, grafiği, eksenleri, lejantı ve şekilleri incele.
- Problemi gerçekten çöz. Yalnızca çözümde yapılabilecekleri sıralama;
  kendi çözüm düşünceni, kullandığın matematiksel ilişkileri, işlemleri
  ve ulaştığın sonucu açıkça göster.
- Çözüm sürecini aşağıdaki dört algoritmik düşünme basamağı altında yaz.
- Her basamağı gerçek problemdeki gözlem ve verilerle doldur; ancak basamağın
  düşünme mantığını tek bir problem türüne ait hazır şablona indirgeme.
  Özellikle Soyutlama bölümünde, görselde gerçekten bulunan ilişkiyi kullan;
  başka problemlerde geçerli olmayan hazır bir ilişkiyi genelleştirme.
- Ortaokul öğrencisinin anlayabileceği doğal birinci tekil şahıs dili
  kullan: “Önce ... fark ettim”, “Buradan ... anlıyorum”, “Şimdi ...” gibi.
- Gerektiğinde ara işlemleri yaz ve her işlemin nedenini kısaca açıkla.
- Çözümün sonunda sonucu problem bağlamında ifade et ve kontrol et.

DOĞRULUK VE BELİRSİZLİK KURALLARI
- Görselde olmayan bilgi uydurma; okunamayan değerleri tahmin ederek
  kesinmiş gibi kullanma.
- Bir değer okunamıyorsa bunu açıkça belirt. Çözüm ancak o değer
  bilinmeden yapılabiliyorsa ilerle; yapılamıyorsa hangi bilginin eksik
  olduğunu ve bunun çözümü neden engellediğini açıkla.
- Grafik veya tabloda bir noktayı yorumlarken hangi çizgiye/nesneye ait
  olduğunu ve koordinatların hangi sırayla okunduğunu dikkatle doğrula.
- Verilenlerle çelişen bir sonuç elde edersen çelişkiyi gizleme; hangi
  varsayımın veya görsel okumasının belirsiz olduğunu belirt.
- Bir çözüm yolunu gereğinden fazla karmaşıklaştırma. Ancak işlemleri
  atlayıp doğrudan cevaba da geçme.
- Çözümün doğru olduğunu varsayma: işlemleri yeniden gözden geçir ve
  olası hata noktalarını ayrı bölümde dürüstçe yaz.

## 1. Ayrıştırma — Problemi anlama
Öğrenci olarak:
- Problemin benden ne istediğini kendi cümlelerinle yaz.
- Verilen bilgileri, koşulları ve önemli nicelikleri belirle.
- Grafik/tablo/şekil varsa, önemli değerleri ve bunların hangi nesneye
  ait olduğunu açıkça yaz.
- Gerekliyse problemdeki alt görevleri kendi anlayışına göre belirt.
Bu bölümde henüz hesaplamayı bitirme; ancak problemi anlamak için
verileri somut ve doğru biçimde kaydet.

## 2. Soyutlama — Belirleyici bilgiyi ve temel yapıyı fark etme
- Ayrıştırma bölümünde belirlediğin bilgiler arasından çözüm açısından
  belirleyici olanları seç ve neden önemli olduklarını açıkla.
- Verilenler ile problemdeki koşullar arasındaki temel ilişkiyi, bağımlılığı,
  karşılaştırmayı, değişimi, düzeni, örüntüyü veya ortak yapıyı belirle.
- Bir bilginin başka bir bilgiyi açıklamaya, belirlemeye veya elde etmeye
  nasıl katkı sağladığını göster.
- Gereksiz veya ikincil bilgileri, gerçekten problem açısından gereksiz
  olduğundan emin olduğun ölçüde ayırt et.
- Uygunsa farklı değerler, durumlar veya gösterimler arasında korunan ortak
  yapıyı ya da örüntüyü belirt.
- Bu bölümü belirli bir problem türüne ait hazır bir ilişkiyi otomatik
  uygulayarak yazma. Görseldeki problem hangi yapıyı destekliyorsa onu
  belirle; desteklemiyorsa başka bir ilişki uydurma.
- Buradaki amaç çözümde kullanılacak işlemleri yapmak değil, çözümün
  dayanacağı matematiksel yapıyı ve gerekli bilgilerin neden gerekli olduğunu
  açıklamaktır. Sayısal hesaplamaları, işlem sırasını ve çözüm planını
  **Algoritma Tasarımı** bölümüne bırak.
- Gereksiz olduğu kesin olmayan bir bilgiyi kesin biçimde gereksiz ilan etme.

## 3. Algoritma Tasarımı — Kendi çözümümü oluşturma
- Çözüm için seçtiğin yolu adım adım uygula.
- Her adımda hangi bilgiyi kullandığını ve neden o işlemi yaptığını yaz.
- Gerekli matematiksel işlemleri, hesaplamaları, eşitlikleri veya
  denklemleri göster; bunları açıklamasız biçimde sıralama.
- Birden fazla makul çözüm yolu varsa birini seç ve neden uygun olduğunu
  kısaca belirt. Görseldeki veriler birden fazla yoruma açıksa bunu söyle.
- Bu bölümün sonunda bulduğun cevabı birimiyle ve problem bağlamında yaz.

## 4. Hata Ayıklama — Çözümümü kontrol etme
Çözümünü bitirdikten sonra gerçekten kontrol et:
- Grafik, tablo ve problem metninden okuduğun değerleri yeniden karşılaştır.
- İşlemlerini ve aritmetik hesaplamalarını gözden geçir.
- Birimleri ve sonucun soruda istenen niceliğe karşılık gelip gelmediğini
  kontrol et.
- Sonucun hareketin yönü, değişim biçimi veya diğer koşullarla tutarlı
  olup olmadığını değerlendir.
- Bir hata ya da belirsizlik fark edersen açıkça belirt ve gerekiyorsa
  çözümünü düzelt. Sorun görmüyorsan hangi kontrolün bunu desteklediğini
  yaz; kontrol etmediğin bir şeyi kontrol etmiş gibi gösterme.

## Öğretmen için: Rehber botta izlenmesi gereken olası hata noktaları
Öğrenci çözümünden ayrı, kısa bir son bölüm ekle. Bu bölümde yalnızca
bu problem özelinde botun öğrenciye rehberlik ederken yanlış yapabileceği
somut noktaları belirt. Örneğin yanlış grafik çizgisini seçme, eksenleri
karıştırma, gerekli matematiksel ilişkiyi erken söyleme, çözümü öğrenci
yerine yapma, belirsiz görsel bilgisini kesin kabul etme veya basamak
hedefinden sapma gibi risklerden yalnızca gerçekten ilgili olanları yaz.
Genel ve her probleme uyabilecek maddeler üretme.

ÇIKTI BİÇİMİ
1. Ayrıştırma
2. Soyutlama
3. Algoritma Tasarımı
4. Hata Ayıklama
5. Öğretmen için: Rehber botta izlenmesi gereken olası hata noktaları

Başlıkların altında anlaşılır maddeler ve gerekli matematiksel işlemler
bulunsun. Sadece yapılabilecekleri anlatan genel öneriler yazma: görseldeki
problemi gerçekten çöz ve çözümünü tamamla.
"""

    parts = [
        {"text": prompt},
        {
            "inline_data": {
                "mime_type": uploaded_file.type,
                "data": image_data
            }
        }
    ]

    return gemini_generate(
        parts,
        temperature=0.1,
        max_tokens=1800,
        include_problem_image=False
    )

# ============================================================
# İLK SORUYU OLUŞTUR
# ============================================================

def conversation_text(chat_history):

    text = ""
    for message in chat_history:
        role = "ÖĞRENCİ" if message["role"] == "user" else "REHBER"
        text += f"{role}: {message['content']}\n"
    return text


def previous_stages_text(current_step):

    stages = list(BASAMAK_TALIMATLARI.keys())
    current_index = stages.index(current_step)
    text = ""

    for stage in stages[:current_index]:
        history = st.session_state.chat_storage.get(stage, [])
        if history:
            text += f"\n===== {stage} =====\n"
            text += conversation_text(history)

    return text


def ilk_soruyu_olustur(current_step, problem_analysis):

    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAĞIN AKADEMİK AMACI:
{BASAMAK_TALIMATLARI[current_step]}

ALT BECERİLER:
{ALT_BECERILER[current_step]}

PROBLEM ANALİZİ:
{problem_analysis}

ÖNEMLİ:
Yüklenen gerçek problem görselini incele.
İlk soru yüklenen gerçek problemdeki bilgi, koşul veya gösterimlerden
hareket etmelidir; fakat sorunun düşünme amacı basamak protokolündeki
genellenebilir ilerleme rehberine dayanmalıdır. Gerçek probleme özgü ayrıntıyı
kullan, ancak tek bir problem türüne ait hazır ilişkiyi öğrenciye dayatma.

YANITI YALNIZCA ŞU JSON YAPISINDA VER:
{{
  "alt_beceri": "...",
  "basamak_tamamlandi": false,
  "soru": "..."
}}

İlk soru için basamak_tamamlandi her zaman false olsun.
Soru alanında 1-2 kısa cümle bulunsun; soru sormak zorunlu değildir ve en fazla bir soru kullanılmalıdır.
"""

    raw = gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=150,
        include_problem_image=True,
        include_annotation_image=True,
        include_flowchart_image=(current_step in ["3. Algoritma Tasarımı", "4. Hata Ayıklama"])
    )

    return parse_guidance_response(raw, current_step)


def ogrenciye_cevap_ver(current_step, problem_analysis, chat_history, student_message):

    conversation = conversation_text(chat_history)
    previous = previous_stages_text(current_step)

    stuck_phrases = [
        "takıldım", "bilmiyorum", "yapamadım", "anlamadım",
        "nasıl yapacağım", "nereden başlayacağımı bilmiyorum",
        "yardım et", "yardım lazım", "çözemiyorum", "olmuyor",
        "ne yapacağımı bilmiyorum"
    ]
    student_lower = student_message.lower()
    student_is_stuck = any(phrase in student_lower for phrase in stuck_phrases)

    if student_is_stuck:
        stuck_instruction = """
ÖĞRENCİ TAKILDIĞINI BELİRTTİ. Mevcut basamağın talimatlarıyla uyumlu, kısa ve somut
bir ipucu ver. Ardından gerekliyse yalnızca bir soru sor; soru sormak zorunlu değildir.
İpucu cevabı, işlemi, formülü, denklemi, çözüm yöntemini veya bir sonraki çözüm adımını
öğrencinin yerine söylememelidir.
"""
    else:
        stuck_instruction = """
ÖĞRENCİ TAKILDIĞINI BELİRTMİYOR. Normal rehberlik akışını sürdür. Gereksiz ipucu verme.
"""

    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAK PROTOKOLÜ:
{BASAMAK_TALIMATLARI[current_step]}

ALT BECERİLER:
{ALT_BECERILER[current_step]}

{stuck_instruction}

PROBLEM ANALİZİ:
{problem_analysis}

ÖNCEKİ BASAMAKLARIN DİYALOGLARI:
{previous}

MEVCUT BASAMAK DİYALOĞU:
{conversation}

ÖĞRENCİNİN SON MESAJI:
{student_message}

GÖREV:
Yüklenen gerçek problem görselini incele.
Öğrencinin son mesajını önce anlamını koruyarak değerlendir; öğrencinin söylediği bilgiyi kendi varsayımınla değiştirme.
Mevcut basamağın amacını, diğer bütün genel kurallardan ÖNCELE.
Bir sonraki düşünme adımını destekle. Soru sormak zorunda değilsin; soru sorarsan yalnızca TEK soru kullan.

BASAMAK SINIRI:
Mevcut basamağın hedefini ve özel sınırlarını BASAMAK TALİMATLARI bölümündeki
ilgili protokole göre uygula. Öğrencinin söylemediği sonucu onun adına çıkarma.
Çözüm, formül, denklem, işlem veya cevap verme.

YANITI YALNIZCA ŞU JSON YAPISINDA VER:
{{
  "alt_beceri": "...",
  "basamak_tamamlandi": false,
  "soru": "..."
}}

Basamak tamamlanmadıysa false kullan.
Basamak tamamlandıysa true kullan; bunu yalnızca konuşmada gerçekten yeterli kanıt varsa yap.
Soru alanında yalnızca öğrencinin göreceği 1-2 kısa cümle olsun; en fazla bir soru kullan ve soru sormak zorunda hissetme.
"""

    raw = gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=180,
        include_problem_image=True,
        include_annotation_image=True,
        include_flowchart_image=(current_step in ["3. Algoritma Tasarımı", "4. Hata Ayıklama"])
    )

    return parse_guidance_response(raw, current_step)


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

        include_problem_image=True,
        include_annotation_image=True
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


if "annotation_reset" not in st.session_state:

    st.session_state.annotation_reset = 0


if "annotation_json" not in st.session_state:

    st.session_state.annotation_json = None


if "annotation_image_data" not in st.session_state:

    st.session_state.annotation_image_data = None


if "flowchart_storage" not in st.session_state:

    st.session_state.flowchart_storage = {}


if "flowchart_image_data" not in st.session_state:

    st.session_state.flowchart_image_data = None


if "last_alt_beceri" not in st.session_state:

    st.session_state.last_alt_beceri = {}


if "completed_stages" not in st.session_state:

    st.session_state.completed_stages = []


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
    "🎯 Algoritmik Problem Çözme Rehberi · v22"
)


st.write(

    f"### Mevcut Basamak: "
    f"{st.session_state.current_step}"
)



# ============================================================
# PROBLEM GÖRSELİNİ ÇİZİM ALANINA HAZIRLAMA
# ============================================================

def get_annotation_background():
    """Yüklenen problem görselini öğrenci işaretleme alanına hazırlar."""
    uploaded_file = st.session_state.get("uploaded_file_data")

    if uploaded_file is None:
        return None, None, None

    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        # Problem görselini ekranda yaklaşık 900 px genişliğe
        # sığdırırken en-boy oranını koru.
        max_width = 900
        original_width, original_height = image.size

        display_width = min(
            original_width,
            max_width
        )

        display_height = max(
            1,
            int(
                original_height
                * display_width
                / original_width
            )
        )

        image = image.resize(
            (display_width, display_height),
            Image.LANCZOS
        )

        return image, display_width, display_height

    except Exception:
        return None, None, None


# ============================================================
# PROBLEM YÜKLEME
# ============================================================

if (
    st.session_state.uploaded_file_data
    is None
):

    uploaded = st.file_uploader(
        "📷 Soru Fotoğrafı Yükle",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        st.session_state.uploaded_file_data = uploaded

        with st.spinner("Problem analiz ediliyor..."):
            try:
                st.session_state.problem_analysis = analyze_problem_image(uploaded)
            except Exception as e:
                st.error("Problem analiz edilemedi.")
                st.code(str(e))
                st.session_state.problem_analysis = ""

        st.session_state.chat_storage = {
            stage: [] for stage in BASAMAK_TALIMATLARI
        }
        st.session_state.current_step = "1. Ayrıştırma"
        st.session_state.annotation_reset += 1
        st.session_state.annotation_json = None
        st.session_state.annotation_image_data = None
        st.session_state.flowchart_storage = {}
        st.session_state.flowchart_image_data = None
        st.session_state.completed_stages = []
        st.session_state.last_alt_beceri = {}
        st.rerun()

else:
    if st.button("❌ Soruyu Değiştir"):
        st.session_state.uploaded_file_data = None
        st.session_state.problem_analysis = None
        st.session_state.chat_storage = {
            stage: [] for stage in BASAMAK_TALIMATLARI
        }
        st.session_state.annotation_reset += 1
        st.session_state.annotation_json = None
        st.session_state.annotation_image_data = None
        st.session_state.flowchart_storage = {}
        st.session_state.flowchart_image_data = None
        st.session_state.completed_stages = []
        st.session_state.last_alt_beceri = {}
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

    # --------------------------------------------------------
    # PROBLEM ÜZERİNDE ÖĞRENCİ İŞAREMLEME ALANI (v31 yerleşimi)
    # --------------------------------------------------------
    annotation_image, annotation_width, annotation_height = get_annotation_background()

    if annotation_image is not None:
        st.write("🖍️ **Problem Üzerinde İşaretleme**")
        st.caption("Problemin üzerine işaretleme yapabilirsin. Bot, işaretlediğin alanı problem bağlamında kullanabilir.")

        annotation_col1, annotation_col2 = st.columns([3, 1])
        with annotation_col1:
            annotation_tool_map = {
                "Serbest Çizim": "freedraw",
                "Ok / Çizgi": "line",
                "Dikdörtgen": "rect",
                "Elips": "circle"
            }
            annotation_tool = st.selectbox(
                "İşaretleme aracı:",
                list(annotation_tool_map.keys()),
                key="annotation_tool"
            )
        with annotation_col2:
            annotation_color = st.color_picker(
                "İşaretleme rengi:",
                "#FF0000",
                key="annotation_color"
            )

        annotation_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_color=annotation_color,
            stroke_width=3,
            background_image=annotation_image,
            height=annotation_height,
            width=annotation_width,
            drawing_mode=annotation_tool_map[annotation_tool],
            update_streamlit=False,
            initial_drawing=st.session_state.annotation_json,
            key="problem_annotation_" + str(st.session_state.annotation_reset)
        )

        try:
            if annotation_result is not None and annotation_result.json_data:
                st.session_state.annotation_json = annotation_result.json_data
                current_annotation_image = safe_get_canvas_image_data(annotation_result)
                if current_annotation_image is not None:
                    st.session_state.annotation_image_data = current_annotation_image
        except Exception:
            pass

        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("💾 İşaretlemeyi Kaydet", key="save_problem_annotation"):
                if annotation_result is not None and annotation_result.json_data:
                    log_kaydet({
                        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "id": student_id,
                        "basamak": st.session_state.current_step,
                        "tip": "Problem Üzeri Çizim",
                        "icerik": str(annotation_result.json_data)
                    })
                    st.success("İşaretleme kaydedildi.")
                else:
                    st.info("Önce problem üzerinde bir işaretleme yap.")
        with clear_col:
            if st.button("🧹 İşaretlemeyi Temizle", key="clear_problem_annotation"):
                st.session_state.annotation_reset += 1
                st.session_state.annotation_json = None
                st.session_state.annotation_image_data = None
                st.rerun()

        st.divider()

    st.write(
        "🖼️ **Tasarım ve Planlama Alanı**"
    )



    canvas_result = st_canvas(

        fill_color=fill_color,

        stroke_color=stroke_color,

        stroke_width=3,

        background_color="#ffffff",

        height=450,

        drawing_mode=drawing_mode,

        # Çizim yapılırken Streamlit'in sürekli yeniden çalışmasını önler.
        # Böylece öğrencinin oluşturduğu tasarım yanıp sönmez.
        update_streamlit=False,

        initial_drawing=st.session_state.flowchart_storage.get(
            st.session_state.current_step
        ),

        key=(
            "canvas_"
            +
            st.session_state.current_step
            .replace(" ", "_")
        )
    )

    try:
        if canvas_result is not None and canvas_result.json_data:
            st.session_state.flowchart_storage[st.session_state.current_step] = canvas_result.json_data
            current_flowchart_image = safe_get_canvas_image_data(canvas_result)
            if current_flowchart_image is not None:
                st.session_state.flowchart_image_data = current_flowchart_image
    except Exception:
        pass


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

                first_result = ilk_soruyu_olustur(
                    current_stage,
                    st.session_state.problem_analysis
                )

                first_question = first_result["soru"]
                first_skill = first_result["alt_beceri"]
                st.session_state.last_alt_beceri[current_stage] = first_skill

                st.session_state.chat_storage[current_stage].append({
                    "role": "assistant",
                    "content": first_question,
                    "alt_beceri": first_skill,
                    "basamak_tamamlandi": False
                })

                log_kaydet({
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": student_id,
                    "basamak": current_stage,
                    "alt_beceri": first_skill,
                    "basamak_tamamlandi": False,
                    "tip": "Bot",
                    "icerik": first_question
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

    if current_stage in st.session_state.completed_stages:
        st.success("Bu basamak tamamlandı. Bir sonraki basamağa geçildi.")

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

            "alt_beceri":
                st.session_state.last_alt_beceri.get(current_stage, ""),

            "basamak_tamamlandi":
                False,

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

                result = ogrenciye_cevap_ver(
                    current_stage,
                    st.session_state.problem_analysis,
                    st.session_state.chat_storage[current_stage][:-1],
                    student_message
                )

                answer = result["soru"]
                alt_beceri = result["alt_beceri"]
                stage_completed = result["basamak_tamamlandi"]

                st.session_state.last_alt_beceri[current_stage] = alt_beceri

                st.session_state.chat_storage[current_stage].append({
                    "role": "assistant",
                    "content": answer,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed
                })

                log_kaydet({
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": student_id,
                    "basamak": current_stage,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed,
                    "tip": "Bot",
                    "icerik": answer
                })

                student_turns = sum(
                    1 for m in st.session_state.chat_storage[current_stage]
                    if m["role"] == "user"
                )

                stages = list(BASAMAK_TALIMATLARI.keys())
                stage_index = stages.index(current_stage)

                if stage_completed and student_turns >= 2:
                    if current_stage not in st.session_state.completed_stages:
                        st.session_state.completed_stages.append(current_stage)

                    if stage_index < len(stages) - 1:
                        next_stage = stages[stage_index + 1]
                        st.session_state.current_step = next_stage

                        log_kaydet({
                            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "id": student_id,
                            "basamak": current_stage,
                            "alt_beceri": alt_beceri,
                            "basamak_tamamlandi": True,
                            "tip": "Basamak Geçişi",
                            "icerik": f"{current_stage} tamamlandı → {next_stage}"
                        })

                st.rerun()


            except Exception as e:

                st.error(
                    "Gemini yanıt oluşturamadı."
                )

                st.code(
                    str(e)
                )
