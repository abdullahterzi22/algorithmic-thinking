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
- Öğrencinin henüz söylemediği başka bir veri kaynağını
  (ör. B aracı) soruyla ortaya çıkarmak.
- Çözümü, kritik ilişkiyi veya cevabı açığa çıkaran ipucu vermek.
- Ancak öğrenci takılırsa, yalnızca problemi anlamasına/isteneni-verileni
  ayırt etmesine yarayan ve çözüm yolunu açığa çıkarmayan somut ipucu verilebilir.

""",


    "2. Soyutlama": """

Amaç:
Öğrencinin problemdeki temel bilgileri ayırt etmesini ve bu probleme
özgü, çözümü mümkün kılan kritik matematiksel ilişkiyi kendisinin
keşfetmesini sağlamak.

İKİ TEMEL HEDEF:

1. BİLGİLERİ AYIRT ETME:
- Hangi bilgilerin problem için temel olduğunu, hangilerinin ikincil
  veya doğrudan gerekli olmayabileceğini öğrencinin değerlendirmesini sağla.
- Bir bilgi doğrudan işlemde kullanılmıyor diye otomatik olarak gereksiz
  kabul etme; bağlamı anlamak veya ilişkiyi keşfetmek için önemli olabilir.
- Öğrencinin bu değerlendirmeyi gerekçelendirmesine yardımcı ol.

2. KRİTİK MATEMATİKSEL İLİŞKİYİ KEŞFETME:
- Her problem için kritik ilişkiyi gerçek problem görseli, verileri ve
  öğrencinin önceki cevaplarından belirle.
- İlişkiyi öğrencinin yerine söyleme.
- Öğrenci takılırsa aynı soruyu tekrar etmek yerine kademeli scaffolding
  yap; önce öğrencinin söylediği bilgiye, sonra problemdeki ilgili somut
  niceliklere/görsel özelliklere dikkatini yönelt.
- Öğrencinin bu somut bilgiler arasındaki ilişkiyi kendisinin ifade
  etmesini bekle.

PROBLEM ODAKLI İLERLEME:
- Öğrencinin önceki Ayrıştırma yanıtlarını ve yüklenen problem görselini
  birlikte kullan.
- Öğrencinin zaten fark ettiği bilgi veya ilişkiyi yeniden sordurma.
- Öğrenci yalnızca sayısal değerleri okuyorsa, bu değerlerin problemde
  neyi temsil ettiğini veya birbirleriyle nasıl ilişkili olabileceğini
  düşündür; ilişkiyi söyleme.
- Öğrenci ikincil bir bilgiye odaklanıyorsa onu doğrudan gereksiz ilan
  etme; bağlantısını değerlendirmesini sağla.
- Örüntü/genelleme yalnızca problem yapısı uygunsa ele alınsın.

SCAFFOLDING SINIRI:
Öğrenci bilmiyorsa, belirsizse veya takılmışsa destek ver; ancak destek
öğrencinin bilişsel işini devralmasın. İpucu problemdeki veriye/görsele
konuyu daraltabilir, fakat cevabı, formülü, işlemi, denklemi, yöntemi veya
kritik ilişkiyi açıkça veremez.

YASAK:
- “Bu doğrusal ilişkidir.” demek.
- Kritik ilişkiyi öğrencinin yerine kurmak.
- Denklem veya formül vermek.
- Eğim/sabit değişim gibi çözümün püf noktasını öğrencinin yerine söylemek.
- Kullanılacak işlemi veya yöntemi söylemek.
- Öğrencinin yerine örüntü/genelleme oluşturmak.
- Aynı soruyu farklı kelimelerle tekrar tekrar sormak.
- Öğrencinin zaten söylediği bilgiyi yeniden aratmak.
- Öğrenciyi cevap/sonuç yönünde adım adım sürüklemek.

TAMAMLAMA:
Öğrenci önemli bilgileri yeterince ayırt etmiş ve kritik ilişkiyi kendi
ifadeleriyle ortaya koymuşsa aynı düşünmeyi uzatma. Basamak geçişinde
uygulamanın mevcut tamamlanma kurallarını koru.

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
,

    "5. Metabilişsel Yansıtma": """

Amaç:
Öğrencinin tamamladığı algoritmik problem çözme sürecini
metabilişsel açıdan fark etmesini, izlemesini ve değerlendirmesini sağlamak.

Bu basamak üç boyut üzerinden ilerler:

1. PLANLAMA:
- Problemi çözmeye başlamadan önce neyi amaçladığını,
- hangi bilgileri kullanacağını,
- nasıl bir yol izleyeceğini ve neden o yolu seçtiğini
  fark etmesini sağla.

2. İZLEME:
- Çözüm sırasında ne yaptığını,
- kullandığı bilgilerin amacına hizmet edip etmediğini,
- düşünme ve işlem adımlarını nasıl kontrol ettiğini,
- gerektiğinde düşüncesini nasıl değiştirdiğini
  fark etmesini sağla.

3. DEĞERLENDİRME:
- Oluşturduğu çözümün ve kullandığı yolun problem koşullarıyla
  ne ölçüde uyumlu olduğunu,
- hangi kararlarının işe yaradığını,
- neyi farklı yapabileceğini,
- benzer bir problemde neyi yeniden kullanabileceğini
  değerlendirmesini sağla.

İlerleme sırası:
PLANLAMA → İZLEME → DEĞERLENDİRME

Yasak:
- Öğrencinin yerine kendi sürecini yorumlamak.
- "Şurada iyi plan yaptın." gibi değerlendirme yapmak.
- Öğrencinin neden belirli bir adımı seçtiğini onun yerine açıklamak.
- Çözümü yeniden yapmak.
- Yeni bir çözüm yolu öğretmek.
- "Doğru/yanlış" gibi doğrudan değerlendirme ifadeleri kullanmak.

Bu basamakta amaç yeni bir matematiksel çözüm üretmek değil,
öğrencinin kendi düşünme sürecini fark etmesi ve değerlendirmesidir.

"""
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
    "5. Metabilişsel Yansıtma": [
        ("MY1", "Planlama sürecini fark etme"),
        ("MY2", "Çözüm sürecini izleme"),
        ("MY3", "Çözüm sürecini değerlendirme"),
    ],
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
        "Kurduğun ilişkinin ve bulduğun sonucun problem koşullarıyla uyumlu olduğundan nasıl emin oldun?",

    "5. Metabilişsel Yansıtma":
        "Bu problemde düşünme sürecini planlama, izleme ve değerlendirme açısından nasıl değerlendiriyorsun?"
}


# ============================================================
# ANA SİSTEM PROMPTU
# ============================================================

SYSTEM_PROMPT = """

Sen ortaokul matematik öğrencisine rehberlik eden
bir matematik öğretmenisin.

SÜRÜM: v22 — öğrenci yanıtını sınıflandır; bağlama özgü takip sorusu üret ve bilgi tekrarını engelle.

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

16. HER MESAJDA SORU SORMAK ZORUNLU DEĞİLDİR. Soru yalnızca öğrencinin düşünmesini ilerletmek için gerçekten uygun olduğunda kullanılabilir.

17. Öğrencinin cevabını bekle.
18. Uzun açıklamalar yapma.
19. Liste halinde çözüm verme.
20. Öğrencinin düşüncesini kendin tamamlamaya çalışma.
21. Öğrencinin söylediğini gereksiz yere tekrar etme.
22. Hazır övgü kalıpları kullanma.
23. "Harika bir başlangıç noktası!", "Çok güzel!", "Mükemmel!" gibi kalıpları kullanma.
24. Öğrencinin seviyesine uygun kısa ve doğal Türkçe kullan.
25. Sokratik yaklaşımı "her cümleyi soru yapmak" olarak uygulama. Amaç öğrencinin düşünmesini ilerletmektir. Gerektiğinde soru, ipucu, öğretmen yönlendirmesi veya scaffolding kullan.
26. Öğrenci ilerliyorsa gereksiz destek verme.
27. Öğrenci takıldıysa aynı soruyu farklı kelimelerle tekrar etme; desteği bir kademe somutlaştır.
28. Müdahalenin soru olması gerekmiyorsa soru işareti ekleme.
29. Destek verirken öğrencinin henüz kurmadığı matematiksel ilişkiyi, işlemi, formülü, denklemi, sonucu veya çözüm yolunu onun yerine kurma.

============================================================
YANIT BİÇİMİ
============================================================

- Türkçe yaz.
- Genellikle 1 veya 2 kısa cümle kullan.
- Soru sormak zorunlu değildir.
- Öğrencinin son cevabını ve gerçek problem bağlamını dikkate al.
- Bir sonraki düşünme hareketini destekle.
- Takılma varsa somutlaştır; ilerleme varsa gereksiz yardım verme.
- Öğrencinin yerine çözüm üretme.

============================================================
SCAFFOLDING — HER TURDA AKTİF ÖĞRETİCİ DESTEK PROTOKOLÜ
============================================================

BU PROTOKOL SADECE “BİLMİYORUM” DENİNCE DEĞİL, ÖĞRENCİ DÜŞÜNMEDE
TAKILDIĞINDA, AYNI NOKTADA DÖNDÜĞÜNDE VEYA VERDİĞİ CEVAP SORUYU
KARŞILAMADIĞINDA DA UYGULANIR.

ÖĞRENCİ TAKILDIĞINDA:
- Aynı soruyu farklı kelimelerle tekrar etme.
- Genel “hangi bilgiye ihtiyacın var?” kalıbına geri dönme.
- Önce öğrencinin SON SÖYLEDİĞİ SOMUT NOKTAYI kabul et.
- Sonra mevcut basamağın hedefini koruyan TEK BİR MİKRO-İPUCU ver.
- İpucu öğrencinin dikkatini problemdeki belirli bir veri, nesne, zaman,
  grafik parçası, mevcut adım veya kendi söylediği ifadeye yöneltebilir.
- İpucunun içinde öğrencinin henüz kurmadığı matematiksel ilişkiyi,
  işlemi, formülü, sonucu veya çözüm yolunu söyleme.

SCAFFOLDING KADEMESİ:
1. Öğrencinin son söylediği noktaya dön.
2. İlgili somut veri/nesne/grafik özelliğine dikkatini yönelt.
3. Düşünmeyi daralt: iki veri, iki seçenek veya tek bir karşılaştırma
   üzerinden düşünmesini iste; doğru seçeneği söyleme.
4. Gerekirse aynı düşünceyi tekrar sormak yerine öğrencinin mevcut
   cevabının problemde neyi açıkladığını düşündür.

ÖNEMLİ: Her scaffolding turunda destek seviyesi BİR KADEME ilerlemelidir.
Öğrenci cevap veremediğinde aynı seviyede kalma. Öğrenci ilerlediyse
tekrar destek verme; yeni düşünme adımına geç.

CEVAP VERİLDİYSE:
- Öğrenci sorulan bilgiyi zaten söylediyse o bilgiyi yeniden isteme.
- “340 ve 204” dediyse tekrar “hangi değerler?” deme.
- “İlk 2 saatte aldığı yolu geçen süreye oranlarım” dediyse bunu yeni
  bir veri sorusuna çevirme; mevcut basamağın hedefinde bir sonraki
  düşünme noktasına geç.
- “Dedim ya”, “grafikte yazıyor”, “evet” gibi tepkileri DÖNGÜ sinyali
  kabul et ve soru türünü değiştir.

HER BASAMAKTA SCAFFOLDING'İN YÖNÜ:
- Ayrıştırma: öğrencinin belirlediği hedefi ve bilgi zincirini somut
  veriler üzerinden yapılandırmasına yardım et; çözüm hesabına geçme.
- Soyutlama: öğrencinin belirlediği veriler arasında karşılaştırma,
  değişim, sabitlik veya ilişkiyi fark etmesine yardım et; kritik ilişkiyi
  onun yerine söyleme.
- Algoritma Tasarımı: öğrencinin söylediği adımları sıraya koymasına,
  eksik bağlantıyı kendi çiziminden fark etmesine yardım et; sonraki
  işlemi söyleme.
- Hata Ayıklama: çizimde/çözümde tek bir adımı veya koşulu test etmeye
  yönelt; hatanın yerini söyleme.
- Metabilişsel Yansıtma: öğrencinin düşüncesini ifade etmesini
  kolaylaştırmak için cümle başlangıcı veya somut bir önceki kararını
  hatırlat; onun adına değerlendirme yapma.

MİKRO-İPUCU ÖRNEKLERİ:
- Ayrıştırma: “340 ve 204 değerlerini zaten belirledin; şimdi bunların
  ihtiyaç zincirinde hangi boşluğu karşıladığını düşünür müsün?”
- Soyutlama: “340 km ve 204 km'yi birlikte düşündüğünde, bu iki değerde
  neyin değiştiğine bakmayı dener misin?”
- Algoritma: “Çizdiğin iki adımı düşün; hangisinin önce gelmesi gerektiğine
  karar verirken elindeki bilgiyi ölçüt olarak kullanabilir misin?”
- Hata Ayıklama: “Bu adımın ürettiği bilgi bir sonraki kutuda gerçekten
  kullanılabiliyor mu, çiziminden kontrol eder misin?”
- Metabiliş: “Az önce seçtiğin yolu düşün; bu seçimi yapmana hangi kararın
  yardımcı olmuştu?”

SON SINIR:
Scaffolding bilişsel yükü azaltabilir ve dikkati daraltabilir; ancak
öğrencinin yerine matematiksel ilişki kuramaz, işlem seçemez, formül
veremez, sonuç çıkaramaz veya çözüm yolunu tamamlayamaz.

DÖNGÜ KIRMA KURALI:
Aynı bilgi/ilişki hakkında iki ardışık turda benzer soru sorulmuşsa
üçüncü turda soru biçimini değiştir. Öğrencinin mevcut cevabını veri
olarak kullan ve daha somut bir mikro-ipucuna geç. Aynı “hangi bilgi /
veri / ilişki?” kalıbını art arda kullanmak YASAKTIR.

ÖZEL OLARAK SOYUTLAMA:
Öğrenci kritik matematiksel ilişkiyi kuramadığında ilişkiyi söyleme.
Bunun yerine öğrencinin zaten fark ettiği iki veya daha fazla somut
niceliği yan yana getirerek aralarındaki değişimi, karşılaştırmayı veya
ortak özelliği incelemesini iste. Öğrenci yine takılırsa karşılaştırmanın
boyutunu daralt; fakat ilişkinin adını söyleme.

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
AYRIŞTIRMA İÇİN SORU BANKASI SINIRI
============================================================

Aşağıdaki genel soru türleri yalnızca ilgili basamaklarda kullanılabilir.
MEVCUT BASAMAK "1. Ayrıştırma" ise bu genel soru bankasını kullanma.
Özellikle ilişki, değişim, hız, süre, işlem, hesaplama, grafik
özelliği veya çözüm adımı çağrıştıran sorular üretme.

Ayrıştırmada soru yönü yalnızca geriye doğru olabilir:
- Ne isteniyor?
- Bunu belirlemek için hangi bilgi gerekiyor?
- Bu bilgiye ulaşmak için hangi bilgi gerekiyor?
- Elindeki verilenlerden hangisi bu ihtiyacı karşılıyor?
- Şu ana kadarki bilgiler yeterli mi?
- Eksik olduğunu düşündüğün bilgi nedir?

Öğrenci bir bilgi söylediğinde o bilgi zincirde bir HALKA olarak kabul edilir.
AI o bilgiyi yeniden aratamaz ve öğrencinin söylemediği bir sonraki halkayı
sorunun içine yerleştiremez.

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
    include_annotation_image=False,
    include_flowchart_image=False
):

    final_parts = []
    final_parts.extend(parts)

    # Öğrenci işaretleme alanı varsa, arka plan zaten problem görselini
    # içerdiği için botun görsel olarak işaretlenmiş sürümü görmesi yeterlidir.
    if include_annotation_image and st.session_state.get("annotation_image_data") is not None:
        image_part = image_part_from_canvas(st.session_state.get("annotation_image_data"))
        if image_part is not None:
            final_parts.append(image_part)
    elif include_problem_image:
        image_part = get_problem_image_part()
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
    """Gemini v29 yanıtını çözer; soru artık zorunlu değildir."""
    fallback_skill = ALT_BECERILER[current_step][0][0]
    fallback_text = str(raw_text or "").strip()
    cleaned = fallback_text.replace("```json", "").replace("```", "").strip()
    if not cleaned.startswith("{"):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
    try:
        data = json.loads(cleaned)
        message = str(data.get("mesaj", data.get("soru", ""))).strip()
        question = str(data.get("soru", message)).strip()
        intervention = str(data.get("müdahale_tipi", data.get("mudahale_tipi", "soru"))).strip().lower()
        alt_beceri = str(data.get("alt_beceri", fallback_skill)).strip()
        completed = bool(data.get("basamak_tamamlandi", False))
        if intervention not in {"soru", "ipucu", "yonlendirme", "scaffolding"}:
            intervention = "soru" if "?" in message else "yonlendirme"
        if not message: raise ValueError("mesaj alanı boş")
        valid_codes = {code for code, _ in ALT_BECERILER[current_step]}
        if alt_beceri not in valid_codes: alt_beceri = fallback_skill
        return {"mesaj": message, "soru": question, "müdahale_tipi": intervention, "alt_beceri": alt_beceri, "basamak_tamamlandi": completed}
    except Exception:
        return {"mesaj": fallback_text, "soru": fallback_text, "müdahale_tipi": "soru" if "?" in fallback_text else "yonlendirme", "alt_beceri": fallback_skill, "basamak_tamamlandi": False}


def _normalize_text(text):
    """Tekrar kontrolü için Türkçe metni sadeleştirir."""
    text = str(text or "").casefold()
    text = text.translate(str.maketrans({"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _last_assistant_question(chat_history):
    for message in reversed(chat_history or []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _affirmative_reply(text):
    normalized = _normalize_text(text)
    return normalized in {
        "evet", "evet yeterli", "yeterli", "sanirim evet", "bence yeterli",
        "hayir", "hayır", "hayir yeterli degil", "yeterli degil", "değil"
    }


def _previous_student_message(chat_history):
    """Mevcut yanıttan önceki en son öğrenci mesajını döndürür."""
    for message in reversed(chat_history or []):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _question_seen_before(chat_history, question):
    """Aynı veya çok benzer bir sorunun daha önce sorulup sorulmadığını denetler."""
    q = _normalize_text(question)
    if not q:
        return False
    q_words = set(q.split())
    for message in (chat_history or []):
        if message.get("role") != "assistant":
            continue
        old = _normalize_text(message.get("content", ""))
        if old == q:
            return True
        old_words = set(old.split())
        if len(q_words | old_words) and len(q_words & old_words) / len(q_words | old_words) > 0.78:
            return True
    return False


def _ayristirma_guard(result, chat_history, student_message):
    """Ayrıştırmada yanıt türünü dikkate alır; bilgi tekrarını ve çözüm adımına kaymayı engeller."""
    question = str(result.get("soru", "")).strip()
    previous_question = _last_assistant_question(chat_history)
    previous_student = _previous_student_message(chat_history)
    q_norm = _normalize_text(question)
    prev_norm = _normalize_text(previous_question)
    student_norm = _normalize_text(student_message)
    previous_student_norm = _normalize_text(previous_student)

    # 1) Öğrenci grafikteki bilgiyi zaten gösterdiğini söylüyorsa veriyi yeniden aratma.
    graph_already_seen = any(phrase in student_norm for phrase in (
        "grafikte yaziyor zaten", "grafikte var zaten", "grafikte yaziyor",
        "grafikte var", "grafikte gorunuyor", "grafikte goruluyor",
        "soruda yaziyor zaten", "verilenlerde yaziyor"
    ))
    if graph_already_seen:
        # Önceki öğrenci mesajında somut olarak adlandırılmış verileri takip sorusuna taşı.
        if ("2 saat" in previous_student_norm or "iki saat" in previous_student_norm) and (
            "uzak" in previous_student_norm or "mesafe" in previous_student_norm
        ):
            tailored = (
                "Başlangıçtaki ve 2 saat sonraki uzaklıkları seçtin. "
                "Bu iki bilgiyi kullanarak hangi bilgiyi belirlemeyi amaçlıyorsun?"
            )
        elif "baslangic" in previous_student_norm and ("uzak" in previous_student_norm or "mesafe" in previous_student_norm):
            tailored = (
                "Başlangıç uzaklığını belirttin. Bu bilgi, problemde aradığın uzaklıkla "
                "nasıl bir bağlantı kuruyor?"
            )
        elif previous_student:
            tailored = (
                f"Az önce belirttiğin ‘{previous_student[:110]}’ bilgisini düşün. "
                "Bu bilgi, belirlemek istediğin şeyle nasıl bağlantılı?"
            )
        else:
            tailored = "Grafikte gördüğün bilgiyi tekrar aramana gerek yok. Bu bilgi, problemde aradığın şeyle nasıl bağlantılı?"
        result["soru"] = tailored
        result["basamak_tamamlandi"] = False
        return result

    # 2) Kısa evet/hayır yanıtını yalnızca hemen önceki soruya verilmiş cevap olarak yorumla.
    if _affirmative_reply(student_message) and any(
        phrase in prev_norm for phrase in ("yeterli mi", "yeterli degil", "eksik mi")
    ):
        if student_norm.startswith("evet") or student_norm == "yeterli":
            result["soru"] = "Belirlediğin bilgilerden hangisinin, aradığın şeyi açıklamada en önemli olduğunu düşünüyorsun?"
        else:
            result["soru"] = "Mevcut bilgilerle ilgili hangi noktanın eksik kaldığını düşünüyorsun?"
        result["basamak_tamamlandi"] = False
        return result

    # 3) Öğrenci işlem önerdiyse hesaplamaya geçme; işlemle hangi bilgiyi aradığını netleştir.
    solution_markers = (
        "farkini al", "toplarim", "cikaririm", "bolerim", "carparim",
        "hesaplarim", "islem yap", "formul", "denklem", "hizini bul",
        "egim", "bolme yap", "carpma yap", "oranini bul"
    )
    if any(marker in student_norm for marker in solution_markers):
        result["soru"] = (
            "Bu işlemle hangi bilgiyi belirlemeye çalıştığını kendi sözlerinle açıklayabilir misin?"
        )
        result["basamak_tamamlandi"] = False
        return result

    # 4) Öğrenci somut veri veya nicelik söylediyse onu yeniden isteme; sonraki düşünme halkasını sor.
    has_concrete_data = any(token in student_norm for token in (
        "saat", "dakika", "km", "kilometre", "uzaklik", "mesafe", "grafik",
        "baslangic", "sonraki", "arac", "araci", "veri", "nokta", "deger"
    ))
    if has_concrete_data and any(phrase in prev_norm for phrase in (
        "hangi veri", "hangi bilgiyi", "hangi bilgiye", "hangi deger", "hangi değeri"
    )):
        result["soru"] = (
            "Belirttiğin bu bilgiyi, problemde istenen şeye ulaşmak için nasıl bir ihtiyaçla ilişkilendiriyorsun?"
        )
        result["basamak_tamamlandi"] = False
        return result

    # 5) Ayrıştırma basamağında çözüm/işlem çağrıştıran soruları engelle.
    forbidden_question_markers = (
        "hangi adimi atman", "hangi islemi", "nasil hesap", "hangi formul",
        "egim", "hizini", "sureyi bul", "degisim miktari", "denklemi yaz",
        "islem yap", "sonucu bul", "hesaplayabilir misin"
    )
    if any(marker in q_norm for marker in forbidden_question_markers):
        result["soru"] = "Belirlediğin bilgilerden hangisinin, aradığın şeyi açıklamada önemli olduğunu düşünüyorsun?"
        result["basamak_tamamlandi"] = False
        return result

    # 6) Aynı veya benzer soru tekrar ediyorsa önceki yanıt türüne göre farklı bir takip sorusu seç.
    if _question_seen_before(chat_history, question):
        if any(token in student_norm for token in ("b ile", "b araci", "b aracı", "b'nin", "b nin")):
            result["soru"] = "B ile ilgili belirttiğin bilgilerin, aradığın uzaklığı açıklamak açısından neyi netleştirdiğini düşünüyorsun?"
        elif any(token in student_norm for token in ("yeterli", "eksik", "bilmiyorum", "emin degilim")):
            result["soru"] = "Şu ana kadar belirlediklerinden hangisi hâlâ açıklanmaya ihtiyaç duyuyor?"
        elif has_concrete_data:
            result["soru"] = "Belirttiğin bilgiyi, problemde istenenle ilişkilendirdiğinde hangi bağlantıyı kuruyorsun?"
        else:
            result["soru"] = "Şu ana kadar söylediklerinden hangisi, neyi bulmamız gerektiğini anlamana yardımcı oluyor?"
        result["basamak_tamamlandi"] = False

    return result


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

    if current_step == "1. Ayrıştırma":
        progression = """
ÖZEL KURAL — AYRIŞTIRMA / GERİYE DOĞRU ZİNCİR:

BU AŞAMADA ÇÖZÜM YAPILMAZ.
Amaç yalnızca öğrencinin problemi şu zincirde kendi başına
parçalamasıdır:

İSTENEN
→ BUNU BULMAK İÇİN GEREKEN BİLGİ
→ BU BİLGİYİ BULMAK İÇİN GEREKEN BİLGİ
→ VERİLEN/TEMEL BİLGİ

ÖĞRENCİ TAKILIRSA SCAFFOLDING YAP:
Öğrenci “bilmiyorum”, “anlamadım” veya benzeri bir ifade kullanırsa
aynen aynı soruyu tekrar etme. Önce öğrencinin son söylediği hedefe
dön; ardından gerçek problemde o hedefle ilgili somut bilgilere
dikkatini yönelt. Gerekirse ipucunu bir kademe daha somutlaştır.
Ancak öğrencinin henüz söylemediği çözüm kaynağını, niceliği,
matematiksel ilişkiyi veya işlemi sen söyleme.

ÖRNEK SCAFFOLDING SINIRI:
Öğrenci “Bilmiyorum.” derse “Bunu belirlemek için hangi bilgiye
ihtiyacın var?” sorusunu aynen tekrar etmek yerine “Soruda bu
nicelikle ilgili hangi bilgilerin verildiğine bakabilir misin?” gibi
problem bağlamına dayalı bir destek kullanılabilir. Daha fazla
desteğe ihtiyaç varsa, problemdeki ilgili zaman/değer gibi somut
verilere bakmasını iste; fakat bu veriler arasındaki işlemi veya
ilişkiyi söyleme.

KATI KURAL:
Öğrencinin henüz söylemediği bir sonraki halkayı SEN ortaya çıkaramazsın.
Soru, öğrencinin düşünmesi gereken halkayı isim olarak içinde taşımamalıdır.

ÖNCEKİ SORU-CEVAP BAĞI ZORUNLUDUR:
Öğrencinin kısa cevabını mutlaka hemen önceki rehber sorusuyla birlikte yorumla.
“Evet/hayır” gibi kısa cevapları yeni bilgi gibi yorumlama. Öğrenci bir
bilgiyi zaten söylediyse onu yeniden aratma.

BİLGİ TEKRARI YASAĞI:
Öğrencinin verdiği veri, nicelik veya grafik bilgisi mevcut zincirde
bir halka olarak kabul edilir; aynı şeyi farklı kelimelerle tekrar
sorma. Sonraki soru, henüz ele alınmamış bir sonraki düşünme halkasına
yönelsin.

AYRIŞTIRMADA KESİNLİKLE ÇÖZÜM YÖNLENDİRMESİ YAPMA:
- Nasıl hesaplanır?
- Hangi işlem yapılır?
- Hangi formül kullanılır?
- Hangi ilişkiyi kullanırsın?
- Hızını/süreyi/değişimi nasıl bulursun?
- Hangi grafik noktasını kullanırsın?
- Denklem nasıl bulunur?

Öğrenci bir işlem veya çözüm önerisi söylerse bunu geliştirme ya da
sonraki işlemi sordurma. Ayrıştırmanın hedefinde kalarak öğrencinin
o işlemle hangi bilgiyi belirlemeye çalıştığını düşündür.

Öğrenci yeterli ayrıştırmayı kendi ifadeleriyle tamamladıysa sırf
daha fazla soru üretmek için aynı düşünmeyi uzatma.
"""
    elif current_step == "2. Soyutlama":
        progression = """
SOYUTLAMA İÇİN ÖZEL İLERLEME:
Ayrıştırmada ortaya çıkan bilgileri temel al.
Önce hangi bilgilerin gerçekten gerekli olduğunu, hangilerinin gereksiz olabileceğini düşündür.
Sonra nicelikleri, ilişkileri, değişimleri ve uygun ise örüntü/genel yapıyı fark ettir.
Öğrencinin önceki aşamadaki düşüncelerini yeniden düzenlemesine yardımcı ol; ilişkiyi onun yerine söyleme.
"""
    elif current_step == "3. Algoritma Tasarımı":
        progression = """
ALGORİTMA TASARIMI İÇİN ÖZEL İLERLEME:
Ayrıştırma ve Soyutlama aşamalarındaki öğrenci düşüncelerini kullan.
Öğrencinin çözüm düşüncesini adım adım akış şemasına dönüştürmesine rehberlik et.
İlk adımı, sonraki adımı veya işlemi öğrencinin yerine söyleme.
Akış şeması görselini de incele ve soruyu mevcut çizime bağla.
"""
    elif current_step == "4. Hata Ayıklama":
        progression = """
HATA AYIKLAMA İÇİN ÖZEL İLERLEME:
Öğrencinin oluşturduğu akış şemasını ve önceki aşamalardaki düşüncelerini incele.
Önce çözümü test ettir. Bir tutarsızlık seziliyorsa öğrencinin kendisinin ilgili noktayı bulmasını sağla; ardından düzeltmesini ve yeniden test etmesini iste.
Sorun görünmüyorsa ve yaş düzeyine uygun başka bir yol gerçekten mümkünse alternatif yolu düşündür; alternatif yöntem zorunlu değildir.
"""
    else:
        progression = """
METABİLİŞSEL YANSITMA İÇİN ÖZEL İLERLEME:

Bu basamak yalnızca öğrencinin kendi düşünme sürecine dönük yansıtma içindir.
Yeni bir matematiksel çözüm öğretme veya çözümü yeniden yaptırma.

Sıra:
PLANLAMA → İZLEME → DEĞERLENDİRME

Önce PLANLAMA: öğrencinin çözüm öncesinde amacını, kullanmayı düşündüğü
bilgileri ve izleyeceği yolu nasıl belirlediğini fark ettir.
Sonra İZLEME: öğrencinin çözüm sırasında kendi düşünmesini, kullandığı
bilgileri ve adımlarını nasıl kontrol ettiğini fark ettir.
Son olarak DEĞERLENDİRME: öğrencinin kullandığı yolun ve sonucunun süreç
açısından ne kadar işe yaradığını, neyi farklı yapabileceğini ve benzer
bir problemde neyi yeniden kullanabileceğini değerlendirmesini sağla.

Bir seferde yalnızca tek boyuta odaklan. Öğrenci bir boyutu açıklamadan
diğerine geçme. Öğrencinin yerine onun düşünme sürecini yorumlama.
"""

    prompt = f"""
{SYSTEM_PROMPT}

MEVCUT BASAMAK:
{current_step}

BASAMAĞIN AKADEMİK AMACI:
{BASAMAK_TALIMATLARI[current_step]}

ALT BECERİLER:
{ALT_BECERILER[current_step]}

{progression}

AŞAMA ÖNCELİĞİ:
Mevcut basamak 1. Ayrıştırma ise, öğrencinin henüz çözüm yapmasına izin verme. İlk soru yalnızca istenen veya istenene ulaşmak için gerekli bilgi zincirini başlatmalıdır; grafik üzerindeki çözüm yolunu, matematiksel ilişkiyi veya hesaplamayı hedefleme.

Mevcut basamak 5. Metabilişsel Yansıtma ise, yeni matematiksel içerik öğretme. Öğrencinin tamamladığı süreci PLANLAMA → İZLEME → DEĞERLENDİRME sırasıyla kendi ifadeleri üzerinden fark etmesini ve değerlendirmesini sağla.

PROBLEM ANALİZİ:
{problem_analysis}

ÖNEMLİ:
Yüklenen gerçek problem görselini incele.
İlk soru BU PROBLEMDEKİ gerçek bilgilere dayanmalıdır.
Genel, her probleme uyabilecek mekanik bir soru sorma.


YANITI YALNIZCA ŞU JSON YAPISINDA VER:
{{
  "alt_beceri": "...",
  "müdahale_tipi": "soru",
  "basamak_tamamlandi": false,
  "mesaj": "...",
  "soru": "..."
}}

İlk soru için basamak_tamamlandi her zaman false olsun.
Soru alanında yalnızca tek kısa yönlendirici soru bulunsun.
"""

    raw = gemini_generate(
        [{"text": prompt}],
        temperature=0.2,
        max_tokens=150,
        include_problem_image=True,
        include_flowchart_image=(current_step in ["3. Algoritma Tasarımı", "4. Hata Ayıklama", "5. Metabilişsel Yansıtma"])
    )

    return parse_guidance_response(raw, current_step)


def _norm_text(text):
    """Metin karşılaştırması için Türkçe karakterleri koruyarak sadeleştirir."""
    text = str(text or "").lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _recent_student_statements(chat_history, limit=8):
    """Öğrencinin daha önce verdiği bilgileri modele görünür kılar."""
    statements = []
    for item in chat_history:
        if item.get("role") == "user":
            content = str(item.get("content", "")).strip()
            if content:
                statements.append(content)
    return statements[-limit:]


def _explicit_stage_completion(text):
    norm = _normalize_text(text)
    phrases = (
        "basamagi tamamladim",
        "bu basamagi tamamladim",
        "basamak tamamlandi",
        "bu basamak bitti",
        "basamak bitti",
        "asama tamamlandi",
        "bu asamayi tamamladim",
        "asama tamamladim",
        "burayi tamamladim",
        "burasi tamam",
        "tamamdir gecelim",
        "gecebiliriz"
    )
    return any(_normalize_text(p) in norm for p in phrases)


def _ayristirma_subskill_transition(current_skill, student_message):
    """Öğrencinin Ayrıştırma zincirinde bir halkayı tamamladığını belirgin
    biçimde söylediği durumlarda bir sonraki alt beceriyi seçer."""
    msg = _normalize_text(student_message)
    transitions = {
        "A1": (
            "A2",
            ("isteneni belirledik", "isteneni belirlemis olduk",
             "isteneni belirledim", "istenen belirlendi",
             "isteneni bulduk", "isteneni buldum"),
            "İstediğimiz bilgiyi belirledik. Şimdi bu bilgiyi belirleyebilmek için problemde hangi bilgilere ihtiyaç duyduğumuzu ayıklayalım."
        ),
        "A2": (
            "A3",
            ("gereken bilgiyi belirledik", "gereken bilgileri belirledik",
             "gereken bilgiyi bulduk", "gerekli bilgileri bulduk",
             "parcalara ayirdik", "ayirmis olduk"),
            "İhtiyaç duyduğumuz bilgileri belirledik. Şimdi bu bilgilerin problemdeki hangi verilerle bağlantılı olduğunu netleştirelim."
        ),
    }
    if current_skill in transitions:
        next_skill, markers, message = transitions[current_skill]
        if any(marker in msg for marker in markers):
            return next_skill, message
    return None, None


def _turn_diagnosis(current_step, chat_history, student_message):
    msg = _norm_text(student_message)
    loop_markers = ("bunu konustuk", "yazdim ya", "dedim ya", "soyledim ya", "grafikte yaziyor", "grafikte var zaten", "az once dedim", "tekrar soruyorsun", "bunu zaten soyledim")
    stuck_markers = ("bilmiyorum", "anlamadim", "bulamiyorum", "emin degilim", "goremiyorum", "yapamiyorum", "nereden baslayacagimi bilmiyorum", "ne yapacagimi bilmiyorum", "takildim")
    if _explicit_stage_completion(student_message): state = "AÇIK TAMAMLAMA"
    elif any(_norm_text(x) in msg for x in loop_markers): state = "DÖNGÜ / ÖNCEKİ MÜDAHALE İŞE YARAMADI"
    elif any(_norm_text(x) in msg for x in stuck_markers): state = "TAKILMA"
    elif len(msg.split()) <= 2 and msg in {"evet", "hayir", "olabilir", "sanirim"}: state = "ÖNCEKİ MÜDAHALEYE KISA YANIT"
    else: state = "İLERLEME / KISMİ CEVAP"
    previous_assistant = [str(x.get("content", "")).strip() for x in chat_history if x.get("role") == "assistant" and str(x.get("content", "")).strip()]
    recent_guidance = previous_assistant[-4:]
    settled = _recent_student_statements(chat_history)
    return (f"SON TUR DURUMU: {state}\nÖĞRENCİNİN ÖNCEKİ AÇIKLAMALARI (BUNLARI YENİDEN İSTEME):\n" + ("\n".join(f"- {x}" for x in settled) if settled else "- Henüz önceki öğrenci açıklaması yok.") + "\nSON REHBER MESAJLARI (AYNI DÜŞÜNME HAREKETİNİ TEKRARLAMA):\n" + ("\n".join(f"- {x}" for x in recent_guidance) if recent_guidance else "- Henüz rehber mesajı yok.") + "\nKARAR: İlerleme varsa yeni bilişsel adıma geç. Takılma varsa daha somut destek ver. Döngü varsa önceki müdahaleyi terk et. Soru sormak zorunlu değildir.")


def _question_too_similar(question, chat_history, threshold=0.76):
    q = _norm_text(question)
    if not q or "?" not in str(question): return False
    old_questions = [str(x.get("content", "")) for x in chat_history if x.get("role") == "assistant"][-4:]
    for old in old_questions:
        if "?" not in old: continue
        old_norm = _norm_text(old)
        if old_norm and (q == old_norm or difflib.SequenceMatcher(None, q, old_norm).ratio() >= threshold): return True
    return False


def _response_guard(result, current_step, chat_history, student_message):
    message = str(result.get("mesaj", result.get("soru", ""))).strip() or "Düşünceni bir sonraki adım için biraz daha açabilirsin."
    result["basamak_tamamlandi"] = False
    forbidden = ("y =", "y=", "eğimini bul", "egimi bul", "eğimi hesapla", "egimi hesapla", "önce eğim", "once egim", "formülü kullan", "formulu kullan", "denklemi yaz", "denklem kur", "şu işlemi yap", "su islemi yap", "doğru cevap", "dogru cevap", "hızını bul", "hizini bul", "sabit değişimi bul", "sabit degisimi bul")
    if any(_norm_text(x) in _norm_text(message) for x in forbidden):
        message = {"1. Ayrıştırma": "Belirlediğin bilgileri yerinde tutalım; şu anda yalnızca problemin hangi kısmının açıklığa kavuştuğuna bakalım.", "2. Soyutlama": "Belirlediğin somut bilgileri karşılaştırmaya dönelim; aralarındaki özelliği sen ifade et.", "3. Algoritma Tasarımı": "Çizdiğin adımlara dönelim; mevcut adımlar arasındaki bağlantıyı kendi düşüncenle netleştir.", "4. Hata Ayıklama": "Çözümündeki tek bir adımı seçip problem koşuluyla karşılaştırmaya dön.", "5. Metabilişsel Yansıtma": "Kendi süreç kararlarından birine dönüp bunu nasıl verdiğini açıklamaya devam et."}[current_step]
        result["müdahale_tipi"] = "yonlendirme"
    if _question_too_similar(message, chat_history):
        result["müdahale_tipi"] = "scaffolding"
        message = {"1. Ayrıştırma": "Aynı noktaya dönmeyelim; şimdi belirlediğin bilgilerin hangisinin hâlâ açıklığa kavuşmadığını düşünelim.", "2. Soyutlama": "Önceki karşılaştırmayı tekrarlama; belirlediğin iki somut bilgiden bir özelliği seçip onu incele.", "3. Algoritma Tasarımı": "Aynı adımı yeniden kurma; çizimindeki mevcut kutulardan birinin girdisi ile çıktısı arasındaki bağı incele.", "4. Hata Ayıklama": "Aynı kontrolü tekrarlama; çözümündeki başka bir adımı tek bir örnek üzerinden sınamayı dene.", "5. Metabilişsel Yansıtma": "Aynı yansıtma boyutuna dönme; önceki kararlarından farklı bir anını hatırla."}[current_step]
    intervention = result.get("müdahale_tipi")
    if intervention not in {"soru", "ipucu", "yonlendirme", "scaffolding"}: intervention = "soru" if "?" in message else "yonlendirme"
    if intervention == "soru" and "?" not in message: intervention = "yonlendirme"
    result["müdahale_tipi"] = intervention
    result["mesaj"] = message
    result["soru"] = message
    return result


def ogrenciye_cevap_ver(current_step, problem_analysis, chat_history, student_message):
    """V30: Öğrencinin mevcut alt becerisini izleyerek düşünmeyi ilerleten uygun müdahale üretir."""
    conversation = conversation_text(chat_history)
    previous = previous_stages_text(current_step)
    current_skill = st.session_state.get("last_alt_beceri", {}).get(current_step, ALT_BECERILER[current_step][0][0])
    skill_names = dict(ALT_BECERILER[current_step])
    next_skill = ""
    if current_step == "1. Ayrıştırma":
        codes = [code for code, _ in ALT_BECERILER[current_step]]
        if current_skill in codes and codes.index(current_skill) < len(codes) - 1:
            next_skill = codes[codes.index(current_skill) + 1]
    stage_specific = {"1. Ayrıştırma": "İSTENEN → GEREKEN BİLGİ → O BİLGİ İÇİN GEREKEN BİLGİ → VERİLEN/TEMEL BİLGİ zincirini öğrencinin kurmasına yardım et. Çözüm hesabına geçme.", "2. Soyutlama": "Belirlenen somut bilgileri gerekli/ikincil ve ilişkili/ilişkisiz yönlerden düşündür; uygun ise değişim, sabitlik veya örüntüyü fark ettir. Kritik ilişkiyi söyleme.", "3. Algoritma Tasarımı": "Öğrencinin kendi düşüncesini akış şemasına dönüştürmesine yardım et; mevcut çizimi kullan. Sonraki işlemi veya adımı söyleme.", "4. Hata Ayıklama": "Hata olduğunu varsaymadan belirli bir adımı/koşulu test ettir; hatayı ve düzeltmeyi söyleme.", "5. Metabilişsel Yansıtma": "PLANLAMA → İZLEME → DEĞERLENDİRME sırasıyla öğrencinin kendi sürecini fark etmesini sağla; onun adına değerlendirme yapma."}[current_step]
    prompt=f"""{SYSTEM_PROMPT}
MEVCUT BASAMAK: {current_step}
BASAMAK PROTOKOLÜ:
{BASAMAK_TALIMATLARI[current_step]}
BASAMAĞA ÖZEL KARAR: {stage_specific}
ALT BECERİLER: {ALT_BECERILER[current_step]}
MEVCUT ALT BECERİ: {current_skill} — {skill_names.get(current_skill, "")}
SONRAKİ ALT BECERİ: {next_skill} — {skill_names.get(next_skill, "") if next_skill else ""}
PROBLEM ANALİZİ:
{problem_analysis}
ÖNCEKİ BASAMAKLAR:
{previous}
MEVCUT DİYALOG:
{conversation}
TUR DURUMU:
{_turn_diagnosis(current_step, chat_history, student_message)}
ÖĞRENCİNİN SON MESAJI:
{student_message}

V30 KARAR PROTOKOLÜ:
1. Öğrenci ilerliyorsa söylediği bilgiyi sabitle; yeniden sorma ve gereksiz yardım verme.
2. Kısmi cevap varsa yalnızca eksik düşünme hareketine destek ver.
3. Öğrenci takıldıysa aynı soruyu başka kelimelerle tekrarlama. Son söylediği somut noktadan hareketle problemi/görseli daraltan bir ipucu veya kısa öğretmen yönlendirmesi kullan.
4. "Bunu konuştuk", "dedim ya", "grafikte yazıyor" döngü sinyalidir. Önceki soru biçimini bırak ve farklı bir düşünme hareketine geç.
5. SORU SORMAK ZORUNLU DEĞİLDİR. Uygun olduğunda soru, uygun değilse kısa ipucu veya öğretmen yönlendirmesi kullan.
6. Scaffolding kademesi: öğrencinin son noktası → somut problem/görsel öğe → küçük karşılaştırma/seçim → basamak hedefiyle bağlantı. Her kademede cevabı/işlemi/ilişkiyi öğrencinin yerine kurma.
7. Gerçek problem görselini ve analizini birlikte kullan; görselde olmayan ayrıntıyı uydurma.
8. Mevcut basamağın sınırını koru.
9. Öğrenci açıkça "Basamağı tamamladım" veya eşdeğerini söylemedikçe basamak_tamamlandi=false.
10. Mevcut alt becerinin yapıldığını öğrenci açıkça belirtiyorsa aynı alt beceride kalma; bir sonraki alt beceriye geçiş yapan kısa bir yönlendirme üret. Özellikle Ayrıştırma'da zincirin sırasını koru.
11. Öğrenci bir alt beceriyi tamamladığını söylediğinde, onun zaten söylediği bilgiyi tekrar sordurma veya yeniden doğrulatma.
12. 1-2 kısa, doğal Türkçe cümle kullan. Yapay övgü yok.

YANITI YALNIZCA BU JSON YAPISINDA VER:
{{
  "alt_beceri": "...",
  "müdahale_tipi": "soru | ipucu | yonlendirme | scaffolding",
  "basamak_tamamlandi": false,
  "mesaj": "Öğrencinin göreceği kısa mesaj",
  "soru": "Mesaj soruysa aynı mesaj; değilse boş"
}}
"""
    raw=gemini_generate([{"text":prompt}], temperature=0.2, max_tokens=220, include_problem_image=True, include_annotation_image=True, include_flowchart_image=(current_step in ["3. Algoritma Tasarımı","4. Hata Ayıklama","5. Metabilişsel Yansıtma"]))
    result = parse_guidance_response(raw,current_step)

    # Ayrıştırma zincirinde açık bir alt-beceri ilerlemesini modelin
    # genel bir cümleye döndürmesine izin verme; geçişi deterministik tut.
    if current_step == "1. Ayrıştırma":
        forced_skill, forced_message = _ayristirma_subskill_transition(current_skill, student_message)
        if forced_skill:
            result["alt_beceri"] = forced_skill
            result["müdahale_tipi"] = "yonlendirme"
            result["mesaj"] = forced_message
            result["soru"] = ""

    return _response_guard(result, current_step, chat_history, student_message)


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
# DAYANIKLI SOHBET KONTROL NOKTASI
# ============================================================

CHAT_BACKUP_FILE = "tez_sohbet_yedekleri.json"

def _load_chat_backups():
    try:
        if not os.path.isfile(CHAT_BACKUP_FILE):
            return {}
        with open(CHAT_BACKUP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_chat_backup(student_id):
    if not student_id:
        return
    try:
        backups = _load_chat_backups()
        backups[str(student_id)] = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_step": st.session_state.get("current_step", "1. Ayrıştırma"),
            "chat_storage": st.session_state.get("chat_storage", {}),
            "completed_stages": st.session_state.get("completed_stages", []),
            "last_alt_beceri": st.session_state.get("last_alt_beceri", {}),
            "problem_analysis": st.session_state.get("problem_analysis"),
            "problem_session_id": st.session_state.get("problem_session_id"),
        }
        with open(CHAT_BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(backups, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _restore_chat_backup(student_id):
    if not student_id:
        return False
    try:
        backups = _load_chat_backups()
        backup = backups.get(str(student_id))
        if not isinstance(backup, dict):
            return False
        saved_chat = backup.get("chat_storage")
        if not isinstance(saved_chat, dict):
            return False
        normalized_chat = {stage: saved_chat.get(stage, []) for stage in BASAMAK_TALIMATLARI}
        if not any(normalized_chat.values()):
            return False
        st.session_state.chat_storage = normalized_chat
        saved_step = backup.get("current_step", "1. Ayrıştırma")
        st.session_state.current_step = saved_step if saved_step in BASAMAK_TALIMATLARI else "1. Ayrıştırma"
        st.session_state.completed_stages = [s for s in backup.get("completed_stages", []) if s in BASAMAK_TALIMATLARI]
        st.session_state.last_alt_beceri = backup.get("last_alt_beceri", {}) if isinstance(backup.get("last_alt_beceri", {}), dict) else {}
        if backup.get("problem_analysis"):
            st.session_state.problem_analysis = backup["problem_analysis"]
        st.session_state.problem_session_id = backup.get("problem_session_id")
        return True
    except Exception:
        return False


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_file_data" not in st.session_state:

    st.session_state.uploaded_file_data = None


if "problem_session_id" not in st.session_state:

    st.session_state.problem_session_id = None


if "backup_restored_for_student" not in st.session_state:

    st.session_state.backup_restored_for_student = None


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


    if st.session_state.backup_restored_for_student != student_id:
        _restore_chat_backup(student_id)
        st.session_state.backup_restored_for_student = student_id


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

if st.session_state.current_step == "5. Metabilişsel Yansıtma":
    st.info(
        "🧠 Bu aşama üç boyutta ilerler: **Planlama → İzleme → Değerlendirme**. "
        "Amaç çözümü yeniden yapmak değil, kendi düşünme sürecini fark etmek ve değerlendirmektir."
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


        st.session_state.problem_session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        st.session_state.chat_storage = {
            stage: []
            for stage in BASAMAK_TALIMATLARI
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


# ============================================================
# YÜKLENMİŞ PROBLEM
# ============================================================

else:

    if st.session_state.problem_session_id is None:
        st.session_state.problem_session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    if st.button(
        "❌ Soruyu Değiştir"
    ):

        st.session_state.uploaded_file_data = None
        st.session_state.problem_analysis = None
        st.session_state.problem_session_id = None

        try:
            backups = _load_chat_backups()
            backups.pop(str(student_id), None)
            with open(CHAT_BACKUP_FILE, "w", encoding="utf-8") as f:
                json.dump(backups, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        st.session_state.chat_storage = {stage: [] for stage in BASAMAK_TALIMATLARI}
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
    # PROBLEM ÜZERİNDE ÖĞRENCİ İŞAREMLEME ALANI
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
            update_streamlit=True,
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
                        "problem_session_id": st.session_state.get("problem_session_id"),
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

                first_result = ilk_soruyu_olustur(
                    current_stage,
                    st.session_state.problem_analysis
                )

                first_question = first_result.get("mesaj", first_result.get("soru", ""))
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
                    "problem_session_id": st.session_state.get("problem_session_id"),
                    "basamak": current_stage,
                    "alt_beceri": first_skill,
                    "basamak_tamamlandi": False,
                    "müdahale_tipi": first_result.get("müdahale_tipi", "soru"),
                    "tip": "Bot",
                    "icerik": first_question
                })

                _save_chat_backup(student_id)
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
        if current_stage == "5. Metabilişsel Yansıtma":
            st.success("🎯 Beş aşamalı süreç tamamlandı. Artık genel süreç değerlendirmesine geçebilirsin.")
        else:
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

            "problem_session_id":
                st.session_state.get("problem_session_id"),

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

        # Gemini çağrısından ÖNCE öğrenci mesajını kalıcı kontrol
        # noktasına yaz.
        _save_chat_backup(student_id)


        # ----------------------------------------------------
        # GEMINI YANITI
        # ----------------------------------------------------

        # V29: Basamak geçişi yalnızca öğrenci açıkça tamamladığını söylediğinde yapılır.
        if _explicit_stage_completion(student_message):
            stages = list(BASAMAK_TALIMATLARI.keys())
            stage_index = stages.index(current_stage)
            if current_stage not in st.session_state.completed_stages:
                st.session_state.completed_stages.append(current_stage)
            log_kaydet({
                "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": student_id,
                "problem_session_id": st.session_state.get("problem_session_id"),
                "basamak": current_stage,
                "alt_beceri": st.session_state.last_alt_beceri.get(current_stage, ""),
                "basamak_tamamlandi": True,
                "tip": "Basamak Geçişi",
                "icerik": f"Öğrenci açıkça tamamladı: {current_stage}"
            })
            _save_chat_backup(student_id)
            if stage_index < len(stages) - 1:
                st.session_state.current_step = stages[stage_index + 1]
            st.rerun()

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

                answer = result.get("mesaj", result.get("soru", ""))
                alt_beceri = result["alt_beceri"]
                stage_completed = result["basamak_tamamlandi"] and _explicit_stage_completion(student_message)

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
                    "problem_session_id": st.session_state.get("problem_session_id"),
                    "basamak": current_stage,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed,
                    "müdahale_tipi": result.get("müdahale_tipi", ""),
                    "tip": "Bot",
                    "icerik": answer
                })

                student_turns = sum(
                    1 for m in st.session_state.chat_storage[current_stage]
                    if m["role"] == "user"
                )

                stages = list(BASAMAK_TALIMATLARI.keys())
                stage_index = stages.index(current_stage)

                _save_chat_backup(student_id)

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
