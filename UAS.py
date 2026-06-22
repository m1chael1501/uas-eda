import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ===== LOGO URL (ISI SENDIRI) =====
LOGO_KAMPUS_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR164155YRb6drcEPtycjXGlCaGMWwVuqa_gc_CLJY-4g&s"
LOGO_MYBCA_URL = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJQAlAMBEQACEQEDEQH/xAAcAAACAgMBAQAAAAAAAAAAAAABAgAGAwUHBAj/xABBEAABAwMCAgUJAwkJAAAAAAABAAIDBAURBhIhMQcTQVFhFCIycYGRk6GxFVLBFjNCU1VictHSIyQlQ0VkgqKy/8QAGgEAAgMBAQAAAAAAAAAAAAAAAAECBAUDBv/EADERAAICAgADBQcEAgMAAAAAAAABAgMEERIhMQUTQVFhFDJScYGRsRUjM6EiQtHh8P/aAAwDAQACEQMRAD8ArYK9QeZY4yggO0oEOCgTHBQRZkaUEWOMoEOEERxlBEcIEOExMcIImRqBDBBEyBAg4QIiA2KUDAEiSKoEGozIEEBwgTHagiOECHagizIECHCCI4QRHCBDhMTHagizIEERwgQ4QIcckEWAoBClAxQkySKoEGqzIEEBwgTHagiOECHagixwUCHBQRHBQRMgTEOECY4QRHCBDgoIjgoEHKBEKAAUDFBSJIqoQajHCCI4QRMjUCGBA5nCCLHDgOZCNEWzIOSCIzSDyOUC2O0jOMjKeiOzIEhMZpB5EJkd7MgKBDgoEMCgQwKBBygRMoABKBkBSJJFVHNBpsyBBEcIEOEEWdK6MrRS1FqqquspYZt8+1hkYHYAA5Z8SsrtC2UZqMWa3Z1UXW5NeJmsV2039sizUVra/rXvHlT42kSOGSfHHA49ijdTf3feSl9CdN2P3vdQj9Rr/R6c01W+Xz0PXzVH5mkAGxpHpOweA5hKmeRfHgUtJeIr4Y2PLvHHbfgey40Vq1BpN1xgpGU56l0kTgwNcwtzkHHMZCjXO2i/gb3zJWV05GPxpa5bPJbqSkrujx8raWDyhtJIN4jG7c3ODnv4Kdk5wy9N8to511xswdpLevwVrR1o+2LuwSNzSwYllyOB7m+0/LKvZl3dVvXV8kZuBj9/at9Fzf8AwbHXlZSeWtt9BTwRiA7pnxxgZf3ZHcPquODXLh45Pr0O3alsHPuoLp1KuCtAydDgoEMCgQwKAJlAtEygAZQMgKQ0Vcc0I0mZAUERwUCHBQRZ1q3O+xOjTrh5shpHSD+N/L5kLFs/ey9ev4Nyt9zib9Cq9GVJ1+o2ykebSwudnxPmj6lXc+eqdeZQ7Or3dvyROkes8o1NLHnLKWJsftI3H6hPAhqlPzIdpT4r9eSLbef8G6PG03J/k7IcfvOxn6lUaf3cvi9dmhf+zh8Ppr7mDo3l8p07XUbv8uVwA8HNH45U+0Fw3RkQ7Mlx0Sj6v+xqdrNF6S3Px5fPyB7ZCOA9TR9PFKW8vI0vdX4CKWDi7/2f5OeF7nuc97i57iXOceZJ4krYSSWkefe29sIKZEcFAmEFAhgUBomUCJlAAJQMIKQ0ishJGkxwmRY4QJjcxhBFotl81m67WRlqjt4pYm7BuE+/IbyGNo7gqdWGq7O84t9S5fmu2ru+HRg0nqX8nDUubQipfOGjcZdm0DPDke9Sycbv9c9aOeNlez7/AMd7NdWVxrbpLXTxA9bN1ro88xnO3PyXaFfBBQXgV7LHOxza6s3mptWy6gpIqbyIU7I5OsO2Xfu4EfdGOZVejEVMnJvZYyc2WRFR4df2WDothqY2V1Q9hbSSBoY88NzhnJHhg81W7RlFuMV1LfZcZJSk+hXtZXv7ZvDjE4mlgzHEOw97vb9AreJR3VfPqyhnZHfW8uiNICrRSGBQIYFMQcoFoYFAtEygNEygNAygNBBSGkVtJGixwUyLHCBGxslorb3WGkt7GPlEZkO920BoIHP2hc7bYVR4pk6qZ2y4Ym/HR9qLtgp/jhVvb6PX7Fj9Pv8AT7jDo/1D+pp/jhP2+j1+wv06/wBPueO4aSvlthMtTRF0beLnQuD8evHFTry6ZvSZxsw761tow6cu/wBi3JtaKZtSQxzQx0m0DPbnB+nap3097Dg3ohj3dzPj1s3V71zXXOkdS09OyiheNsm1+9zh3ZwMBV6cGMJcTe2d7+0bLI8MVr+zQ0Furq87aGkmn8Y2Ej38lbnZCHvPRShTZZ7kdm4j0ZqBzd3kQb4GVoP1XD22heJ3XZ2S/D+zx1thu9A0uqqCdrB+k0bh7wukMiqfSRxsxL6/eia9pBC7Fc3Vq0zdLrSCqo4o+qLi0F8m3OOar25dVUuGXUtU4N10OOCWj2fkPff1UHxguft9Hm/sdf0vJ9PuB+ir5Gxz3RQYaMnEoQs+h+f2E+zMlLovuVwOyM96umeiZQMIKBldUNmgxgpIQ4KCB0zofo/7O415HNzYGn1Dcfq1ZXaU+cYfU1ezocpTOkrMNMGUARwygDjWurdFbtTzRUrNsczWytjaORdwIA9YPvW7h2OdO5eB53OqUL2o+JZNJ6EZ1bKy+N3PPFlKeQ/j7z4clVyc574avuXMbs9a47uvkX6GGOCNscLGsY0YDWjACzG23tmqklyRkQMBHBAHKtYtp6/VDaG108bZQRE9zBjfIe/1D8Vt4jlCnjm+XX6Hnc5RtyO7rXPo/n/0dMttFFb6GCkg/NxMDR4+Kx7Juc3J+Jv11quCjHoj1KBMSYboZB3tITXUT6HBz5p293BenPGta5AygQQUDRX8qCNBjAqQhgeHNMidy6O6LyPSVFkYdODO7P73EfLCwcyfHdI3cSHBTFGTXdwqLdpyaSidI2pe9scboxlwyeJHsBSxYRnalLoPKnKFTcepW+jyt1HXXSR9wmq30DYzk1DMAu7NpwrOZCiMNQ6lTCnkTm+PodAqqiKmgfPPIyOKMbnvccABZyi5PSNKUlFbZSNNQM1NqSr1HPGfJYXiKjY8c8D0seH1J7lfvk6KlSuvVmdjxV90r306IveAFnmkU/U2uqe11D6OgjFVUsOHuJxGw93DiSr2PgysXFJ6Rn5PaEanwxW3/RW29It6EhcYaIt+7sd9dyt/p9WurKP6pdvojdQdIlPJb6gzU7oa1kZMbR5zHu7OPZ7Vwl2fJTWntFmPakHB7Wma/o2tr6u41F2qdz+pO1jnfpSO9I+wH5rr2hYowVcfE4dmUuVjtl/5vqdK5BZBuHloq1lVUVkUZz5NII3H97aCfqFOUHFJvxIRmpNpeB6yoEzgc/CeQdz3D5r08fdR4+fvP5v8iZTI6CCgaNBlczQYwKZEz00Tqmoip2elNI2MesnH4puXCmxKPE0j6PpIW01LDAwYbEwMA8AMLzUnttno0tLQk9ZSwODJ6iGN3PD5AD80KLfRA2kZI5I5ow+J7Xsdyc05BSaaemHUpfSba66otZrIKyQ0tPh0tIBgEfe4c8eKv4FkFPha5vxKHaFU5V8SfJeBYdKUTaDTtvp2jBELXO8XO4k+8qtkT47ZMt0Q4KoxMuo6x9BYq6ri/ORQOcz+LHBKmKnZGL8WF83CuUl4I4TuJJLiS4nJJ5k969GeX69QgoFodoc5zWsaXOcQGtHMk8ghi029I7jpy2Ns9np6JuC5jcyOH6TzxJ9685fb3tjmepx6VTWoIzXevjtlsqayb0YWF2O89g9pUaoOyaivElbNVwc34FX6MKh9VQ3OaV26SSr3vPiWglXe0IqMopeRR7Mm5wlJ9Wy6rPNI4LcW9XcayP7tRIPc4r0tb3CL9F+DyVq1ZJerPPlTIBDkAaFcjQYwKYtFm6PKPy7V1C0jLYCZ3f8AHl8yFwzJ8NL9eR2xIcVy9OZ3YclhG2cl1xY79edTVNRT2qokp2hscTvNwQBz4nvJWxiXVV1JOXMyMui221tR5F06P7LW2SyGC4HbLJIZOqDsiMd2eXZnh3qjl3Rts3Eu4lUqq9SNrqR0bNP3F0voCnfn3LlSm7I68ztc0q5b8h7BUMqrJQTsOWvp2H/qErU4zaY65KUE0Z6+ljrqOeknGYp43RvA7iMKMZOMlJeA5xUouL8TjF50rdrTUPY+llqIQTsnibuDh4gcit6rKqsXXmeeuw7a5PltGvgttwndthoKp7u4Qu/kurtguskcVTY3yiyz6BsM0upHuroSz7Pw57HccPI80fiqeZelV/j/ALFzBxn325L3fydY5LGN0B2nnhAEaAPRAHqQASgDhF982+3Md1ZN/wC3L0lP8cfkvweVv/ln82eHK6HIIKBpGiBXHZfYwKYjpfQxRbqm5XBw4sY2Bp9fnO+jVn9oT5Rj9S/gQ6y+h1QLMNEiAFe9rGlziA0DJJPAIA5b0h6whuERtFqk3wbv7xMOT8cmt7xnmfBa2FiuD7yf2MnNylJd3A9HRnqiKCMWWvkDBuJpXu5cTxYfbxHrwo52M2+9j9R4OQl+1L6HSwcrLNUmEAa7UFzZZ7PVV8hB6phLQe13YPeulNTsmoeZyusVcHN+B49G2yS22WM1OTWVLjUVDjzL3cePq5Lpk2KdnLouSIY1bhWt9XzZvHuDGlziA0DJJ7FXLDZySu1/eX1s7qKeJlMXnqmmEHzexbcMCpRXEuZhWdo3cb4Ohc+j68117t1VNcZGyPjqNjS1gbgbWns9ZVDNphVNKPkaOFdO6Dc/MtRVMuHDNVjZqa6N/wBy4+/j+K9Fjfwx+R5nKWr5/M1WV2OAWnggZpAuBfCCmRZuLPqW8WWndTWutNPE95kc0RsOXEAZ4g9gChOmux7ktk4XWVrUXo2A13qf9qv+FH/SoeyU/D+SXtV3xf0hhrrUx/1V/wAKP+lNYlHw/kTy7/i/Br7jfbtc27LhcKieP7jnYb7hwXaFNcOcY6ONl1k+UmeALqcR+YQIsVo1pfbWxsUVX10LeAjqG7wB4Hn81VsxKrHtrXyLNeZdWtb2jdDpPu2ONDRE9/nfzXH9Nr+Jnf8AUrPhRpb3q663psTKp8LY4pBKxkbMDcOWcniu9WJXXvRXuy7Ldb6If8ttR/tN/wAJn9KPY6PhD23I8xKjV9/qYJIJri90UjS146tgyD4gJrEpT2okXmXyWnI0mewKwVdGztV/ulohfDbat0Mcjt7gGNOTjHaPALlZRXY9zWzvVkW1LUGe38tdR/tN3wmfyXP2Oj4Tp7dkfF+DTVdVNW1MlTVP6yaU7nuwBk+xd4xUFwroVpyc5OT6sw5UiJAUDRpcquXxgU9kWhgVIiOCmIYFMjobKkIYFMixgUC0EFAhgUAEFAg5QAQUAHKBaICgNBygNEygNAygNBaeCANNlVdmg0MCpERgU0yLQwKkLQ4KCOggqQtDApkdDApiCCmLQ2UC0EFABygQcoAOUCJlABygCZQAMoAZp4IGabkVUL4wKYmhsqREYFPYhgU0yLQ2UxDApkdDAp7FoOUxaCCnsWggpiYcoDQcoDQcoETKBEygCZQBMoAZp4IGakqoXiAoAIKexND5UiIQUCGBKlsTQwKeyOhgUxBygWg5T2LQQU9i0HKNhoOU9i0HKA0TKA0TKYaJlAaJlINDNPBAaNdtCql0m0IAm0IEHCexNDAJiGATEEBMWhg31p7I6DhPYaDhAggJi0HCBaDhMWg4RsNEwmGiYQGiYQGiYS2GgtHBAaP/2Q=="

col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.image(LOGO_KAMPUS_URL, width=120)

with col3:
    st.image(LOGO_MYBCA_URL, width=120)

with col2:
    st.markdown("""
    <div style="text-align:center;">
        <h2>ANALISIS SENTIMEN APLIKASI MYBCA</h2>
        <h4>Proyek UAS Statistika - Michael Zidane</h4>
    </div>
    """, unsafe_allow_html=True)


import streamlit as st

st.markdown("""
##  Video Perbandingan Fitur Aplikasi MyBCA dan BCA Mobile

Berikut adalah perbandingan visual antara aplikasi **MyBCA** dan **BCA Mobile**
untuk melihat perbedaan fitur dan pengalaman pengguna.
""")

st.video("https://youtu.be/P9dDAr4Uip0?si=CGoHUQhOpds04_rh")

st.markdown("""
###  Latar Belakang
Aplikasi MyBCA merupakan layanan digital dari Bank BCA yang dirancang untuk memudahkan nasabah dalam melakukan berbagai transaksi perbankan secara online. Aplikasi ini ditujukan untuk pengguna muda dengan pendekatan digital-first.

Namun, dalam praktiknya, terdapat berbagai ulasan pengguna di Google Play Store yang menunjukkan adanya perbedaan persepsi terhadap kualitas aplikasi, terutama dibandingkan dengan aplikasi BCA Mobile.

---



### Tujuan Analisis
Analisis ini bertujuan untuk:

- Mengidentifikasi alasan mengapa pengguna lebih memilih BCA Mobile dibandingkan MyBCA
- Menganalisis permasalahan utama yang dialami pengguna pada aplikasi MyBCA berdasarkan ulasan
- Menemukan insight dari feedback pengguna sebagai dasar evaluasi aplikasi
- Memberikan rekomendasi atau solusi perbaikan untuk meningkatkan kualitas dan pengalaman pengguna MyBCA

---


""")

st.sidebar.markdown("##  Identitas Mahasiswa")

st.sidebar.markdown("""
**Nama:** Michael Zidane  
**NIM:** 20254920002  
**Program Studi:** Statistika  
**Proyek:** Sentiment Analysis MyBCA  
""")



# ===== LOAD DATA =====
df_raw = pd.read_csv("data_scraping.csv")

st.subheader(" Data Hasil Scraping")
st.dataframe(df_raw.head(20))

# ===== DATA HASIL CLEANING =====
df_clean = pd.read_csv("hasil_cleaning.csv")

st.subheader("🧹 Data Setelah Cleaning")
st.write(f"Jumlah data: {df_clean.shape[0]} baris")

st.dataframe(df_clean.head(20))
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))

custom_stopwords = {
    'nya','ga','gak','aja','lot','ya','yg','banget','udah','gk','my'
}

stop_words.update(custom_stopwords)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)

    words = text.split()

    words = [w for w in words if w not in stop_words and len(w) > 1]
    words = [stemmer.stem(w) for w in words]

    return " ".join(words)

df = pd.read_csv("data_scraping.csv")

df['cleaned_text'] = df['content'].apply(preprocess_text)

output_df = df[['content','cleaned_text']]

st.markdown("##  Word Cloud Analisis Kata")

text_data = df['cleaned_text'].dropna().astype(str)

all_text = " ".join(text_data)

wc = WordCloud(
    width=900,
    height=400,
    background_color="white",
    colormap="viridis",
    max_words=200
).generate(all_text)

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)

text_data = df['cleaned_text'].dropna().astype(str)

st.markdown("##  Top 20 Kata Paling Sering Muncul")
from sklearn.feature_extraction.text import CountVectorizer

# ambil data cleaned text
text_data = df['cleaned_text'].dropna().astype(str)

# vectorizer untuk hitung frekuensi kata
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(text_data)

# jumlah frekuensi kata
sum_words = X.sum(axis=0)

words_freq = [
    (word, sum_words[0, idx])
    for word, idx in vectorizer.vocabulary_.items()
]

# urutkan dari yang terbesar
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# ambil 20 kata teratas
top_words = words_freq[:20]

# convert ke dataframe
df_top = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df_top['Kata'][::-1], df_top['Frekuensi'][::-1])

ax.set_title("Top 20 Kata Paling Sering Muncul")
ax.set_xlabel("Frekuensi")
ax.set_ylabel("Kata")

st.pyplot(fig)

# ==============================
# INTERPRETASI HISTOGRAM KATA
# ==============================

st.markdown("---")


st.info("""
Grafik frekuensi kata menunjukkan bahwa ulasan pengguna MyBCA didominasi oleh
kata-kata yang berkaitan dengan akses layanan, seperti **login**, **masuk**,
**verifikasi**, dan **update**. Hal ini mengindikasikan bahwa pengalaman pengguna
dalam mengakses aplikasi menjadi salah satu aspek yang paling sering dibahas.

Selain itu, kemunculan kata-kata bernada negatif seperti **gagal** dan **susah**
menunjukkan adanya permasalahan yang masih dirasakan oleh sebagian pengguna.
Temuan ini sejalan dengan tujuan penelitian, yaitu mengidentifikasi faktor-faktor
yang menyebabkan pengguna memberikan penilaian kurang positif terhadap aplikasi
MyBCA.

Berdasarkan hasil tersebut, aspek **login**, **verifikasi akun**, serta kualitas
pembaruan aplikasi dapat menjadi fokus evaluasi untuk meningkatkan kepuasan
pengguna dan memperkuat daya saing MyBCA dibandingkan BCA Mobile.
""")

# =====================================================
# TF-IDF ANALYSIS
# =====================================================

import numpy as np

st.markdown("---")
st.subheader(" Analisis TF-IDF")

st.markdown("""
TF-IDF (Term Frequency - Inverse Document Frequency) digunakan untuk mengubah
data teks menjadi data numerik yang dapat diproses oleh algoritma Machine Learning.

Metode ini memberikan bobot yang lebih tinggi pada kata-kata yang dianggap penting
dalam suatu ulasan dan memberikan bobot yang lebih rendah pada kata yang terlalu
sering muncul pada seluruh dokumen.
""")

# Transform data menggunakan TF-IDF yang sudah dilatih
X_tfidf = tfidf.transform(df['cleaned_text'].fillna("").astype(str))

# Ambil nama fitur
feature_names = tfidf.get_feature_names_out()

# Hitung rata-rata bobot TF-IDF
mean_scores = np.asarray(X_tfidf.mean(axis=0)).ravel()

# DataFrame hasil TF-IDF
tfidf_df = pd.DataFrame({
    'Kata': feature_names,
    'Skor TF-IDF': mean_scores
})

# Ambil 15 kata tertinggi
top_tfidf = tfidf_df.sort_values(
    by='Skor TF-IDF',
    ascending=False
).head(15)

# Tampilkan tabel
st.write("###  Top 15 Kata Dengan Bobot TF-IDF Tertinggi")
st.dataframe(top_tfidf)

# Visualisasi
st.write("###  Visualisasi TF-IDF")

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    top_tfidf['Kata'],
    top_tfidf['Skor TF-IDF']
)

ax.set_title("Top 15 TF-IDF Features")
ax.set_xlabel("Skor TF-IDF")
ax.set_ylabel("Kata")

plt.gca().invert_yaxis()

st.pyplot(fig)

# Interpretasi otomatis
st.markdown("""
###  Interpretasi TF-IDF

Grafik TF-IDF menunjukkan kata-kata yang memiliki tingkat kepentingan tertinggi
dalam ulasan pengguna MyBCA. Semakin tinggi nilai TF-IDF, semakin besar kontribusi
kata tersebut dalam membedakan sentimen atau topik ulasan.

Kata-kata dengan bobot tertinggi dapat dianggap sebagai kata yang paling
merepresentasikan pengalaman pengguna terhadap aplikasi MyBCA dan menjadi
fitur utama yang digunakan model Machine Learning dalam proses klasifikasi sentimen.
""")

# ==============================
# EVALUASI MODEL
# ==============================

st.markdown("---")
st.markdown("#  Evaluasi Model Machine Learning")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Akurasi Model",
    "83.50%"
)

col2.metric(
    "Data Training",
    "800"
)

col3.metric(
    "Data Testing",
    "200"
)

st.markdown("##  Classification Report")

report_df = pd.DataFrame({
    "Precision":[0.80,1.00],
    "Recall":[1.00,0.51],
    "F1-Score":[0.89,0.67]
}, index=["Negative","Positive"])

st.dataframe(
    report_df,
    use_container_width=True
)


st.info("""
Model klasifikasi sentimen berhasil mencapai akurasi sebesar **83,50%**.
Nilai ini menunjukkan bahwa model mampu mengklasifikasikan sebagian besar
ulasan pengguna MyBCA dengan baik ke dalam kategori sentimen positif maupun negatif.
""")

st.subheader("Confusion Matrix")

cm_data = [
    [133, 0],
    [33, 34]
]

fig, ax = plt.subplots(figsize=(6,4))

im = ax.imshow(cm_data)

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(["Negative","Positive"])
ax.set_yticklabels(["Negative","Positive"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_data[i][j],
                ha="center",
                va="center",
                fontsize=14)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.info("""
### Interpretasi Hasil Model

Berdasarkan confusion matrix, model mampu mengidentifikasi
133 ulasan negatif dengan benar dan 34 ulasan positif dengan benar.

Masih terdapat 33 ulasan positif yang diprediksi sebagai negatif.
Hal ini menunjukkan bahwa model cenderung lebih sensitif dalam mendeteksi
sentimen negatif dibandingkan sentimen positif.

Meskipun demikian, tingkat akurasi sebesar 83,50% menunjukkan bahwa model
sudah cukup baik untuk digunakan dalam analisis sentimen ulasan pengguna MyBCA.
""")

# =====================================================
# DISTRIBUSI SENTIMEN
# =====================================================

st.markdown("---")
st.header(" Distribusi Sentimen Pengguna")

sentiment_counts = {
    "Negative": 796,
    "Positive": 204
}

df_sentiment = pd.DataFrame({
    "Sentimen": sentiment_counts.keys(),
    "Jumlah": sentiment_counts.values()
})

fig, ax = plt.subplots(figsize=(7,4))

ax.bar(
    df_sentiment["Sentimen"],
    df_sentiment["Jumlah"],
    color=["red", "green"]
)

ax.set_title("Distribusi Sentimen Pengguna MyBCA")
ax.set_ylabel("Jumlah Ulasan")

for i, v in enumerate(df_sentiment["Jumlah"]):
    ax.text(i, v + 10, str(v), ha='center')

st.pyplot(fig)

st.subheader("Persentase Sentimen")

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
    [796,204],
    labels=["Negative","Positive"],
    colors=["red", "green"],
    autopct="%1.1f%%"
)

st.pyplot(fig)

st.info("""
### Interpretasi Hasil Analisis Sentimen

Berdasarkan hasil klasifikasi sentimen terhadap 1000 ulasan pengguna MyBCA,
diperoleh sebanyak 796 ulasan (79,6%) memiliki sentimen negatif,
sedangkan 204 ulasan (20,4%) memiliki sentimen positif.

Dominasi sentimen negatif menunjukkan bahwa sebagian besar pengguna masih
mengalami berbagai kendala saat menggunakan aplikasi MyBCA.

Hasil ini mengindikasikan bahwa kualitas layanan dan pengalaman pengguna
masih perlu ditingkatkan agar dapat bersaing dengan aplikasi BCA Mobile
yang telah lebih lama digunakan oleh nasabah.
""")

st.markdown("---")
st.header(" Insight Masalah Utama MyBCA")

st.info("""
Berdasarkan Word Cloud, Histogram Kata, dan Analisis Sentimen,
terdapat beberapa permasalahan yang paling sering dikeluhkan pengguna:

1. Kesulitan Login ke aplikasi
2. Proses Verifikasi yang rumit
3. Gangguan setelah update aplikasi
4. Performa aplikasi yang lambat
5. Error saat melakukan transaksi
6. Kendala sinkronisasi akun
7. Bug pada fitur tertentu
""")

st.markdown("---")
st.header(" Kesimpulan")

st.info("""
1. Mayoritas pengguna memberikan sentimen negatif terhadap aplikasi MyBCA.

2. Kata-kata yang paling sering muncul berkaitan dengan login, update,
verifikasi, dan masalah akses aplikasi.

3. Hasil analisis menunjukkan bahwa pengguna masih lebih nyaman menggunakan
BCA Mobile karena dinilai lebih stabil dan mudah digunakan.

4. Akurasi model klasifikasi mencapai 83,50% sehingga cukup baik untuk
mengidentifikasi sentimen pengguna.

5. Diperlukan peningkatan stabilitas sistem, penyederhanaan proses login,
serta perbaikan bug agar kualitas layanan MyBCA semakin baik.
""")

st.markdown("---")
st.header(" Rekomendasi Perbaikan untuk MyBCA")

st.info("""
Berdasarkan hasil analisis ulasan pengguna, beberapa rekomendasi yang dapat
diberikan kepada pengembang MyBCA adalah:

✅ Meningkatkan stabilitas aplikasi setelah update

✅ Menyederhanakan proses login dan verifikasi

✅ Mempercepat respons aplikasi saat transaksi

✅ Mengurangi bug yang sering muncul pada fitur utama

✅ Meningkatkan kualitas layanan pelanggan untuk menangani keluhan pengguna

✅ Mengadopsi pengalaman pengguna (user experience) yang telah berhasil
diterapkan pada aplikasi BCA Mobile
""")

st.subheader("🤖 Prediksi Sentimen")

user_input = st.text_area(
    "Masukkan Review Pengguna",
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan"
)

if st.button("Prediksi Sentimen"):

    if user_input.strip():

        vector = tfidf.transform([user_input.lower()])

        prediction = model.predict(vector)[0]

        st.write(f"Kode Prediksi Model: {prediction}")

        if prediction == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😡 Sentimen Negatif")

    else:
        st.warning("Masukkan teks terlebih dahulu")
