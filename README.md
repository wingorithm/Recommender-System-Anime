# Recommender System For Anime - Erwin Gunawan

## Project Overview
Anime telah menjadi aspek yang signifikan dalam hiburan global, dengan jutaan penggemar yang terlibat dengan beragam genre, dan story tellingnya. Sekitar 750 - 800 juta orang menonton anime di seluruh dunia, mewakili audiens yang luas dan beragam yang mencakup berbagai wilayah dan demografi [[1]](https://www.smh.com.au/culture/movies/800-million-and-growing-why-everyone-wants-a-piece-of-the-anime-action-20240314-p5fcek.html). Popularitas yang luar biasa ini menggarisbawahi kebutuhan yang semakin meningkat akan sistem yang efektif yang membantu penonton mengexplore lanskap anime yang terus berkembang.

Hal ini selaras dengan pasar anime global mencerminkan lonjakan minat, dengan valuasi sekitar $31,23 miliar pada tahun 2023. Proyeksi menunjukkan tingkat pertumbuhan tahunan gabungan (CAGR) sebesar 9,8% dari tahun 2024 hingga 2030, yang berpotensi mencapai $60,07 miliar pada akhir dekade ini. Pertumbuhan ini menyoroti meningkatnya permintaan konten anime dan integrasinya ke media straming seperti netflix, crunchyroll, dll [[2]](https://www.grandviewresearch.com/industry-analysis/anime-market/toc)[[3]](https://finance.yahoo.com/news/anime-market-research-report-2024-080300191.html?guccounter=1).

Namun seiring berkembangnya industri anime, platform yang menayangkan anime menghadapi tantangan dalam menghubungkan pengguna secara efektif dengan konten yang sesuai dengan preferensi mereka. Hal ini sangat penting mengingat banyaknya judul yang tersedia dan beragamnya selera penonton. Sistem rekomendasi anime yang dipersonalisasi menawarkan solusi yang menarik dengan memanfaatkan data interaksi pengguna dan metadata anime untuk menyusun saran yang disesuaikan.

<img src="https://img.freepik.com/premium-photo/anime-invasion-s-japanese-animation-global-fans-concept-anime-culture-global-fandom-animation-appreciation-japanese-influence-crosscultural-connection_918839-31777.jpg" alt="Reccomender System Anime" title="Reccomender System Anime" width="100%">

Pentingnya sistem tersebut semakin terlihat dari dampak finansial yang diamati pada domain lain. Misalnya, Netflix memperkirakan bahwa mesin rekomendasinya menghemat lebih dari $1 miliar bagi perusahaan setiap tahunnya dengan mengurangi _subscriber churn_ melalui tampilan dan pilihan dipersonalisasi [[4]](netflix-recommendation-engine-worth-1-billion-per-year). Selain itu, sekitar 80% aktivitas penonton Netflix didorong oleh rekomendasi yang dipersonalisasi, yang menunjukkan efektivitas sistem tersebut dalam meningkatkan keterlibatan dan kepuasan pengguna [[5]](https://www.linkedin.com/pulse/netflix-recommender-system-case-study-ashish-gupta/).

Menggabungkan sistem rekomendasi untuk anime tidak hanya mengatasi tantangan teknis tetapi juga memberikan manfaat ekonomi dan pengalaman pengguna. Ini meningkatkan kemampuan menemukan konten, meningkatkan keterlibatan pengguna, dan sejalan dengan tujuan yang lebih luas untuk mempertahankan dan mengembangkan basis pengguna. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi anime yang optimal dengan memanfaatkan algoritma _Machine Learning_ (ML) untuk mengatasi tantangan ini dan berkontribusi untuk mengingkatkan value pada ruang hiburan digital.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, maka permasalahan yang akan dibahas dalam proyek ini dirumuskan sebagai berikut:

1. Bagaimana cara menyiapkan data anime, preferensi pengguna, dan rating atau interaksi sehingga dapat digunakan secara efektif untuk membuat model machine learning untuk sistem rekomendasi?

2. Bagaimana cara mengembangkan model machine learning yang dapat merekomendasikan judul anime kepada pengguna berdasarkan preferensi mereka dan memberikan rekomendasi anime serupa berdasarkan judul yang dipilih?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:

1. Melakukan tahap persiapan data anime, penonton, dan rating sehingga data siap digunakan pada model machine learning untuk sistem rekomendasi.
2. Membuat model machine learning untuk sistem rekomendasi buku terbaik kepada pengguna.
3. Menguji model dan sistem yang dibangun, serta membuat table akhir dari setiap fungsi prediksi.

### Solution Statements
Untuk mengatasi tantangan explorasi / rekomendasi anime, sistem ini menerapkan dua fungsi utama:

- User-Based Collaborative Filtering:
  
  Sistem merekomendasikan judul anime kepada pengguna terpilih dengan menganalisis preferensi pengguna dengan selera yang sama. Dengan menyaring judul anime yang telah ditonton pengguna, sistem mengidentifikasi rekomendasi baru yang sesuai dengan minat mereka.

  - Strengths: Secara efektif menangkap preferensi dan pola yang didorong oleh komunitas, memberikan rekomendasi yang sangat personal.

  - Limitations: Bergantung pada data interaksi pengguna, yang dapat menimbulkan tantangan bagi pengguna baru dengan riwayat yang jarang.

- Content-Based Recommendations Using Embedding Similarity:

  Sistem mengidentifikasi N judul anime serupa teratas berdasarkan kesamaan kosinus penyematannya. Penyematan ini mewakili berbagai atribut dan hubungan di antara anime, yang memungkinkan sistem untuk menyarankan konten terkait saat judul anime tertentu dipilih.

  - Strengths: Ideal untuk eksplorasi konten, memungkinkan pengguna untuk menemukan anime dengan karakteristik atau tema yang serupa.

  - Limitations: Rekomendasi mungkin kurang beragam karena berfokus pada konten yang sangat terkait. Sistem pendekatan ganda ini menyeimbangkan personalisasi dengan eksplorasi konten, memenuhi kebutuhan penggemar anime berpengalaman dan pendatang baru. Dengan menggabungkan penyaringan kolaboratif dengan kesamaan berbasis penyisipan, sistem ini meningkatkan kepuasan pengguna, keterlibatan, dan penemuan konten.

## Data Understanding
![image](https://github.com/user-attachments/assets/8d51db72-9e4b-40dc-8610-bf4b00e48194)

[Dataset ini](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) memberikan gambaran rinci tentang preferensi penonton, dan statistik data terhadap suatu judul anime, yang sangat cocok untuk eksplorasi data pada domain anime dan membuat recommender system. Terdiri dari 3 dataset utama:

1. `anime-dataset-2023.csv`, 24,905 sampel data anime yang berisi informasi berharga untuk menganalisis dan memahami karakteristik, peringkat, popularitas, dan jumlah penonton.

- ![image](https://github.com/user-attachments/assets/017485a7-aebb-42c8-a893-9ccaf0878b96)
  - anime_id: ID unik untuk setiap anime.
  - Nama: Nama anime dalam bahasa aslinya.
  - Nama bahasa Inggris: Nama anime dalam bahasa Inggris.
  - Nama lain: Nama asli atau judul anime (bisa dalam bahasa Jepang, Mandarin, atau Korea).
  - Nilai: Nilai atau peringkat yang diberikan untuk anime.
  - Genre: Genre anime, dipisahkan dengan koma.
  - Sinopsis: Deskripsi singkat atau ringkasan alur anime.
  - Tipe: Jenis anime (misalnya, serial TV, film, OVA, dll.).
  - Episode: Jumlah episode dalam anime.
  - Ditayangkan: Tanggal saat anime ditayangkan.
  - Tayang Perdana: Musim dan tahun saat anime ditayangkan perdana.
  - Status: Status anime (misalnya, Selesai Ditayangkan, Sedang Ditayangkan, dll.).
  - Produser: Perusahaan produksi atau produser anime.
  - Pemberi Lisensi: Pemberi lisensi anime (misalnya, platform streaming).
  - Studio: Studio animasi yang menggarap anime.
  - Sumber: Materi sumber anime (misalnya, manga, novel ringan, orisinal).
  - Durasi: Durasi setiap episode.
  - Rating: Peringkat usia anime.
  - Peringkat: Peringkat anime berdasarkan popularitas atau kriteria lainnya.
  - Popularitas: Peringkat popularitas anime.
  - Favorit: Jumlah kali anime ditandai sebagai favorit oleh pengguna.
  - Dinilai Oleh: Jumlah pengguna yang menilai anime.
  - Anggota: Jumlah anggota yang telah menambahkan anime ke daftar mereka di platform.
  - URL Gambar: URL gambar atau poster anime.

2. `users-details-2023.csv`, 731,290 sampel data user yang berisi informasi untuk menganalisis perilaku dan preferensi pengguna di platform anime.

- ![image](https://github.com/user-attachments/assets/18d88a1d-b22c-461b-84b9-91359f1d25c3)
  - Mal ID: ID unik untuk setiap pengguna.
  - Nama pengguna: Nama pengguna pengguna.
  - Jenis kelamin: Jenis kelamin pengguna.
  - Tanggal lahir: Tanggal lahir pengguna (dalam format ISO).
  - Lokasi: Lokasi atau negara pengguna.
  - Bergabung: Tanggal saat pengguna bergabung dengan platform (dalam format ISO).
  - Hari Ditonton: Jumlah total hari yang dihabiskan pengguna untuk menonton anime.
  - Skor Rata-rata: Skor rata-rata yang diberikan oleh pengguna untuk anime yang telah ditontonnya.
  - Menonton: Jumlah anime yang sedang ditonton oleh pengguna.
  - Selesai: Jumlah anime yang diselesaikan oleh pengguna.
  - Ditunda: Jumlah anime yang ditunda oleh pengguna.
  - Dihentikan: Jumlah anime yang dihentikan oleh pengguna.
  - Rencana untuk Ditonton: Jumlah anime yang direncanakan pengguna untuk ditonton di masa mendatang.
  - Total Entri: Jumlah total entri anime dalam daftar pengguna.
  - Ditonton Ulang: Jumlah anime yang ditonton ulang oleh pengguna.
  - Episode yang Ditonton: Jumlah total episode yang ditonton oleh pengguna.

3. `users-score-2023.csv`, 24,325,191 sampel data yang berisi interaksi user terhadap suatu anime.

- ![image](https://github.com/user-attachments/assets/b462496c-7f3e-4ada-b77c-b9330797237e)
  - user_id: ID unik untuk setiap pengguna.
  - Username: Nama pengguna pengguna.
  - anime_id: ID unik untuk setiap anime.
  - Anime Title: Judul anime.
  - rating: Peringkat yang diberikan pengguna untuk anime.

Setelah memahami dataset kemudian 3 dataset diatas perlu evaluasi awal untuk mengidentifikasi masalah data yang dapat memengaruhi hasil analisis, sebelum melanjutkan ke tahap data preparation.

```python
  duplicates_score = df_score.duplicated().sum()
  duplicates_user = df_user.duplicated().sum()
  duplicates_anime = df_anime.duplicated().sum()

  missing_values_score = df_score.isnull().sum().sum()
  missing_values_user = df_user.isnull().sum().sum()
  missing_values_anime = df_anime.isnull().sum().sum()

  nan_values_score = df_score.isna().sum().sum()
  nan_values_user = df_user.isna().sum().sum()
  nan_values_anime = df_anime.isna().sum().sum()
```
1. Pengecekan Duplikasi
Dataset diperiksa untuk keberadaan baris duplikat menggunakan metode `.duplicated()`.

Hasil: Dataset tidak memiliki {duplicates} baris duplikat, yang dapat memengaruhi kualitas model jika tidak ditangani.

2. Pengecekan Missing Values
Dataset diperiksa untuk nilai yang hilang (missing values) menggunakan fungsi `.isnull().sum()`.

Hasil: Tidak terdapat {missing_values} missing values di dataset.

3. Pengecekan NaN Values
Dataset juga diperiksa untuk keberadaan nilai NaN menggunakan fungsi `.isna().sum()`.

Hasil: Tidak terdapat {nan_values} nilai NaN di dataset.

```
The data has issues:
There are 232 missing values in df_score.
There are 1648695 missing values in df_user.
There are 232 NaN values in df_score.
There are 1648695 NaN values in df_user.
```

Dari hasil evaluasi awal didapati bahwa data-data yang diperoleh tidak sepenuhnya bersih, hal ini dicatat dan akan dilakukan data cleaning pada tahap preprocessing. Selain memahami deskripsi setiap fiturnya, *Exploratory Data Analysis* (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi juga dilakukan.

1. **Univariate Analysis**

![image](https://github.com/user-attachments/assets/7016ae26-2016-4049-bf59-a2e2baf4e3e2)

Dari analisis data univariat, diperoleh beberapa informasi berikut:
- Sebagian besar rank anime terkonsentrasi antara 7 dan 10, yang menunjukkan kecenderungan ke arah peringkat yang lebih tinggi. Frekuensi meningkat secara signifikan pada nilai integer, yang menunjukkan pengguna sering memberikan peringkat bilangan bulat.
-Genre yang paling populer adalah Komedi, Fantasi, dan Petualangan, dengan Komedi mendominasi jumlah tersebut. Genre seperti Ero dan Grimace adalah yang paling sedikit terwakili, yang menunjukkan audiens yang lebih kecil atau judul yang lebih sedikit.
- Anime TV adalah jenis yang paling umum, diikuti oleh Film dan OVA. Anime Spesial dan Musik lebih jarang.
- Mayoritas data mencantumkan produser sebagai UNKNOWN, yang menunjukkan data yang hilang atau tidak bersih secara signifikan. Di antara produser yang teridentifikasi, Pink Pineapple dan TNK memiliki jumlah yang lebih tinggi.
- Pengguna Pria mendominasi kumpulan data, diikuti oleh pengguna Wanita.
- Polandia memiliki jumlah pengguna tertinggi, diikuti oleh Jerman dan Kanada. 

2. **Multivariate Analysis**

![image](https://github.com/user-attachments/assets/e7a3d89b-341d-4069-ae59-9f7a24f60fa9)
![image](https://github.com/user-attachments/assets/9d32e446-35e3-42b5-af57-9f7e4e763e2c)

Heatmap korelasi (Correlation Matrix) memberikan beberapa wawasan penting:
- _Feature_ seperti Favorit dan Anggota saling terkait erat, yang menunjukkan bahwa keduanya menangkap _engagement behavior_ yang serupa.
- `Popularity` tampaknya memiliki hubungan yang beragam, dipengaruhi secara positif oleh faktor-faktor seperti `studio` dan `genre` tetapi secara negatif oleh jumlah `membership `.
- Korelasi yang lemah di antara pasangan fitur lainnya menunjukkan ketergantungan langsung yang terbatas, yang mungkin menunjukkan bahwa fitur-fitur ini menangkap aspek-aspek berbeda dari data anime.

## Data Preparation
Sebelum membangun model recommendation, dilakukan beberapa langkah _data preparation_ untuk memastikan kualitas data dan kompatibilitas dengan algoritma _neural network-based recommender system model_. Berikut adalah proses dan hasil dari tahap data preparation berdasarkan kode yang digunakan:

```python
df_anime_cleaned = df_anime[df_anime['Genres'] != 'Unknown']
df_anime_cleaned = df_anime_cleaned[df_anime_cleaned['Type'] != 'Unknown']
```

1. Handling Unknown Values

   Menghapus baris-baris di `df_anime` pada kolom `Genres` atau `type` berisi "Unknown". Hal ini diperlukan karena keberadaan nilai "Unknown" menimbulkan gangguan ke dalam kumpulan data, nilai-nilai ini tidak memberikan informasi yang berguna bagi model. Menghapus baris-baris ini memastikan bahwa model berfokus pada atribut yang valid dan bermakna.

```python
df_user_cleaned = df_user.drop(columns=['Gender', 'Birthday', 'Location'])
columns_to_impute = ['Days Watched', 'Mean Score', 'Watching', 'Completed',
                     'On Hold', 'Dropped', 'Plan to Watch', 'Total Entries',
                     'Rewatched', 'Episodes Watched']
for col in columns_to_impute:
    df_user_cleaned[col].fillna(df_user_cleaned[col].median(), inplace=True)

df_score_cleaned = df_score.dropna(subset=['Username'])
```

2. Handling Missing Values

   pada `df_user`, kolom seperti `Gender`, `Birthday`, dan `Location` di hapus karena mengandung sangat banyak missing values. Hal ini juga menimbang bahwa kolom-kolom ini tidak digunakan pada training karena rendahnya relevansi. untuk _feature numeric_ pada `df_user`, seperti `Days Watched`, dll, missing values di isi dengan median. Pada `df_score`, baris data dengan _missing values_ pada kolom kritikal (Username) di hapus.

   Hal ini dilakukan karena proporsi data yang hilang dalam kolom categorical yang tinggi membuat imputasi tidak dapat diandalkan. Menghilangkan kolom-kolom ini menghindari pengenalan informasi yang bias atau keliru. Untuk kolom numeric, imputasi median memastikan bahwa data yang hilang diisi tanpa distribusi yang menyimpang, sehingga membuat kumpulan data menjadi _robust_ untuk pelatihan model. Menghapus baris yang tidak lengkap dalam `df_score` memastikan data pelatihan konsisten dan akurat

```python
scaler = MinMaxScaler(feature_range=(0, 1))

df_score_cleaned['scaled_score'] = scaler.fit_transform(df_score_cleaned[['rating']])
```

3. Feature Scaling

   Menskalakan kolom peringkat di `df_score` menggunakan `MinMaxScaler` untuk menormalkan nilai antara 0 dan 1. Hal ini dilakuan karena penskalaan fitur numerik membantu model menyatu lebih cepat selama pelatihan dan memastikan bahwa fitur-fitur tersebut sebanding. Nilai yang dinormalisasi sangat penting dalam arsitektur jaringan neural untuk menghindari masalah yang disebabkan oleh _large numerical ranges_.

```python
user_encoder = LabelEncoder()
df_score_cleaned["user_encoded"] = user_encoder.fit_transform(df_score_cleaned["user_id"])
num_users = len(user_encoder.classes_)

anime_encoder = LabelEncoder()
df_score_cleaned["anime_encoded"] = anime_encoder.fit_transform(df_score_cleaned["anime_id"])
num_animes = len(anime_encoder.classes_)
```
4. Encoding Identifiers

   Melakukan proses _encoding_ pada `user_id` dan `anime_id` menggunakan `LabelEncoder`. Memetakan setiap pengguna dan anime ke bilangan bulat unik dan menyimpannya sebagai user_encoded dan anime_encoded. Hal ini diperlukan karena sistem rekomendasi beroperasi pada representasi numerik pengguna dan item. ID pengodean memastikan bahwa penyematan yang digunakan dalam jaringan saraf dipetakan dengan tepat.

```python
df_score_cleaned = shuffle(df_score_cleaned, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]
```

5. Shuffling and Splitting Data
   
   melakukan pengacakan pada dataset `df_score_cleaned` untuk memastikan keacakan. Serta membagi data menjadi set pelatihan (80%) dan pengujian (20%). Hal ini dilakukan karena pengacakan mengurangi bias selama pelatihan dengan mencampur titik data dari distribusi yang berbeda. Memisahkan set data memberikan metrik evaluasi yang tidak bias untuk kinerja model pada data yang tidak terlihat.
   
## Modeling
Pada tahap ini data yang telah dipersiapan akan digunakan untuk data latih bagi model recommender dan fungsi-fungsi recommendeer

### 1. Arsitektur Model
```python
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def RecommenderNet(num_users, num_animes, embedding_size=128):
    user = Input(name='user_encoded', shape=[1])
    user_embedding = Embedding(
        name='user_embedding', input_dim=num_users, output_dim=embedding_size
    )(user)

    anime = Input(name='anime_encoded', shape=[1])
    anime_embedding = Embedding(
        name='anime_embedding', input_dim=num_animes, output_dim=embedding_size
    )(anime)

    dot_product = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    flattened = Flatten()(dot_product)

    dense = Dense(64, activation='relu')(flattened)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[user, anime], outputs=output)
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001),
        metrics=[root_mean_squared_error, MeanAbsoluteError()]
    )
    return model

model = RecommenderNet(num_users, num_animes)
model.summary()
```
   Sistem rekomendasi dirancang untuk memprediksi interaksi antara pengguna dan judul anime menggunakan embedding-based neural network architecture.. Berikut ini adalah penjelasan terperinci dari komponen model:

- Input:

   User Input: Satu bilangan bulat yang mewakili encoded userID.
   Anime Input: Satu bilangan bulat yang mewakili encoded animeID.

- Embeddings:

   User Embedding Layer: Memetakan User Input ke dalam representasi vektor padat dengan embedding size yang dapat dikonfigurasi (default adalah 128).
   Anime Embedding Layer: Memetakan anime input ke dalam representasi vektor padat dengan embedding size yang sama dengan user embedding.

- Dot Product:

   Menghitung produk titik dari penyematan pengguna dan anime. Operasi ini menangkap kesamaan antara pengguna dan anime di ruang laten.
   Dot Product dinormalisasi, memastikan bahwa output tetap dalam rentang yang konsisten.

- Flattening:
  
   Hasil perkalian titik diratakan menjadi tensor 1 dimensi, sehingga kompatibel dengan lapisan padat berikutnya.

- Dense Layers:
  
   First Dense Layer: fully connected layer dengan _x_ (64) unit dan aktivasi ReLU. Lapisan ini menangkap hubungan non-linier antara fitur laten.
   Output Layer: fully connected layer dengan satu unit dan aktivasi sigmoid. Aktivasi sigmoid menskalakan keluaran ke rentang 0 hingga 1, yang mewakili skor interaksi yang diprediksi.

- Compilation:

   - Loss Function:  Model menggunakan Mean Squared Error (MSE) untuk meminimalkan perbedaan antara skor interaksi yang diprediksi dan aktual.
   - Metrics: Dua metrik evaluasi didefinisikan:
     - Root Mean Squared Error (RMSE): Mengukur akar kuadrat dari rata-rata perbedaan kuadrat antara prediksi dan nilai sebenarnya.
     - Mean Absolute Error (MAE): Menghitung rata-rata perbedaan absolut antara prediksi dan nilai sebenarnya.
   - Optimizer: Menggunakan Adam dengan tingkat lr awal 0,001.

- Callback Mechanisms
  
   Selama pelatihan, beberapa mekanisme panggilan balik diterapkan untuk meningkatkan kinerja, mengelola tingkat pembelajaran, dan mencegah overfitting:
   - Learning Rate Scheduler (lr_callback)
   - Model Checkpoints (model_checkpoints)
   - dan Early Stopping (early_stopping)

### 2. Recommender Function
- Item Based Recommendation Function
![image](https://github.com/user-attachments/assets/d862994b-78ff-4248-a601-6762a3dc0ddf)

   Dalam fungsi ini, saya menetapkan ambang batas untuk merekomendasikan hanya anime yang telah dinilai oleh sejumlah pengguna minimum. Ini memastikan bahwa anime yang direkomendasikan telah menerima cukup banyak penilaian, yang mencerminkan tingkat popularitas atau keterlibatan pengguna tertentu. Fungsi ini akan mencari top-N similar animes berdasarkan cosine similarity dari embeddings.

- User Based Recommendation Function

![image](https://github.com/user-attachments/assets/3a9d28a4-d6a7-4525-bd5e-4a8e273fb9e8)
![image](https://github.com/user-attachments/assets/306d60c9-bfa4-464f-8a4f-2f6a8569032a)

   Dalam fungsi ini setidaknya ada 3 tahapan. Pertama fungsi `find_similar_users` akan mencari user yang paling mendekati random user yang di generate. Fungsi ini menghitung tingkat kesamaan melalui weighted matrix dan mengembalikan DataFrame user. Kedua fungsi `get_user_preferences` menggunakan user dari function pertama, dan menganalisis anime-anime yang dinilai bagus oleh si user, dan memvisualisasikan itu kedalam word cloud, yang menjadi gambaran genres apa yang dipreferensikan. Ketiga fungsi `recommend_animes_by_user` akan merekomendasikan judul anime kepada user yang di input sesuai dengan preferensi user yang mendekatinya. FUngsi ini menyaring anime yang telah ditonton oleh pengguna terpilih dan mengidentifikasi rekomendasi baru berdasarkan preferensi pengguna serupa.

## Evaluation

   Berdasarkan model machine learning yang sudah dibangun menggunakan embedding layer dengan Adam optimizer dan mean squared error loss function, metrik yang digunakan untuk mengevaluasi kinerja model adalah **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, dan **Validation Loss**. 

- Root Mean Squared Error (RMSE)

   Perhitungan RMSE dapat dilakukan menggunakan rumus berikut:

$$RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}$$

   Dimana: $$\( N \)$$ adalah jumlah data, $$\( y_i \)$$ adalah nilai observasi (nilai sebenarnya), $$\( \hat{y}_i \)$$ adalah nilai prediksi.

   Hasil nilai RMSE yang rendah menunjukkan bahwa variasi nilai yang dihasilkan dari model sistem rekomendasi mendekati variasi nilai observasinya. Artinya, semakin kecil nilai RMSE, maka akan semakin dekat nilai yang diprediksi dan diamati.

- Mean Absolute Error (MAE)

   Metrik lainnya yang digunakan untuk mengevaluasi model adalah **Mean Absolute Error (MAE)**, yang dapat dihitung menggunakan rumus berikut:

$$MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|$$

   Dimana: $$\( N \)$$ adalah jumlah data, $$\( y_i \)$$ adalah nilai observasi, $$\( \hat{y}_i \)$$ adalah nilai prediksi.

   MAE memberikan gambaran tentang seberapa besar kesalahan rata-rata yang terjadi antara nilai prediksi dan nilai sebenarnya. Nilai MAE yang lebih kecil menunjukkan model yang lebih akurat dalam memprediksi.

- Validation Loss

   **Validation Loss** mengukur kesalahan model pada data yang tidak terlihat sebelumnya selama proses training. Pengukuran ini membantu dalam menilai apakah model mengalami overfitting atau underfitting.

   Pada model yang digunakan, **binary crossentropy** digunakan sebagai fungsi loss, yang menunjukkan seberapa baik model dalam memprediksi kategori yang benar.

   Berikut adalah visualisasi hasil **training** dan **validation error** dari metrik **RMSE**, **MAE**, serta **training** dan **validation loss** dalam bentuk grafik plot:

![image](https://github.com/user-attachments/assets/2d12826e-b66e-43df-9cd4-f8bf3b5e5270)

## Kesimpulan
Dari evaluasi yang dilakukan, ditemukan bahwa:
- Hasil evaluasi model menunjukkan bahwa seluruh metrics penilaian `RMSE`, `MAE`, `Validation Loss`, mengalami perbaikan seiring dengan bertambahnya epoch. hal ini juga dibuktikan dengan hasil pengujian _User Based Recommendation Function_, dan _Item Based Recommendation Function_ yng berhsail memberikan top-N recommendation sebagai output.
- Dalam penelitian ini data telah diproses dan disiapkan dengan baik melalui langkah-langkah seperti _Handling Unknown Values_, _Handling Missing Values_, _Feature Scaling_, _Encoding Identifiers_, _Shuffling and Splitting Data_. Alhasil dapat digunakan dengan baik pada tahap pelatihan dan pengujian.
- Dari penelitian ini dapat dilihat bahwa pendekatan _machine learning_ dalam hal model weight dan fungsi recommender berdampak untuk menginkatkan efesiensi rekomendasi judul anime kepada pengguna berdasarkan preferensi user / judul anime.
     
Evaluasi ini tidak hanya menunjukkan performa teknis tetapi juga relevansi model dengan kebutuhan bisnis, menjawab pertanyaan inti, dan memberikan manfaat ekonomi bagi penyedia platform streaming dan meningkatkan kepuasan pengalaman pengguna.

## Reference
[1] 800 million and growing, why everyone wants a piece of the anime action? -> https://www.smh.com.au/culture/movies/800-million-and-growing-why-everyone-wants-a-piece-of-the-anime-action-20240314-p5fcek.html

[2] Anime Market Size, Share & Trends Analysis Report By Type -> https://www.grandviewresearch.com/industry-analysis/anime-market/toc

[3] Anime Market Research Report 2024 -> https://finance.yahoo.com/news/anime-market-research-report-2024-080300191.html?guccounter=1

[4] netflix recommendation engine worth 1 billion per year -> https://www.businessinsider.com/netflix-recommendation-engine-worth-1-billion-per-year-2016-6

[5] Netflix Recommender system - Case Study -> https://www.linkedin.com/pulse/netflix-recommender-system-case-study-ashish-gupta/
