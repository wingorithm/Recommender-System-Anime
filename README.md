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

2. **Multivariate Analysis**

![image](https://github.com/user-attachments/assets/e7a3d89b-341d-4069-ae59-9f7a24f60fa9)
![image](https://github.com/user-attachments/assets/9d32e446-35e3-42b5-af57-9f7e4e763e2c)


## Data Preparation

## Modeling

## Evaluation

## Reference
[1] 800 million and growing, why everyone wants a piece of the anime action? - https://www.smh.com.au/culture/movies/800-million-and-growing-why-everyone-wants-a-piece-of-the-anime-action-20240314-p5fcek.html

[2] Anime Market Size, Share & Trends Analysis Report By Type - https://www.grandviewresearch.com/industry-analysis/anime-market/toc

[3] Anime Market Research Report 2024 - https://finance.yahoo.com/news/anime-market-research-report-2024-080300191.html?guccounter=1

[4] netflix recommendation engine worth 1 billion per year - https://www.businessinsider.com/netflix-recommendation-engine-worth-1-billion-per-year-2016-6

[5] Netflix Recommender system - Case Study - https://www.linkedin.com/pulse/netflix-recommender-system-case-study-ashish-gupta/
