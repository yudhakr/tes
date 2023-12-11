# Laporan Proyek Machine Learning - Ferry Ardianto Rismawan

## Project Overview

<p align="center">
  <img width="300" src="https://user-images.githubusercontent.com/44900042/140647265-979928c9-a9dd-4ed4-800e-297c44f42e04.png" alt="Sumber : https://www.kaggle.com/andrewmvd/udemy-courses">
</p>

Pada proyek ini, akan dibuat sistem rekomendasi kursus online untuk pengguna di Udemy. Udemy merupakan platform penyedia kursus online terbuka secara masif dari Amerika yang ditujukan untuk orang dewasa dan pelajar profesional. Di era yang serba digital ini, banyaknya informasi dan sumber daya yang tersedia sesuai kebutuhan (_on demand_). Tidak terkecuali dalam kasus memilih kursus online. Permintaan dunia kerja yang semakin beragam mendorong manusia untuk terus belajar dan mengasah kemampuan. Namun, dengan adanya banyak sekali informasi dan sumber daya tentu membuat kita menghabiskan waktu hanya untuk memilih apa yang ingin dipelajari [[1](https://doi.org/10.1016/j.procs.2017.12.067)]. 

Tidak sedikit pengguna yang belajar melalui kursus online adalah orang yang baru memasuki bidang tersebut. Oleh karena itu, pengguna merasa bingung memilih kursus yang harus diikuti selanjutnya [[2](https://doi.org/10.1109/CIST.2018.8596516)]. Oleh karena itu diperlukan sebuah sistem rekomendasi agar pengguna yang baru saja belajar bidang-bidang baru bisa mendapatkan rekomendsi kursus online yang sesuai dengan bidang minat mereka. Selain sebagai sarana periklanan, sistem rekomendasi juga membuat kursus online yang baru atau jarang diikuti oleh pengguna menjadi dikenal karena sebelumnya sulit untuk ditemukan atau bahkan mempermudah pengguna menemukan kursus yang diharapkan. 


## Business Understanding

### Problem Statements

Berdasarkan permasalahan di atas, berikut rumusan masalah yang perlu diselesaikan pada proyek ini:
- Sistem rekomendasi apa yang sesuai untuk diterapkan pada kasus ini?
- Bagaimana cara membuat sistem rekomendasi kursus online untuk pengguna di Udemy?

### Goals

Berikut tujuan dari proyek ini:
- Membuat sistem rekomendasi kursus online untuk pengguna di Udemy.
- Memberikan rekomendasi untuk kursus online yang kemungkinan disukai pengguna.

### Solution Approach

Solusi yang dapat dilakukan untuk mencapai tujuan proyek ini diantaranya:
- Untuk pra-pemrosesan data dilakukan beberapa teknik diantaranya:
  - Membersihkan data duplikasi.
  - Membersihkan teks judul pada kolom course_title dari stopwords.
  - Membersihkan teks judul pada kolom course_title dari karakter spesial.

  Visualisasi data dapat dilihat lebih lengkap di bagian _Data Understanding_.

- Untuk persiapan data (sebelum dimasukkan ke model) dilakukan Vektorisasi menggunakan _TF-IDF Vectorizer_ untuk ekstraksi fitur pada teks judul kursus.

- Pembuatan sistem rekomendasi dilakukan dengan pendekatan _content-based filtering_ berdasarkan dataset yang ada. Sehingga sistem rekomendasi dibuat untuk memberikan rekomendasi pada pengguna terhadap kursus online yang sebelumnya diikuti/dibeli. Beberapa algoritma yang digunakan untuk membuat sistem rekomendasi di proyek ini diantaranya:
  - Sistem rekomendasi berbasis model, yakni dengan algoritma K-Nearest Neighbor. Algoritma tersebut dipilih karena lebih mudah diaplikasikan dan cukup sesuai untuk kasus klasterisasi di sistem rekomendasi. Algoritma ini berasumsi bahwa suatu data yang serupa memiliki kedekatan. Cara kerja dari algoritma ini adalah sebagai berikut (diterjemahkan dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]):
    - Muat datanya.
    - Inisialisasi nilai K (banyak tetangga/kelompok).
    - Pada setiap data:
      - Hitung _euclidean distance_ antara kueri yang diberikan dan contoh yang ada pada data tersebut dengan rumus berikut:
        ![Rumus Euclidean Distance](https://user-images.githubusercontent.com/44900042/140647363-4f4595c1-1c5a-4ef9-b4a3-e247014bdca2.png)
      - Tambahkan jarak dan urutan dari contoh pada koleksi yang berurutan.
    - Pilih entri K paling awal pada koleksi yang berurutan.
    - Dapatkan label dari entri K yang dipilih.
    - Apabila kasus regresi, kembalikan nilai rata-ratanya. Apabila kasus klasifikasi, kembalikan nilai labelnya.

    Algoritma ini digunakan karena memiliki kelebihan dan kekurangan sebagai berikut (diterjemahkan dari [[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)]):
    - Kelebihan:
      - Algoritma yang mudah digunakan dan sederhana.
      - Algoritma yang sangat fleksibel, dapat diimplementasikan pada kasus klasifikasi, regresi dan pencarian.
    - Kekurangan:
      - Algoritmanya menjadi lebih lambat secara signifikan karena jumlah sampel/contoh dan/atau prediktor/variabel yang meningkat.

  - Sistem rekomendasi berbasis algoritma _cosine similarity_. Algoritma ini dipilih karena relatif mudah digunakan dan digunakan sebagai pembanding sistem rekomendasi yang menggunakan model. _Cosine similarity_ secara singkat, digunakan untuk mengukur kemiripan antara dua buah vektor dan kesamaan arahnya dengan cara menghitung nilai sudut kosinus dari kedua vektor. Rumus yang digunakan sebagai berikut:

    ![Rumus Cosine Similarity](https://user-images.githubusercontent.com/44900042/140647871-12f552ff-d2a3-42a9-ad39-91aeb86bf831.png)

    Nilai x, y adalah nilai vektor dan k adalah nilai _cosine similarity_ dari vektor x dan y.

## Data Understanding

![Sampul Dataset](https://user-images.githubusercontent.com/44900042/140647912-203efabb-a1e2-45bc-b5a7-2d56332a1533.png)

Tabel di bawah ini merupakan informasi dari dataset yang digunakan:

| Jenis                   | Keterangan                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| Sumber                  | [Kaggle Dataset: Udemy Courses](https://www.kaggle.com/andrewmvd/udemy-courses)                  |
| Lisensi                 | License was not specified at source                                                              |
| Kategori                | Bisnis, Edukasi, Komunitas Online                                                                |
| Rating Penggunaan       | 10.0 (Gold)                                                                                      |
| Jenis dan Ukuran Berkas | zip (694 kB)                                                                                     |


Gambar di bawah ini merupakan sampel dari dataset pada berkas `udemy_courses.csv`:

![Pratinjau udemy_courses.csv](https://user-images.githubusercontent.com/44900042/140647923-56a34071-011e-42bd-8fd3-8ed734d53771.png)

Kemudian gambar di bawah ini merupakan informasi dataset pada berkas `udemy_courses.csv`:

<img width="300" src="https://user-images.githubusercontent.com/44900042/140647931-c262e982-59d2-48e3-a249-42d75d0d639b.png" alt="Informasi udemy_courses.csv">

Berkas `udemy_courses.csv` berisi informasi mengenai detail kursus online yang ada di aplikasi Udemy. Data yang ada relatif bersih dan tidak terdapat nilai kosong. Berikut ini adalah uraian variabel dari setiap kolom pada dataset:

  1. Kolom `course_id` merupakan kolom dengan data id dari kursus.
  1. Kolom `course_title` merupakan kolom dengan data judul dari kursus.
  1. Kolom `url` merupakan kolom dengan data url dari kursus.
  1. Kolom `is_paid` merupakan kolom dengan data jenis kursus yang hanya berisi tipe data boolean yang menyatakan kursus tersebut berbayar atau gratis.
  1. Kolom `price` merupakan kolom dengan data harga dari kursus dalam satuan dollar.
  1. Kolom `num_subscribers` merupakan kolom dengan data jumlah pengguna yang berlangganan pada setiap kursus.
  1. Kolom `num_reviews` merupakan kolom dengan data jumlah ulasan pengguna yang berlangganan pada setiap kursus.
  1. Kolom `num_lectures` merupakan kolom dengan data jumlah pengajar pada setiap kursus.
  1. Kolom `level` merupakan kolom dengan data tingkat kesulitan dari setiap kursus.
  1. Kolom `content_duration` merupakan kolom dengan data durasi konten pada setiap kursus.
  1. Kolom `publised_timestamp` merupakan kolom dengan data waktu publikasi kursus.
  1. Kolom `subject` merupakan kolom dengan data subjek yang diajarkan pada kursus.

  Kumpulan gambar di bawah ini merupakan visualisasi dari dataset yang digunakan:
  - Data Numerik
  
    ![Visualisasi price + is_paid](https://user-images.githubusercontent.com/44900042/140647958-b3434dc0-0647-4591-9c9d-1a9f25064283.png)
    
    ![Visualisasi num_subscribers + is_paid](https://user-images.githubusercontent.com/44900042/140647964-8607d621-e86c-4260-bf79-ea0163ae3682.png)
    
    ![Visualisasi num_reviews + is_paid](https://user-images.githubusercontent.com/44900042/140647969-a378810d-eb0f-4bd8-9346-1561e8268a85.png)
    
    ![Visualisasi num_lectures + is_paid](https://user-images.githubusercontent.com/44900042/140647978-830abc4a-25d3-4031-b01c-4f5b09f44001.png)
    
    ![Visualisasi content_duration + is_paid](https://user-images.githubusercontent.com/44900042/140647984-06631362-09c8-49b6-b767-04b0383f7a88.png)
    
    ![Visualisasi price + subject + is_paid](https://user-images.githubusercontent.com/44900042/140648128-dc971dd3-6696-49ad-8632-f6cdf6715b62.png)
    
    ![Visualisasi num_subscribers + subject + is_paid](https://user-images.githubusercontent.com/44900042/140648131-fb6dda36-c879-4820-bff7-782b5212820b.png)
    
    ![Visualisasi num_reviews + subject + is_paid](https://user-images.githubusercontent.com/44900042/140648132-69283077-e59c-4b94-9b95-7343ff1b6393.png)
    
    ![Visualisasi num_lectures + subject + is_paid](https://user-images.githubusercontent.com/44900042/140648133-d66636e3-ab14-44b6-b43e-bf9c7f9f7e94.png)
    
    ![Visualisasi content_duration + subject + is_paid](https://user-images.githubusercontent.com/44900042/140648134-92268cae-2441-4059-8b0e-399b1c20c1e2.png)

  - Data Kategori
    
    ![Visualisasi level + is_paid + num_subscribers](https://user-images.githubusercontent.com/44900042/140648231-2b7931a3-0ff8-4f5c-96e5-766dd898be8f.png)
    
    ![Visualisasi subject + is_paid + num_subscribers](https://user-images.githubusercontent.com/44900042/140648234-498a34ca-3cc7-4a2c-9e50-23576337eb67.png)


## Data Preparation

Seperti yang sudah dijelaskan pada bagian _Solution approach_, berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data:
- Membersihkan data duplikasi. Hal ini dilakukan karena data duplikat dapat menyebabkan munculnya redundansi dalam hasil sistem rekomendasi yang akan dibuat. Oleh karena itu data duplikasi ini perlu dihilangkan karena data tersebut sudah terdapat dalam dataset. Proses ini dilakukan dengan menggunakan fungsi `drop_duplicates` dari _dataframe_ dataset.
- Membersihkan teks judul pada kolom `course_title` dari `stopwords`. Hal ini dilakukan untuk mencegah redundansi pada data teks judul dengan cara menghapus informasi tingkat rendah sehingga sistem rekomendasi nantinya dapat fokus pada informasi yang lebih penting. Selain itu, menghapus `stopwords` dapat mengurangi ukuran dataset. Proses ini dilakukan dengan menggunakan fungsi `remove_stopwords` pada modul [neattext](https://blog.jcharistech.com/neattext/).
- Membersihkan teks judul pada kolom course_title dari karakter spesial. Hal ini dilakukan untuk mencegah kebingungan sistem rekomendasi dengan cara menghapus karakter khusus yang memiliki informasi rendah. Proses ini dilakukan dengan menggunakan fungsi `remove_special_characters` pada modul [neattext](https://blog.jcharistech.com/neattext/).
- Konversi teks judul yang telah dibersihkan menjadi vektor TF-IDF. Hal ini dilakukan untuk melakukan ekstraksi fitur pada teks judul kursus yang nantinya akan dikonversi menjadi vektor dengan nilai numerik. Proses ini dilakukan dengan menggunakan fungsi [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) pada modul scikit-learn. Proses perhitungannya yaitu:
  - Menghitung nilai _Term Frequency_ dari sebuah kata atau kalimat dalam dokumen. Salah satu cara yang paling sederhana adalah menghitung jumlah awal kata/kalimat yang muncul dalam dokumen kemudian menyesuaikan frekuensi berdasarkan panjang dokumen. Secara matematis, nilainya akan dihitung dengan rumus berikut:
  
    ![Rumus Term Frequency](https://user-images.githubusercontent.com/44900042/140648290-7dd59d6d-e23a-43f6-b82e-068f3e8a62f3.png)

  - Menghitung nilai _Inverse Document Frequency_ dari sebuah kata/kalimat dalam satu set dokumen. Semakin dekat nilainya ke 0 maka semakin umum sebuah kata/kalimat. Metrik ini dirumuskan sebagai berikut:
    
    ![Rumus Inverse Document Frequency](https://user-images.githubusercontent.com/44900042/140648306-6fa1e544-1219-4559-a5b0-d1e314d2ef5b.png)

  - Menghitung nilai TF-IDF. Hal ini dilakukan dengan cara mengalikan nilai TF dengan nilai IDF untuk menentukan seberapa relevan kata/kalimat tersebut dalam suatu dokumen. Secara matematis dirumuskan sebagai berikut:
    
    ![Rumus TF-IDF](https://user-images.githubusercontent.com/44900042/140648332-7f079785-0e4a-434f-adca-f5e6d5a74f6a.png)


## Modeling

Setelah dilakukan pra-pemrosesan data, tahap selanjutnya adalah membuat sistem rekomendasi dengan pendekatan _content-based filtering_.

  1. Menggunakan model K-Nearest Neighbor

     Untuk membangun model klasterisasi ini, digunakan fungsi [NearestNeighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) dari scikit-learn dengan parameter metriks _euclidean_. Fungsi tersebut kemudian diinisiasikan sebagai model yang selanjutnya dilakukan _fitting_ terhadap data yang ada pada _dataframe_. Kemudian dibuat fungsi `getRecommendedCourses_model` untuk memberikan rekomendasi terhadap suatu judul kursus online dengan skenario, apabila pengguna membeli/berlangganan kursus online tersebut, maka berikan rekomendasi ini sebagai kursus online yang mungkin disukai. Hasil rekomendasinya adalah sebagai berikut:
     
     <img width="500" src="https://user-images.githubusercontent.com/44900042/140648359-6ab8dfbc-9a4d-461e-acfa-dd7ff4261b1a.png" alt="Rekomendasi KNN">

  1. Menggunakan _cosine similarity_

     Metode selanjutnya, sistem rekomendasi diberikan dengan cara menghitung nilai _cosine similarity_ dari vektor TF-IDF dari judul kursus pada dataset menggunakan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari scikit-learn. Implementasinya dengan cara memanggil fungsi `cosine_similarity` dengan argumen _dataframe_ sebagai objeknya. Kemudian hasil dari perhitungannya disimpan pada _dataframe_ baru. Untuk proses pemberian rekomendasi, dibuat fungsi `getRecommendedCourses_cosine` yang akan memberikan rekomendasi terhadap suatu judul kursus online dengan skenario yang sama dengan sebelumnya.

     Proses pada fungsi tersebut ialah, melakukan pencarian kolom dari suatu judul kursus pada _dataframe_ baru hasil perhitungan _cosine similarity_. Lalu diurutkan nilainya berdasarkan nilai _cosine similarity_ tertinggi dan juga urutannya. Setiap urutan ke-2 terakhir hingga ke-n terakhir merupakan kandidat yang memiliki nilai _cosine similarity_ yang sama maka akan ditampilkan rekomendasinya. Urutan paling akhir merupakan nilai _cosine similarity_ dari kolom dengan judul kursus yang sama. Untuk lebih jelasnya, hasil rekomendasi dapat dilihat di bawah ini:
     
     <img width="500" src="https://user-images.githubusercontent.com/44900042/140648372-2f302793-2305-42fe-a5ec-982b2ec07e61.png" alt="Rekomendasi Cosine Similarity">


## Evaluation

Untuk mengukur kinerja sistem rekomendasi dengan model KNN dan _cosine similarity_ digunakan metriks _precision_.

_Precision_ adalah metrik yang dapat digunakan pada kasus klasterisasi untuk menghitung jumlah item rekomendasi yang relevan (_similar_) dengan kategori item yang dipilih. Perhitungan nilai _precision_ dapat menggunakan rumus berikut [[4](https://towardsdatascience.com/evaluating-clustering-results-f13552ee7603)].

<img width="500" src="https://user-images.githubusercontent.com/44900042/140648402-043ceb0f-d5a8-463c-9b89-f5b399feb1af.png" alt="Rumus Precision">

Metriks ini memiliki kelebihan untuk berfokus pada bagaimana performa klasterisasi model terhadap data yang relevan _(similar)_, namun kekurangannya metrik ini tidak memperhitungkan data yang kurang relevan. Selain itu, metrik ini juga terbatas pada permasalahan klasterisasi biner [[5](https://machinelearninginterview.com/topics/machine-learning/evaluation-metrics-for-recommendation-systems/)].

Penerapan pada kode dilakukan secara manual. Fungsi yang dibuat menerima argumen berupa kueri input yang nantinya akan dicocokan dengan hasil sistem rekomendasi berdasarkan subjeknya. Berikut adalah hasil implementasinya.
```python
# Fungsi untuk menghitung nilai presisi dari sistem rekomendasi
def precision(query:pd.DataFrame, rec_result:pd.DataFrame):
  relevant = 0
  for result in rec_result['subject'].values.tolist():
    if query['subject'].values == result:
      relevant += 1
  return relevant/len(rec_result)
```

- Nilai _precision_ model KNN
  
  ![Skor Precision KNN](https://user-images.githubusercontent.com/44900042/140648427-5f224fe4-2904-4fd3-a893-e905ac976c7f.png)

- Nilai _precision_ algoritma _cosine similarity_

  ![Skor Precision Cosine Similarity](https://user-images.githubusercontent.com/44900042/140648436-5085290d-c33a-4260-a1de-89eb6c2ab0ad.png)
  

Pada model ini, nampak bahwa nilai _precision_ dari model KNN sudah cukup baik dengan skor mencapai 90% dan 70% pada sistem rekomendasi yang menggunakan _cosine similarity_. Hal ini memungkinkan rekomendasi kursus online lebih sesuai dengan kursus online yang telah dibeli/dipelajari oleh pengguna.


# Referensi

[[1](https://www.sciencedirect.com/science/article/pii/S1877050917328314)] 
Z. Gulzar, A. A. Leema, en G. Deepak, “PCRS: Personalized Course Recommender System Based on Hybrid Approach”, Procedia Computer Science, vol 125, bll 518–524, 2018. https://doi.org/10.1016/j.procs.2017.12.067

[[2](https://ieeexplore.ieee.org/abstract/document/8596516)] K. Dahdouh, L. Oughdir, A. Dakkak, en A. Ibriz, “Smart Courses Recommender System for Online Learning Platform”, in 2018 IEEE 5th International Congress on Information Science and Technology (CiSt), 2018, bll 328–333. https://doi.org/10.1109/CIST.2018.8596516

[[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)] Harrison, O. (2019, July 14). _Machine Learning Basics with the K-Nearest Neighbors Algorithm_. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

[[4](https://towardsdatascience.com/evaluating-clustering-results-f13552ee7603)] Mallawaarachchi, Vijini. (2020, June 09). Evaluating Clustering Results. https://towardsdatascience.com/evaluating-clustering-results-f13552ee7603

[[5](https://machinelearninginterview.com/topics/machine-learning/evaluation-metrics-for-recommendation-systems/)] MLNerds. (2021, July 23). Evaluation Metrics for Recommendation Systems. https://machinelearninginterview.com/topics/machine-learning/evaluation-metrics-for-recommendation-systems/

**---Ini adalah bagian akhir laporan---**
