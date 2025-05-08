# RTC KELOMPOK 7 - IMPLEMENTASI ALGORITMA SVM, K-NN, DAN NEURAL NETWORK UNTUK MONITORING KUALITAS UDARA

Supervisor : Ahmad Radhy, S.Si., M.Si

Nama Kelompok :

Muhammad Misbahul Huda - 2042221008
Calrin Norika - 2042221026
Christian Candra Winata - 2042221065

**Monitoring Polution** ini berisi kumpulan modul yang dibuat dengan bahasa Rust, dan fungsinya untuk mengklasifikasi kualitas udara. Di dalamnya digunakan tiga algoritma utama, yaitu *Neural Network*, *Support Vector Machine* (SVM), dan *k-Nearest Neighbor* (kNN). Beberapa modul juga sudah terhubung dengan Qt supaya bisa menampilkan antarmuka yang interaktif dan gampang digunakan oleh pengguna.

**Panduan Pengerjaan Proyek Monitoring Polution**

1. instal ubuntu, rust, QT
2. melakukan pemrograman Rust sine dan cosine menggunakan pendekatan taylor series
3. melakukan pemrograman Rust sine dan cosine menggunakan lookup table
4. melakukan pemrograman Rust Machine Learning dengan metode Support Vector Machine
5. (SVM) dan kNN dengan menginput kan dataset
6. Melakukan pemrograman Rust Neural Network
7. Membuat arsitektur Neural Network 
8. Membuat module class, function, structure, program Rust Project
9. Program ini dapat di jalankan di Ubuntu Linux
10. Membuat akun GitHub
11. Pengaplikasian Neural Network dengan backend menggunakan Rust dan Frontend menggunakan QT untuk aplikasi desktop 
12. Memasukkan seluruh data pemrograman ke akun GitHub
13. Membuat Laporan
14. Membuat PPT
    
**Analisa Hasil Pembuatan Proyek**

•	Efisiensi Bahasa Pemrograman Rust: Rust terbukti efisien dan cepat dalam memproses data selama eksperimen berlangsung.
•	Penggunaan Rust memerlukan ketelitian tinggi, khususnya saat memakai library eksternal.
•	Sintaks Rust bisa bervariasi tergantung pada fitur dari library yang digunakan (sebagian, keseluruhan, atau kombinasi tertentu).
•	Pemahaman dokumentasi setiap library sangat penting agar integrasi berjalan optimal.
•	Klasifikasi Data dengan SVM-KNN: Metode SVM-KNN diuji untuk mengukur akurasi terhadap dataset yang tersedia.
•	Hasil menunjukkan akurasi yang relatif rendah pada metode SVM.
•	Upaya peningkatan performa dilakukan melalui:
•	Normalisasi data (mengurutkan nilai dari kecil ke besar).
•	Penyesuaian parameter gamma.
•	Penyesuaian gamma tidak memberikan dampak signifikan, kemungkinan karena pemilihan nilai gamma yang kurang tepat.
•	Implementasi Neural Network dengan Qt: Neural Network dikembangkan menggunakan framework Qt dan menunjukkan hasil lebih baik dibanding SVM-KNN.
•	Meskipun sintaks Qt untuk antarmuka pengguna berbeda dengan framework lain, adaptasi tetap dapat dilakukan dengan baik.
•	Perbandingan Dataset 5000 vs 8000 Data: Dua dataset digunakan untuk pengujian: satu dengan 5000 data dan satu lagi dengan 8000 data.
•	Dataset 5000 data menghasilkan akurasi yang lebih tinggi dibanding dataset 8000 data.
•	Kemungkinan disebabkan oleh dataset 8000 data yang merupakan hasil augmentasi di Python, sehingga sebagian data menjadi kurang realistis dan menurunkan akurasi model.
