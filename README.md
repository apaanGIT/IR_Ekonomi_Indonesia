# 🔍 Search Engine Berita Ekonomi Indonesia (Information Retrieval)

Projek akhir mata kuliah **Information Retrieval (IR)** berbasis corpus berita berbahasa Indonesia dengan topik **Ekonomi Indonesia**. Projek ini mengintegrasikan pengujian model retrieval dari Google Colab ke dalam aplikasi website interaktif yang dapat dijalankan secara lokal.

Repository: [https://github.com/apaanGIT/IR_EkonomiIndonesia](https://github.com/apaanGIT/IR_EkonomiIndonesia)

---

## 🎯 Fitur Utama

1. **Dual Retrieval Engine (Side-by-Side Comparison)**:
   - **BM25 (Best Configuration)**: Menggunakan formula BM25 dengan saturasi TF, parameter $k_1 = 1.2$, $b = 0.5$, dan smoothed IDF.
   - **Semantic Search (Word2Vec)**: Menggunakan model embedding `w2v_idwiki` dengan teknik *Mean Pooling* pada token dokumen & kueri, lalu dihitung kemiripannya menggunakan *Cosine Similarity*.
2. **Statistik Hasil Pencarian**: Menampilkan jumlah dokumen terambil oleh masing-masing metode serta irisan dokumen (*overlapping*) yang ditemukan oleh kedua metode sekaligus.
3. **Keyword Highlighting Interaktif**: Kata kunci dari kueri yang dicari pengguna otomatis di-highlight berwarna kuning di dalam isi artikel lengkap pada halaman detail.
4. **Navigasi & UI Modern**: Tampilan bersih berbasis Flask + HTML/CSS/JS vanilla dengan layout hasil 2 kolom dan halaman detail artikel lengkap beserta metadata sumber media dan tanggal.

---

## 📁 Struktur Direktori

```text
IR_EkonomiIndonesia/
├── app.py                      # Flask Application entry point (routing & controller)
├── search_engine.py            # Modul IR Engine: BM25, Semantic W2V, dan Scoring
├── preprocessing_utils.py      # Pipeline Preprocessing: Sastrawi stemmer, stopwords, tokenisasi
├── requirements.txt            # Daftar library Python yang dibutuhkan
├── README.md                   # Dokumentasi utama projek
├── issue.md                    # Issue tracker & log proses teknis implementasi
├── implementation.md           # Rencana implementasi teknis
├── kelompok_2_ir_ekonomi_indonesia.py # Script pengujian Colab asli
│
├── data/
│   ├── corpus_gabungan.csv     # Dataset 80 dokumen berita terstandarisasi
│   └── w2v_idwiki.bin          # Model embedding Word2Vec binary (~48 MB)
│
├── static/
│   ├── css/
│   │   └── style.css           # Styling antarmuka website
│   ├── js/
│   │   └── main.js             # Script interaktivitas & text highlighting
│   └── img/
│       └── bg-hero.jpg         # Aset gambar background hero
│
└── templates/
    ├── base.html               # Base template Jinja2 (termasuk footer identitas)
    ├── index.html              # Halaman pencarian utama (hero search)
    ├── results.html            # Halaman hasil pencarian (layout 2 kolom)
    └── detail.html             # Halaman detail artikel penuh & kata kunci
```

---

## ⚙️ Cara Menjalankan Projek di Lokal

### 1. Prasyarat
- Python 3.10 atau versi di atasnya
- Git terinstall

### 2. Clone Repository
```bash
git clone https://github.com/apaanGIT/IR_EkonomiIndonesia.git
cd IR_EkonomiIndonesia
```

### 3. Buat dan Aktifkan Virtual Environment (Disarankan)
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux / WSL**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Jalankan Server Flask
```bash
python app.py
```

Setelah server aktif, buka browser dan akses:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

> **Catatan**: Saat pertama kali server dijalankan, sistem memerlukan waktu sekitar 10–30 detik untuk memuat model Word2Vec binary ke dalam memori.

---

## 🔬 Parameter Model Retrieval

- **Preprocessing BM25**:
  - Case Folding $\rightarrow$ URL Removal $\rightarrow$ Tokenisasi $\rightarrow$ Filter Angka (mempertahankan tahun/desimal) $\rightarrow$ Stopword Removal (kata negasi dipertahankan) $\rightarrow$ Stemming Sastrawi.
- **Konfigurasi BM25**:
  - $k_1 = 1.2$
  - $b = 0.5$
  - IDF Smoothed: $\ln\left(1 + \frac{N - \text{df} + 0.5}{\text{df} + 0.5}\right)$
- **Semantic Word2Vec**:
  - Tanpa stemming (agar token cocok dengan kosakata embedding `w2v_idwiki`).
  - Representasi dokumen/kueri: Rata-rata vektor kata (*Mean Pooling*).
  - Scoring: *Cosine Similarity*.

---

## 🛠️(Troubleshooting)

Jika Anda atau rekan lain menemui kendala saat instalasi atau menjalankan aplikasi, berikut panduan solusinya:

### 1. Error `Failed building wheel for gensim / Visual C++ 14.0 required`
- **Penyebab**: Menggunakan versi Python yang terlalu baru (misal Python 3.13 ke atas / 3.15) di Windows di mana file pre-built `.whl` gensim belum tersedia.
- **Solusi**: Gunakan **Python 3.10, 3.11, atau 3.12**. Di versi ini, `pip install -r requirements.txt` akan mengunduh pre-compiled binary secara otomatis tanpa meminta C++ compiler.

### 2. Error `ImportError: DLL load failed ... An Application Control policy has blocked this file`
- **Penyebab**: Kebijakan keamanan Windows Defender Application Control (WDAC) / Device Guard / AppLocker pada perangkat instansi/kantor yang memblokir eksekusi file `.pyd/.dll` di direktori profil user (`AppData` atau `venv`).
- **Solusi**:
  1. Install Python dengan memilih opsi **"Install for all users"** ke direktori `C:\Program Files\Python311` (bukan di `AppData`). Direktori `Program Files` sudah diizinkan (whitelisted) oleh sistem.
  2. Jalankan pip langsung via binary Python utama:
     ```powershell
     python -m pip install -r requirements.txt
     python app.py
     ```
  3. Atau gunakan environment **WSL (Ubuntu Linux)** di mana kebijakan Device Guard Windows tidak berlaku.

