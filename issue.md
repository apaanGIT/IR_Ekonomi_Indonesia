# 📌 Tracking & Issue Log: Pembuatan Search Engine Ekonomi Indonesia (IR)

Dokumen ini mencatat log implementasi, arsitektur, isu teknis yang ditemukan selama pengerjaan, status resolusi, serta panduan menjalankan projek secara lokal.

---

## 🛠️ Ringkasan Projek & Tech Stack

- **Topik**: Search Engine Berita Ekonomi Indonesia
- **Model Retrieval**: 
  - **BM25** (Parameter terbaik: $k_1 = 1.2$, $b = 0.5$, smoothed IDF)
  - **Semantic Search** (Word2Vec `w2v_idwiki` + Mean Pooling + Cosine Similarity)
- **Backend**: Python Flask (`app.py`, `search_engine.py`, `preprocessing_utils.py`)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Jinja2 Templates)
- **Dataset**: `data/corpus_gabungan.csv` (80 dokumen berita terkurasi)
- **Model Embedding**: `data/w2v_idwiki.bin` (~48 MB binary Word2Vec)

---

## 📋 Task Checklist & Log Eksekusi

| No | Modul / Komponen | File Terkait | Status | Keterangan |
|---|---|---|---|---|
| 1 | Modul Preprocessing | `preprocessing_utils.py` | ✅ Selesai | Diekstrak dari Colab (Sastrawi stemmer, stopword removal, number handling). |
| 2 | Modul IR Engine | `search_engine.py` | ✅ Selesai | Logika BM25, Semantic W2V, dan overlapping calculation terisolasi dalam class `SearchEngine`. |
| 3 | Flask Backend App | `app.py` | ✅ Selesai | Endpoint `/`, `/search`, dan `/detail/<doc_id>` siap melayani request. |
| 4 | Base Template & Footer | `templates/base.html` | ✅ Selesai | Layout umum + footer identitas Kelompok 2 IR Kelas C. |
| 5 | Homepage (Hero Search) | `templates/index.html` | ✅ Selesai | Sesuai prototipe gambar 1 (search box melayang, tombol oranye, clear button). |
| 6 | Results Page (2 Kolom) | `templates/results.html` | ✅ Selesai | Sesuai prototipe gambar 2 (side-by-side BM25 vs Semantic W2V, score normalisasi, ribbon statistik). |
| 7 | Detail Page & Highlighting | `templates/detail.html` | ✅ Selesai | Sesuai prototipe gambar 3-4 (artikel lengkap, badge rank/score/metode, keyword highlighting kuning). |
| 8 | Styling & Visual Assets | `static/css/style.css`, `static/img/bg-hero.jpg` | ✅ Selesai | Font Plus Jakarta Sans, gradient hero blue, card styling, badge tagging. |
| 9 | Client Interactivity | `static/js/main.js` | ✅ Selesai | Fungsi clear input query & multi-word highlight matching di DOM. |
| 10 | Git Initialization & Push | Git Repository | 🔄 In Progress | Inisialisasi repo dan push ke remote `https://github.com/apaanGIT/IR_EkonomiIndonesia`. |

---

## ⚠️ Isu Teknis & Solusi (Issue Log)

### Issue #1: Ekstraksi Preprocessing dari Notebook Colab
- **Gejala**: Kode `preprocessing_utils.py` di file Colab asli terbungkus di dalam baris komentar `# %%writefile preprocessing_utils.py`.
- **Dampak**: Modul tidak dapat langsung di-import jika hanya menjalankan file `.py` hasil export.
- **Solusi**: Dibuat file terpisah `preprocessing_utils.py` dengan logika yang identik (menjaga kata negasi, stopword custom berita, dan regex number filtering).

### Issue #2: Perbedaan Format Skor BM25 vs Semantic W2V
- **Gejala**: BM25 menghasilkan skor relevansi tak berbatas positif ($> 0$), sedangkan Cosine Similarity W2V bernilai $[-1, 1]$ (normalnya $0$ sampai $1$).
- **Dampak**: Pada prototipe UI, keduanya ditampilkan dalam format persentase (`Score: XX%`).
- **Solusi**: 
  - Skor BM25 dinormalisasi secara relatif terhadap skor dokumen ranking 1 pada query tersebut: $\text{Score\%} = \frac{\text{score}}{\text{max\_score}} \times 100\%$.
  - Skor Semantic W2V langsung dikalikan $100\%$ dari nilai cosine similarity.

### Issue #3: Highlighting Kata pada Teks Asli Dokumen
- **Gejala**: Token BM25 melalui proses stemming (misalnya "pertumbuhan" menjadi "tumbuh"), sedangkan teks artikel yang ditampilkan di detail page adalah kalimat utuh (`text_clean`).
- **Dampak**: Jika hanya mengandalkan kata stemmed, pencarian kata asli di artikel bisa meleset.
- **Solusi**: `main.js` mengombinasikan kata kunci query mentah (raw query words) dan token hasil preprocessing, lalu melakukan regex replace berbasis batas kata (`\b`) dengan tag `<mark class="highlight">`.

### Issue #4: Windows App Control / OS DLL Policy pada Python C-Extensions (Local Environment)
- **Gejala**: Eksekusi lokal dengan interpreter Python 3.15 di mesin Windows mengalami `ImportError: DLL load failed while importing vectorized/pcg64: An Application Control policy has blocked this file`.
- **Penyebab**: Kebijakan AppLocker/Windows Defender Application Control pada direktori AppData untuk compiled C-extension (NumPy/Pandas C-DLLs).
- **Catatan untuk Runner**:
  - Jalankan menggunakan virtual environment (`python -m venv venv`) atau pastikan policy memperbolehkan binary C-extension.
  - Alternatif: jalankan di WSL (Windows Subsystem for Linux) atau container Docker jika AppLocker aktif di host Windows.

---

## 🚀 Panduan Menjalankan Aplikasi di Local

1. **Clone repository**:
   ```bash
   git clone https://github.com/apaanGIT/IR_EkonomiIndonesia.git
   cd IR_EkonomiIndonesia
   ```

2. **Buat dan aktifkan Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Linux / Mac / WSL:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan file data tersedia**:
   - `data/corpus_gabungan.csv`
   - `data/w2v_idwiki.bin`

5. **Jalankan web server**:
   ```bash
   python app.py
   ```

6. **Akses di browser**:
   Buka [http://127.0.0.1:5000](http://127.0.0.1:5000)
