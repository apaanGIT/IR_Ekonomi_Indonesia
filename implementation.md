# 📋 Implementation Plan: Website Search Demo — Ekonomi Indonesia

> **Projek**: Information Retrieval — Kelompok 2 IR Kelas C
> **Tujuan**: Mengubah notebook Colab (BM25 + Semantic Word2Vec) menjadi website search demo yang berjalan di localhost
> **Tech Stack**: Python Flask (backend) + HTML/CSS/JS vanilla (frontend)

---

## 📐 Arsitektur Sistem

```
d:\Ahmad\Kampus\TKI\Web\
├── app.py                      # Flask backend (entry point)
├── search_engine.py            # Logika IR: BM25 + Semantic W2V
├── preprocessing_utils.py      # Modul preprocessing (dari Colab)
├── requirements.txt            # Daftar dependency Python
│
├── data/
│   ├── corpus_gabungan.csv     # Dataset 80 dokumen berita
│   └── w2v_idwiki_*.bin        # Model Word2Vec embedding
│
├── static/
│   ├── css/
│   │   └── style.css           # Styling semua halaman
│   ├── js/
│   │   └── main.js             # Interaksi frontend (AJAX search, navigasi)
│   └── img/
│       └── bg-hero.jpg         # Background hero section (peta Indonesia)
│
├── templates/
│   ├── base.html               # Base template (header, footer)
│   ├── index.html              # Halaman pencarian awal
│   ├── results.html            # Halaman hasil pencarian (2 kolom)
│   └── detail.html             # Halaman detail artikel
│
└── kelompok_2_ir_ekonomi_indonesia.py  # File asli Colab (referensi)
```

---

## 🔧 Fase 1: Setup Environment & Persiapan Data

### 1.1 Requirements (`requirements.txt`)

```txt
flask>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
PySastrawi>=1.2
gensim>=4.3
```

### 1.2 Instalasi

```bash
cd d:\Ahmad\Kampus\TKI\Web
pip install -r requirements.txt
```

### 1.3 Persiapan Data

1. **Pindahkan file data ke folder `data/`**:
   - `corpus_gabungan.csv` → `data/corpus_gabungan.csv`
   - `w2v_idwiki_2310511059_2310511076_2310511085_2310511089.bin` → `data/w2v_idwiki.bin` (rename agar lebih simpel)

2. **Buat file `preprocessing_utils.py`** (copy dari kode yang tertulis di notebook section 3.1):

> [!IMPORTANT]
> File `preprocessing_utils.py` HARUS dibuat sendiri karena di file Colab hanya ada dalam bentuk komentar (`# %%writefile preprocessing_utils.py`). Salin kode yang ada di baris 231–273 file Colab, hilangkan tanda `#` di depan setiap baris.

```python
"""Preprocessing bersama untuk Project Akhir IR (Kelompok 2).
Dipakai untuk dokumen DAN query agar konsisten.

    from preprocessing_utils import preprocess, preprocess_query
"""
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

NEGASI = {"tidak", "bukan", "jangan", "tanpa", "belum"}            # dipertahankan
EXTRA_STOPWORDS = {"ujar", "kata", "katanya", "mengatakan", "menjelaskan", "jelas",
                   "dikutip", "menurut", "kepada", "ilustrasi"}     # stopword khas berita
STOPWORDS = (set(StopWordRemoverFactory().get_stop_words()) - NEGASI) | EXTRA_STOPWORDS

_stemmer = StemmerFactory().create_stemmer()
_cache = {}
def stem(w):
    if w not in _cache:
        _cache[w] = _stemmer.stem(w)
    return _cache[w]

KEEP_NUMBERS = True
TOKEN_RE = r"\d+(?:[.,]\d+)+|[a-z0-9]+"

def tokenize(text):
    return re.findall(TOKEN_RE, text.lower())

def filter_number(t):
    if not t[0].isdigit():
        return True
    if not KEEP_NUMBERS:
        return False
    return bool(re.fullmatch(r"(19|20)\d{2}", t)) or ("," in t or "." in t)

def preprocess(text):
    """case folding -> tokenisasi -> filter angka -> stopword -> stemming"""
    tokens = [t for t in tokenize(text) if filter_number(t)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) if t.isalpha() else t for t in tokens]

def preprocess_query(q):
    """Query WAJIB memakai fungsi yang sama dengan dokumen."""
    return preprocess(q)
```

---

## 🔧 Fase 2: Backend — Modul Search Engine (`search_engine.py`)

File ini mengekstrak logika pencarian dari notebook Colab menjadi **class Python murni** yang bisa dipanggil oleh Flask.

### 2.1 Struktur Class

```python
"""
search_engine.py — Modul pencarian BM25 + Semantic Word2Vec
Mengekstrak logika dari kelompok_2_ir_ekonomi_indonesia.py
"""
import math
import re
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from preprocessing_utils import preprocess_query


class SearchEngine:
    def __init__(self, corpus_path, embedding_path):
        """Load corpus dan model saat startup."""
        self.docs = pd.read_csv(corpus_path, parse_dates=["date"])
        self.docs["tokens"] = self.docs["text_final"].fillna("").str.split()
        self.tokenized_corpus = self.docs["tokens"].tolist()
        self.N = len(self.tokenized_corpus)
        self.doc_lengths = [len(t) for t in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / self.N

        # Hitung DF untuk BM25
        self.df_terms = Counter(
            t for ts in self.tokenized_corpus for t in set(ts)
        )

        # Load Word2Vec embedding
        self.wv = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

        # Preprocessing khusus Semantic (tanpa stemming)
        # Siapkan matriks dokumen Word2Vec
        self._build_w2v_index()

        # Konfigurasi BM25 terbaik (dari eksperimen Colab)
        self.BM25_K1 = 1.2
        self.BM25_B = 0.5
        self.BM25_SMOOTH = True

    # ═══════════════════════════════════════
    # BM25
    # ═══════════════════════════════════════

    def _idf_bm25(self, term):
        freq = self.df_terms[term]
        ratio = (self.N - freq + 0.5) / (freq + 0.5)
        return math.log(1 + ratio)  # smoothed

    def _satf(self, f, dl):
        K = self.BM25_K1 * (1 - self.BM25_B + self.BM25_B * dl / self.avgdl)
        return f * (self.BM25_K1 + 1) / (f + K)

    def _bm25_score(self, q_terms, doc_tokens):
        counts = Counter(doc_tokens)
        score = 0.0
        for term in q_terms:
            if counts[term] > 0:
                score += self._idf_bm25(term) * self._satf(counts[term], len(doc_tokens))
        return score

    def search_bm25(self, query, top_k=20):
        """
        Pencarian BM25.
        Return: list of dict [{doc_id, title, media, score, snippet, ...}, ...]
        """
        q_tokens = preprocess_query(query)
        vocabulary = {t for ts in self.tokenized_corpus for t in ts}
        q_terms = [t for t in q_tokens if t in vocabulary]

        if not q_terms:
            return [], q_tokens

        results = []
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            if not any(t in doc_tokens for t in q_terms):
                continue
            score = self._bm25_score(q_terms, doc_tokens)
            if score > 0:
                row = self.docs.iloc[idx]
                results.append({
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "media": row["media"],
                    "date": str(row["date"].date()) if pd.notna(row["date"]) else "",
                    "author": row.get("author", ""),
                    "score": round(score, 4),
                    "text_clean": row["text_clean"],
                    "url": row.get("url", ""),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k], q_tokens

    # ═══════════════════════════════════════
    # Semantic Word2Vec
    # ═══════════════════════════════════════

    def _preprocess_w2v(self, text, buang_stopword=True):
        """Preprocessing untuk W2V: tanpa stemming, mirip P12a."""
        text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
        tokens = re.findall(r"[a-z0-9]+", text)
        tokens = [t for t in tokens if not t.isdigit()]
        if buang_stopword:
            KATA_NEGASI = {"tidak", "bukan", "jangan", "tanpa", "belum"}
            STOPWORDS_W2V = set(StopWordRemoverFactory().get_stop_words()) - KATA_NEGASI
            tokens = [t for t in tokens if t not in STOPWORDS_W2V]
        return tokens

    def _mean_pooling(self, tokens):
        vektor = [self.wv[t] for t in tokens if t in self.wv]
        if not vektor:
            return np.zeros(self.wv.vector_size, dtype=np.float32)
        return np.mean(vektor, axis=0)

    def _build_w2v_index(self):
        """Pre-compute matriks dokumen Word2Vec."""
        self.docs["tokens_w2v"] = self.docs["text_clean"].fillna("").apply(self._preprocess_w2v)
        self.X_w2v = np.vstack([
            self._mean_pooling(toks) for toks in self.docs["tokens_w2v"]
        ])

    def search_w2v(self, query, top_k=20):
        """
        Pencarian Semantic Word2Vec.
        Return: list of dict [{doc_id, title, media, score, ...}, ...]
        """
        q_tokens = self._preprocess_w2v(query)
        q_vec = self._mean_pooling(q_tokens).reshape(1, -1)
        scores = cosine_similarity(q_vec, self.X_w2v).ravel()

        # Ambil top_k
        top_indices = np.argsort(-scores, kind="stable")[:top_k]

        results = []
        for rank, i in enumerate(top_indices, 1):
            row = self.docs.iloc[i]
            results.append({
                "doc_id": row["doc_id"],
                "title": row["title"],
                "media": row["media"],
                "date": str(row["date"].date()) if pd.notna(row["date"]) else "",
                "author": row.get("author", ""),
                "score": round(float(scores[i]), 4),
                "text_clean": row["text_clean"],
                "url": row.get("url", ""),
            })

        return results, q_tokens

    # ═══════════════════════════════════════
    # Gabungan: search() → dipanggil oleh Flask
    # ═══════════════════════════════════════

    def search(self, query, top_k=20):
        """
        Jalankan BM25 dan Semantic W2V secara bersamaan.
        Return dict berisi kedua hasil + statistik.
        """
        bm25_results, bm25_tokens = self.search_bm25(query, top_k)
        w2v_results, w2v_tokens = self.search_w2v(query, top_k)

        # Hitung statistik perbandingan
        bm25_ids = set(r["doc_id"] for r in bm25_results)
        w2v_ids = set(r["doc_id"] for r in w2v_results)
        overlap = bm25_ids & w2v_ids
        only_bm25 = len(bm25_ids - w2v_ids)
        only_w2v = len(w2v_ids - bm25_ids)

        return {
            "query": query,
            "bm25_tokens": bm25_tokens,
            "bm25_results": bm25_results,
            "w2v_results": w2v_results,
            "stats": {
                "bm25_count": len(bm25_results),
                "w2v_count": len(w2v_results),
                "overlap": len(overlap),
                "only_bm25": only_bm25,
                "only_w2v": only_w2v,
            }
        }

    def get_document(self, doc_id):
        """Ambil detail lengkap satu dokumen berdasarkan doc_id."""
        row = self.docs[self.docs["doc_id"] == doc_id]
        if row.empty:
            return None
        row = row.iloc[0]
        return {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "media": row["media"],
            "date": str(row["date"].date()) if pd.notna(row["date"]) else "",
            "author": row.get("author", ""),
            "text_clean": row["text_clean"],
            "url": row.get("url", ""),
        }
```

> [!NOTE]
> `StopWordRemoverFactory()` dipanggil dua kali (satu di `preprocessing_utils.py`, satu di `search_engine.py`). Ini by design karena preprocessing BM25 (dengan stemming) dan W2V (tanpa stemming) berbeda pipeline-nya. Pembuatan stemmer dan stopword list hanya terjadi sekali saat startup, jadi tidak mempengaruhi performa.

---

## 🔧 Fase 3: Backend — Flask App (`app.py`)

```python
"""
app.py — Flask backend untuk demo pencarian IR
Jalankan: python app.py
"""
from flask import Flask, render_template, request, jsonify
from search_engine import SearchEngine
import os

app = Flask(__name__)

# Inisialisasi search engine (sekali saat startup)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
engine = SearchEngine(
    corpus_path=os.path.join(BASE_DIR, "data", "corpus_gabungan.csv"),
    embedding_path=os.path.join(BASE_DIR, "data", "w2v_idwiki.bin"),
)
print(f"✅ Search engine siap! {engine.N} dokumen dimuat.")


@app.route("/")
def index():
    """Halaman utama — form pencarian."""
    return render_template("index.html")


@app.route("/search")
def search():
    """Halaman hasil pencarian."""
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("index.html")

    data = engine.search(query, top_k=20)
    return render_template("results.html", data=data)


@app.route("/detail/<doc_id>")
def detail(doc_id):
    """Halaman detail artikel."""
    query = request.args.get("q", "")
    method = request.args.get("method", "bm25")  # "bm25" atau "w2v"
    rank = request.args.get("rank", "")
    score = request.args.get("score", "")

    doc = engine.get_document(doc_id)
    if doc is None:
        return "Dokumen tidak ditemukan", 404

    # Ambil token query untuk highlighting
    if method == "bm25":
        from preprocessing_utils import preprocess_query
        q_tokens = preprocess_query(query)
    else:
        q_tokens = engine._preprocess_w2v(query)

    return render_template("detail.html",
                           doc=doc,
                           query=query,
                           method=method,
                           rank=rank,
                           score=score,
                           q_tokens=q_tokens)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
```

---

## 🔧 Fase 4: Frontend — Templates HTML

### 4.1 Base Template (`templates/base.html`)

Template dasar yang dipakai semua halaman. Berisi:
- Tag `<head>` dengan link ke CSS
- Header **"Ekonomi Indonesia"** (teks putih, font bold)
- Footer berisi nama kelompok + NIM
- Block `content` yang di-override tiap halaman

```html
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Ekonomi Indonesia{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    {% block content %}{% endblock %}

    <footer>
        <p><strong>By: Kelompok 2 IR Kelas C</strong></p>
        <div class="footer-members">
            <span>2310511059 - Muhammad Rizki Ramadhan Siregar</span>
            <span>2310511085 - Mauleni Risqi Handayani</span>
            <span>2310511076 - Delas Princisa Hakim</span>
            <span>2310511089 - Ahmad Jiddan</span>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

### 4.2 Halaman Pencarian (`templates/index.html`)

Sesuai **Gambar 1** prototipe:
- Full-screen hero section dengan background peta Indonesia + kalkulator
- Judul besar **"Ekonomi Indonesia"** di tengah
- Search bar besar dengan placeholder "Cari artikel disini..."
- Tombol "Cari" berwarna oranye
- Tombol ✕ untuk clear input

```html
{% extends "base.html" %}
{% block content %}
<section class="hero">
    <div class="hero-overlay">
        <h1 class="hero-title">Ekonomi Indonesia</h1>
        <form class="search-box" action="/search" method="GET">
            <div class="search-input-wrapper">
                <span class="search-icon">🔍</span>
                <input type="text" name="q" placeholder="Cari artikel disini..."
                       autocomplete="off" id="search-input">
                <button type="button" class="clear-btn" onclick="clearSearch()">✕</button>
            </div>
            <button type="submit" class="btn-cari">Cari</button>
        </form>
    </div>
</section>
{% endblock %}
```

### 4.3 Halaman Hasil (`templates/results.html`)

Sesuai **Gambar 2** prototipe:
- Header tetap menampilkan judul dan search bar (dengan query terisi)
- **Statistik pencarian**: "Query: ...", "BM25: X", "Semantic W2V: Y", "Beririsan: Z"
- **Dua kolom**: kiri = BM25, kanan = Semantic W2V
- Tiap kolom berisi kartu-kartu hasil (card)
- Tiap card menampilkan: ranking (`#1`), skor (`Score: 90%`), sumber media + doc_id, judul (truncated), snippet teks

```html
{% extends "base.html" %}
{% block content %}
<section class="results-header">
    <h1 class="results-title">Ekonomi Indonesia</h1>
    <form class="search-box search-box-sm" action="/search" method="GET">
        <div class="search-input-wrapper">
            <span class="search-icon">🔍</span>
            <input type="text" name="q" value="{{ data.query }}" id="search-input">
            <button type="button" class="clear-btn" onclick="clearSearch()">✕</button>
        </div>
        <button type="submit" class="btn-cari">Cari</button>
    </form>
</section>

<!-- Statistik -->
<div class="stats-bar">
    <span>Query: <strong>{{ data.query }}</strong></span>
    <span class="stat-badge stat-bm25">BM25: {{ data.stats.bm25_count }}</span>
    <span class="stat-badge stat-w2v">Semantic W2V: {{ data.stats.w2v_count }}</span>
    <span class="stat-badge stat-overlap">Beririsan: {{ data.stats.overlap }}</span>
</div>

<!-- Dua kolom hasil -->
<div class="results-columns">
    <!-- Kolom BM25 -->
    <div class="column">
        <div class="column-header bm25-header">BM25</div>
        {% for r in data.bm25_results %}
        <a href="/detail/{{ r.doc_id }}?q={{ data.query|urlencode }}&method=bm25&rank={{ loop.index }}&score={{ r.score }}"
           class="result-card">
            <div class="card-top">
                <span class="rank">#{{ loop.index }}</span>
                <span class="score">Score: {{ (r.score * 100 / (data.bm25_results[0].score if data.bm25_results else 1))|round(0) }}%</span>
                <span class="source">📰 {{ r.media }} • {{ r.doc_id }}</span>
            </div>
            <h3 class="card-title">{{ r.title }}</h3>
            <p class="card-snippet">{{ r.text_clean[:150] }}...</p>
        </a>
        {% endfor %}
    </div>

    <!-- Kolom Semantic W2V -->
    <div class="column">
        <div class="column-header w2v-header">Semantic W2V</div>
        {% for r in data.w2v_results %}
        <a href="/detail/{{ r.doc_id }}?q={{ data.query|urlencode }}&method=w2v&rank={{ loop.index }}&score={{ r.score }}"
           class="result-card">
            <div class="card-top">
                <span class="rank">#{{ loop.index }}</span>
                <span class="score">Score: {{ (r.score * 100)|round(0) }}%</span>
                <span class="source">📰 {{ r.media }} • {{ r.doc_id }}</span>
            </div>
            <h3 class="card-title">{{ r.title }}</h3>
            <p class="card-snippet">{{ r.text_clean[:150] }}...</p>
        </a>
        {% endfor %}
    </div>
</div>
{% endblock %}
```

> [!NOTE]
> **Normalisasi skor untuk tampilan:**
> - BM25: skor ditampilkan sebagai persentase relatif terhadap skor tertinggi (top-1 = 100%)
> - Semantic W2V: cosine similarity sudah 0–1, langsung dikonversi ke persentase

### 4.4 Halaman Detail Artikel (`templates/detail.html`)

Sesuai **Gambar 3–4** prototipe:
- Header tetap: judul + search bar
- Tombol **"← Kembali ke hasil pencarian"** (navigasi back)
- **Badge info**: metode (BM25 / Semantic W2V), rank, score
- Metadata: ID, judul lengkap (tanpa truncate), nama penulis, tanggal
- **Isi artikel penuh** dengan **keyword highlighting** (teks query yang match diberi background kuning)
- Di bawah artikel: **"Kata kunci kueri:"** + badge tiap token query

```html
{% extends "base.html" %}
{% block content %}
<section class="detail-header">
    <h1 class="results-title">Ekonomi Indonesia</h1>
    <form class="search-box search-box-sm" action="/search" method="GET">
        <div class="search-input-wrapper">
            <span class="search-icon">🔍</span>
            <input type="text" name="q" value="{{ query }}" readonly>
        </div>
    </form>
</section>

<div class="detail-container">
    <!-- Navigasi & badge -->
    <div class="detail-nav">
        <a href="/search?q={{ query|urlencode }}" class="btn-back">← Kembali ke hasil pencarian</a>
        <div class="detail-badges">
            <span class="badge badge-{{ method }}">{{ 'BM25' if method == 'bm25' else 'Semantic W2V' }}</span>
            <span class="badge badge-rank">#{{ rank }}</span>
            <span class="badge badge-score">Score: {{ score }}</span>
        </div>
    </div>

    <!-- Metadata -->
    <p class="detail-id">ID: {{ doc.doc_id }}</p>
    <h2 class="detail-title">{{ doc.title }}</h2>
    <div class="detail-meta">
        <span class="author">✍️ {{ doc.author or 'Nama Penulis' }}</span>
        <span class="date">{{ doc.date }}</span>
    </div>

    <!-- Isi artikel dengan highlighting -->
    <div class="detail-content" id="article-content">
        {{ doc.text_clean }}
    </div>

    <!-- Kata kunci query -->
    <div class="query-keywords">
        <strong>Kata kunci kueri:</strong>
        {% for token in q_tokens %}
        <span class="keyword-badge">{{ token }}</span>
        {% endfor %}
    </div>
</div>

<!-- Script highlighting -->
<script>
    // Highlight kata kunci query dalam teks artikel
    const keywords = {{ q_tokens | tojson }};
    highlightKeywords("article-content", keywords);
</script>
{% endblock %}
```

---

## 🔧 Fase 5: Frontend — CSS (`static/css/style.css`)

### Panduan Desain (dari prototipe)

| Elemen | Warna / Style |
|--------|---------------|
| Background hero | Gambar peta Indonesia + kalkulator/uang, overlay gelap semi-transparan |
| Header results | Background biru navy gradient (`#1a237e` → `#283593`) |
| Tombol "Cari" | Oranye (`#ff9800`), rounded |
| Search bar | Background putih, rounded, shadow |
| BM25 header | Biru solid (`#1565c0`) |
| W2V header | Biru solid (`#1565c0`), border berbeda |
| Card hasil | Background putih, rounded, shadow, border-left biru |
| Score badge | Oranye kecil |
| Rank badge | Biru tua |
| Highlighting | Background kuning (`#ffeb3b`) |
| Keyword badge | Background oranye muda, rounded |
| Footer | Background biru tua, teks putih |
| Font | Sans-serif (system font atau Google Fonts: Inter/Roboto) |

### Key CSS Classes yang Harus Dibuat

```
.hero               → full-screen hero section
.search-box          → wrapper form pencarian (flex row)
.search-input-wrapper → input + icon + clear button
.btn-cari            → tombol Cari (oranye)
.results-columns     → grid 2 kolom (CSS Grid atau Flexbox)
.column-header       → header kolom BM25 / Semantic W2V
.result-card         → kartu hasil (block link, hover effect)
.card-top            → baris atas kartu (rank, score, source)
.detail-container    → wrapper konten detail
.detail-content      → teks artikel
.highlight           → class untuk kata yang di-highlight (kuning)
.keyword-badge       → badge kata kunci di bawah artikel
.stats-bar           → bar statistik perbandingan
```

---

## 🔧 Fase 6: Frontend — JavaScript (`static/js/main.js`)

```javascript
/**
 * main.js — Interaksi frontend
 */

// Clear search input
function clearSearch() {
    document.getElementById("search-input").value = "";
    document.getElementById("search-input").focus();
}

// Highlight kata kunci dalam teks artikel
function highlightKeywords(elementId, keywords) {
    const el = document.getElementById(elementId);
    if (!el || !keywords || keywords.length === 0) return;

    let html = el.textContent; // ambil teks mentah

    // Escape regex special chars, lalu build pattern
    keywords.forEach(kw => {
        const escaped = kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        // Highlight semua kemunculan (case-insensitive)
        const regex = new RegExp(`(${escaped})`, 'gi');
        html = html.replace(regex, '<mark class="highlight">$1</mark>');
    });

    el.innerHTML = html;
}
```

> [!WARNING]
> **Highlighting token stemmed vs teks asli:**
> Token query BM25 sudah di-stem (mis. "pertumbuhan" → "tumbuh"). Jadi highlight di `text_clean` (yang belum di-stem) tidak akan match persis. Ada dua strategi:
>
> **Opsi A (Simpel):** Highlight kata ASLI dari query (sebelum preprocessing), bukan token yang sudah di-stem. Ini yang paling mudah diimplementasikan.
>
> **Opsi B (Akurat):** Untuk tiap token stemmed, cari semua kata di teks yang ketika di-stem menghasilkan token tersebut, lalu highlight kata-kata itu. Ini lebih kompleks.
>
> **Rekomendasi: Gunakan Opsi A** — cukup split query asli menjadi kata-kata, lalu highlight masing-masing. Untuk demo ini sudah cukup informatif.

Untuk **Opsi A**, ubah bagian highlighting di `detail.html`:

```html
<script>
    // Highlight kata ASLI dari query (bukan token stemmed)
    const rawQuery = "{{ query }}";
    const rawKeywords = rawQuery.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    highlightKeywords("article-content", rawKeywords);
</script>
```

---

## 🔧 Fase 7: Background Image

Untuk hero section di halaman utama:
1. **Simpan gambar background** ke `static/img/bg-hero.jpg`
2. Gambar yang cocok: peta Indonesia dengan overlay kalkulator/uang (sesuai prototipe)
3. Jika tidak ada gambar, gunakan CSS gradient sebagai fallback:

```css
.hero {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    /* Jika ada gambar: */
    /* background: url('/static/img/bg-hero.jpg') center/cover no-repeat; */
}
```

---

## ✅ Fase 8: Testing & Menjalankan

### 8.1 Menjalankan Aplikasi

```bash
cd d:\Ahmad\Kampus\TKI\Web
python app.py
```

Buka browser → `http://127.0.0.1:5000`

### 8.2 Checklist Testing

| # | Test Case | Expected |
|---|-----------|----------|
| 1 | Buka halaman utama | Tampil hero section dengan search bar |
| 2 | Ketik "pertumbuhan ekonomi" → klik Cari | Redirect ke `/search?q=pertumbuhan+ekonomi`, tampil 2 kolom hasil |
| 3 | Cek statistik bar | Menampilkan jumlah BM25, W2V, dan beririsan dengan benar |
| 4 | Klik satu kartu hasil BM25 | Redirect ke halaman detail dengan badge "BM25", rank, score |
| 5 | Klik satu kartu hasil W2V | Redirect ke halaman detail dengan badge "Semantic W2V", rank, score |
| 6 | Cek highlighting di detail | Kata-kata query ter-highlight kuning di teks artikel |
| 7 | Klik "← Kembali ke hasil pencarian" | Kembali ke halaman results dengan query yang sama |
| 8 | Cek tombol ✕ di search bar | Input ter-clear |
| 9 | Submit query kosong | Tetap di halaman utama, tidak error |
| 10 | Query yang tidak ada match sama sekali | Halaman results tampil kosong tapi tidak crash |
| 11 | Cek footer | Nama kelompok + NIM tampil di semua halaman |

### 8.3 Known Issue & Mitigasi

| Issue | Penyebab | Solusi |
|-------|----------|--------|
| Startup lambat (30-60 detik) | Loading file embedding W2V (~500MB) | Normal untuk demo lokal. Bisa ditambahkan print progress saat loading |
| Highlighting tidak presisi | Token stemmed ≠ kata asli | Gunakan Opsi A (highlight kata asli query) |
| Skor BM25 bukan persentase | BM25 menghasilkan skor absolut | Normalisasi: `skor / skor_tertinggi * 100` |
| Beberapa artikel author kosong | Data corpus tidak lengkap | Tampilkan fallback "Penulis tidak tersedia" |

---

## 📊 Diagram Alur Interaksi

```
┌─────────────┐     GET /          ┌──────────────┐
│   Browser   │ ──────────────────→│  index.html  │
│   (User)    │                    │  (Hero +     │
│             │     submit form    │   Search)    │
│             │ ──────────────────→│              │
└──────┬──────┘    GET /search?q=  └──────────────┘
       │                                  │
       │                                  ▼
       │                          ┌──────────────┐
       │                          │  app.py      │
       │                          │  search()    │
       │                          │              │
       │                          │  engine.     │
       │                          │  search()    │
       │                          └──────┬───────┘
       │                                 │
       │                    ┌────────────┴────────────┐
       │                    ▼                         ▼
       │            ┌──────────────┐         ┌──────────────┐
       │            │  BM25        │         │  Semantic    │
       │            │  search      │         │  W2V search  │
       │            └──────────────┘         └──────────────┘
       │                    │                         │
       │                    └────────────┬────────────┘
       │                                 ▼
       │                          ┌──────────────┐
       │                          │ results.html │
       │       klik card          │ (2 kolom)    │
       │ ◄────────────────────────│              │
       │    GET /detail/<id>      └──────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                  ┌──────────────┐
│  detail.html │                  │  Highlight   │
│  (Artikel    │◄─────────────────│  Keywords    │
│   lengkap)   │   JS highlight   │  (main.js)   │
└──────────────┘                  └──────────────┘
```

---

## 📝 Catatan untuk Implementor

1. **Jangan ubah logika IR** — algoritma BM25 dan Word2Vec harus persis sama dengan yang ada di notebook Colab. Perbedaan hasil = bug.

2. **Urutan kerja yang disarankan**:
   - Fase 1 (setup) → Fase 2 (search_engine.py) → Fase 3 (app.py) → test via terminal dulu (`curl http://localhost:5000/search?q=test`)
   - Baru setelah backend jalan → Fase 4–6 (frontend)

3. **File yang sudah ada** dan TIDAK BOLEH dimodifikasi:
   - `corpus_gabungan.csv` (data)
   - `w2v_idwiki_*.bin` (model)
   - `kelompok_2_ir_ekonomi_indonesia.py` (referensi)

4. **Referensi line notebook** untuk masing-masing fungsi:
   - BM25 IDF → baris 504–516
   - BM25 satf → baris 551–560
   - BM25 score → baris 719–725
   - BM25 ranking → baris 727–741
   - BM25 konfigurasi final → baris 1019–1021 (`k1=1.2, b=0.5, smooth=True`)
   - Semantic preprocessing → baris 1871–1891
   - Mean pooling W2V → baris 1934–1938
   - Skor W2V → baris 1944–1945

5. **Responsivitas**: Tidak wajib responsive untuk mobile. Website ini hanya demo di laptop/PC.

6. **Tidak perlu database**: Semua data dimuat ke memory dari CSV saat startup.
