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
