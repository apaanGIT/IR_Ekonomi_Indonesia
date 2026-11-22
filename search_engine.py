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
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 1.0

        # Hitung DF untuk BM25
        self.df_terms = Counter(
            t for ts in self.tokenized_corpus for t in set(ts)
        )

        # Load Word2Vec embedding
        self.wv = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

        # Preprocessing khusus Semantic (tanpa stemming)
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
        Return: list of dict [{doc_id, title, media, score, ...}, ...], q_tokens
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
        """Preprocessing untuk W2V: tanpa stemming, sesuai Colab."""
        text = re.sub(r"https?://\S+|www\.\S+", " ", str(text).lower())
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
        Return: list of dict [{doc_id, title, media, score, ...}, ...], q_tokens
        """
        q_tokens = self._preprocess_w2v(query)
        q_vec = self._mean_pooling(q_tokens).reshape(1, -1)
        scores = cosine_similarity(q_vec, self.X_w2v).ravel()

        top_indices = np.argsort(-scores, kind="stable")[:top_k]

        results = []
        for i in top_indices:
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
    # Gabungan: search()
    # ═══════════════════════════════════════

    def search(self, query, top_k=20):
        """
        Jalankan BM25 dan Semantic W2V secara bersamaan.
        Return dict berisi kedua hasil + statistik.
        """
        bm25_results, bm25_tokens = self.search_bm25(query, top_k)
        w2v_results, w2v_tokens = self.search_w2v(query, top_k)

        bm25_ids = set(r["doc_id"] for r in bm25_results)
        w2v_ids = set(r["doc_id"] for r in w2v_results)
        overlap = bm25_ids & w2v_ids
        only_bm25 = len(bm25_ids - w2v_ids)
        only_w2v = len(w2v_ids - bm25_ids)

        return {
            "query": query,
            "bm25_tokens": bm25_tokens,
            "w2v_tokens": w2v_tokens,
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
