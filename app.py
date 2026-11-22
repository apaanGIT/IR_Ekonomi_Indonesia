"""
app.py — Flask backend untuk demo pencarian IR
Jalankan: python app.py
"""
import os
from flask import Flask, render_template, request
from search_engine import SearchEngine
from preprocessing_utils import preprocess_query

app = Flask(__name__)

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
    method = request.args.get("method", "bm25")
    rank = request.args.get("rank", "")
    score = request.args.get("score", "")

    doc = engine.get_document(doc_id)
    if doc is None:
        return "Dokumen tidak ditemukan", 404

    if method == "bm25":
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
