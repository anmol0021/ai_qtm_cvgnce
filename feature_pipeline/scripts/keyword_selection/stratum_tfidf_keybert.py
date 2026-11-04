#!/usr/bin/env python3
"""Reads a JSON request on stdin, writes a JSON response on stdout.
KeyBERT and TF-IDF run in parallel; output is ordered based on KeyBERT."""
import sys, json, re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def load_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_docs(df, title_col, abstract_col):
    parts = []
    for col in (title_col, abstract_col):
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        return []
    docs = parts[0]
    for p in parts[1:]:
        docs = docs + ". " + p
    return [d.strip() for d in docs.tolist() if len(d.strip()) > 20]


def extract_dois(df, doi_col):
    if doi_col not in df.columns:
        return []
    out = []
    for v in df[doi_col].dropna().astype(str):
        v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v.strip(), flags=re.I)
        if re.match(r"^10\.\d{4,9}/.+$", v):
            out.append(v)
    return sorted(set(out))


def run_tfidf(docs, ngram, min_df, max_df, k=None):
    n = len(docs)
    md = min(min_df, max(1, n))
    vec = TfidfVectorizer(ngram_range=ngram, min_df=md, max_df=max_df,
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(docs)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()

    max_score = float(scores.max()) if len(scores) > 0 and scores.max() > 0 else 1.0
    score_map = {}
    for term, score in zip(terms, scores):
        score_map[str(term).lower()] = float(score / max_score)

    order = scores.argsort()[::-1]
    if k is not None:
        order = order[:k]
    sorted_terms = [{"term": str(terms[i]), "tfidf_score": float(scores[i] / max_score)} for i in order]
    return score_map, sorted_terms


_model_cache = {}
MODEL_ALIASES = {
    "allenai-specter": "sentence-transformers/allenai-specter",
    "specter": "sentence-transformers/allenai-specter",
}


def get_model(name):
    target = MODEL_ALIASES.get(name.lower().strip(), name.strip())
    if target not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[target] = SentenceTransformer(target)
    return _model_cache[target]


def run_keybert(docs, ngram, min_df, k, model_name, diversity=0.5, use_mmr=True):
    model = get_model(model_name)
    cv = CountVectorizer(ngram_range=ngram, min_df=min(min_df, max(1, len(docs))),
                         stop_words="english", max_features=3000)
    cv.fit(docs)
    cands = list(cv.get_feature_names_out())
    if not cands:
        return []
    doc_emb = model.encode(docs, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=False)
    centroid = doc_emb.mean(axis=0, keepdims=True)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    cand_emb = model.encode(cands, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False)
    sims = (cand_emb @ centroid.T).ravel()

    if not use_mmr or diversity <= 0:
        order = np.argsort(sims)[::-1]
        if k is not None:
            order = order[:k]
        return [{"term": str(cands[i]), "keybert_score": float(sims[i])} for i in order]

    selected, remaining = [], list(range(len(cands)))
    while remaining and len(selected) < k:
        if not selected:
            idx = remaining[int(np.argmax(sims[remaining]))]
        else:
            red = (cand_emb[remaining] @ cand_emb[selected].T).max(axis=1)
            mmr = (1 - diversity) * sims[remaining] - diversity * red
            idx = remaining[int(np.argmax(mmr))]
        selected.append(idx)
        remaining.remove(idx)
    return [{"term": cands[i], "keybert_score": float(sims[i])} for i in selected]


def main():
    req = json.load(sys.stdin)
    df = load_table(req["path"])
    docs = build_docs(df, req["title_column"], req["abstract_column"])
    if not docs:
        print(json.dumps({"error": "No usable text found in the selected title/abstract columns."}))
        return

    top_n = max(1, int(req.get("top_n", 50)))
    use_kb = bool(req.get("use_keybert", True))
    # Percentage of combination of TF-IDF with KeyBERT (default: 0.10 / 10%)
    alpha = float(req.get("alpha", req.get("tfidf_share", 0.10)))
    alpha = max(0.0, min(1.0, alpha))
    ngram = (int(req.get("ngram_min", 1)), int(req.get("ngram_max", 3)))
    min_df = int(req.get("min_df", 1))
    max_df = float(req.get("max_df", 0.85))
    model_name = req.get("keybert_model", "allenai-specter")
    warnings = []

    # When alpha is 0, rank purely by KeyBERT cosine similarity in descending order (no MMR shuffling)
    use_mmr = (alpha > 0.0)

    # Run KeyBERT and TF-IDF in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_kb = ex.submit(run_keybert, docs, ngram, min_df, max(top_n * 2, top_n), model_name, 0.5, use_mmr) if use_kb else None
        f_tf = ex.submit(run_tfidf, docs, ngram, min_df, max_df, top_n)

        tfidf_map, tfidf_sorted = f_tf.result()
        kb_list = []
        if f_kb:
            try:
                kb_list = f_kb.result()
            except Exception as e:
                warnings.append(f"KeyBERT failed: {e}")

    # Produce final keywords based on the order of KeyBERT with combining score
    keywords = []
    seen = set()

    if kb_list:
        for item in kb_list:
            term = item["term"]
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            kb_score = float(item["keybert_score"])
            tf_score = float(tfidf_map.get(key, 0.0))
            if alpha <= 0.0:
                # When combination is 0, do not consider TF-IDF at all; consider KeyBERT alone
                comb_score = round(kb_score, 4)
                keywords.append({
                    "term": term,
                    "score": comb_score,
                    "combined_score": comb_score,
                    "keybert_score": round(kb_score, 4),
                    "source": "keybert",
                })
            else:
                # Combining score: alpha% TF-IDF + (1 - alpha)% KeyBERT (default: 10% TF-IDF + 90% KeyBERT)
                comb_score = round(alpha * tf_score + (1.0 - alpha) * kb_score, 4)
                keywords.append({
                    "term": term,
                    "score": comb_score,
                    "combined_score": comb_score,
                    "keybert_score": round(kb_score, 4),
                    "tfidf_score": round(tf_score, 4),
                    "source": "hybrid",
                })
            if len(keywords) >= top_n:
                break

        if alpha <= 0.0:
            # Arrange keywords strictly based on KeyBERT score in descending order
            keywords.sort(key=lambda x: x["keybert_score"], reverse=True)
    else:
        for item in tfidf_sorted:
            term = item["term"]
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            tf_score = float(item["tfidf_score"])
            keywords.append({
                "term": term,
                "score": round(tf_score, 4),
                "combined_score": round(tf_score, 4),
                "tfidf_score": round(tf_score, 4),
                "source": "tfidf",
            })
            if len(keywords) >= top_n:
                break

    print(json.dumps({
        "keywords": keywords,
        "dois": extract_dois(df, req.get("doi_column", "")),
        "warnings": warnings,
        "docs_used": len(docs),
    }))


if __name__ == "__main__":
    main()

