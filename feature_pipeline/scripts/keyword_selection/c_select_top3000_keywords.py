#!/usr/bin/env python3
"""
Step 3: final top-3000 keywords from the 300 anchor papers.

Scoring  = stratum-core (api/scripts/tfidf_keybert.py, imported unmodified):
           combined = 0.9 * KeyBERT + 0.1 * TF-IDF, KeyBERT = cosine(SPECTER(candidate),
           centroid of SPECTER(anchor papers)); TF-IDF = stratum run_tfidf over the anchors;
           MMR with diversity 0.5 (stratum default), implemented incrementally (same maths).
Candidates = Gu & Krenn (2024) concept extraction instead of stratum's CountVectorizer
           (which caps candidates at the 3000 most frequent stop-word-stripped n-grams, so
           asking it for 3000 keywords returns every candidate, incl. 'likely', 'apply',
           'review quantum machine'):
             - RAKE phrases (2-5 words) from anchor titles+abstracts, cleaned of
               non-conceptual / generic phrases (same filters as scripts/extract_keywords.py)
             - single words only if they are curated concepts in the corpus vocabulary
               (e.g. 'qubit', 'entanglement'), since the paper keeps multi-word concepts
           Recurrence filter (paper keeps recurring concepts): the phrase must occur in >= 2
           anchor papers, or in >= 2 papers of the 30k corpus; if that leaves < 3000, all
           candidates go to stratum ranking (recurrence is reported per keyword).
"""
import importlib.util, os, sys, json
from collections import Counter, defaultdict
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, HERE)
import extract_keywords as ek  # RAKE + cleaning helpers (same folder)

spec = importlib.util.spec_from_file_location("stratum", os.path.join(HERE, "stratum_tfidf_keybert.py"))
stratum = importlib.util.module_from_spec(spec); spec.loader.exec_module(stratum)

TOP_N, ALPHA, DIVERSITY = 3000, 0.10, 0.5

# extra phrase delimiters / generic words found when reviewing the first anchor run
ek.STOP |= set("""holds hold utilizing utilize utilizes integrates integrate integrating exploiting exploit
exploits aims aim aiming leveraging leverage leverages offers offer offering enabling harnessing harness
promises promise remains remain employing employ employs combining combine combines yields yield
emerging emerged addressing address addresses lies lie validates validate validated outperforms
outperform outperformed explain explains explained revealed reveal reveals written build builds learn
learns classify classifies capturing capture captures experienced programming solve solves solving
achieving exhibits exhibit predicts predict""".split())
ek.BAD_EDGE |= ek.STOP
ek.GENERIC_FIRST |= set("valuable transformative certain natural alternative crucial vital key".split())
ek.GENERIC_PHRASES |= {"knowledge gap", "certain degree", "valuable guidance", "transformative impact",
                       "learning outcome", "natural framework", "alternative framework"}

anch = pd.read_csv(os.path.join(ROOT, "data", "quantum_x_ai_300_anchors.xlsx.csv"))
docs = stratum.build_docs(anch, "Article Title", "Abstract")          # stratum's own text builder
print("anchor docs:", len(docs))

# ---- candidates (paper's RAKE)
anchor_df, surf = Counter(), defaultdict(Counter)
for d in docs:
    sc, s = ek.rake_phrases(ek.clean_text(d), max_words=5)
    keep = [c for c in sc if len(c.split()) >= 2 and ek.is_conceptual(c)]
    anchor_df.update(keep)
    for c in keep: surf[c][s[c]] += 1

# curated single-word concepts from the corpus vocabulary
voc = pd.read_csv(os.path.join(ROOT, "output", "keywords", "concept_vocabulary.csv"))
corpus_df = dict(zip(voc.concept_key, voc.doc_freq))
uni_ok = set(voc[(voc.n_words == 1) & voc.source.str.contains("keyword")].concept_key)
doc_tok_grams = [ek.doc_ngrams(ek.tokenize(ek.clean_text(d)), 1) for d in docs]
for g in doc_tok_grams:
    for c in g:
        if c in uni_ok:
            anchor_df[c] += 1; surf[c][c] += 1

# corpus doc frequency for every candidate (exact phrase match in 30k corpus)
pap = pd.read_parquet(os.path.join(OUT, "_work", "papers_citations.parquet"), columns=["text"])
cand_set = set(anchor_df)
cdf = Counter()
for t in pap.text:
    toks = ek.tokenize(ek.clean_text(t))
    cdf.update({g for g in ek.doc_ngrams(toks, 5) if g in cand_set})
cands = list(anchor_df)
recurring = np.array([anchor_df[c] >= 2 or cdf[c] >= 2 for c in cands])
print(f"RAKE+unigram candidates: {len(cands)}; recurring (tier 1): {int(recurring.sum())}")
display = [surf[c].most_common(1)[0][0] for c in cands]

# ---- stratum scoring
tfidf_map, _ = stratum.run_tfidf(docs, (1, 5), 1, 0.85)
model = stratum.get_model("allenai-specter")
import pickle
cache_p = os.path.join(OUT, "_work", "specter_text_cache.pkl")
cache = pickle.load(open(cache_p, "rb")) if os.path.exists(cache_p) else {}
def enc(texts, bs):
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        for t, e in zip(todo, model.encode(todo, batch_size=bs, normalize_embeddings=True, show_progress_bar=True)):
            cache[t] = e.astype(np.float32)
        pickle.dump(cache, open(cache_p, "wb"))
    return np.stack([cache[t] for t in texts])
doc_emb = enc(docs, 32)
centroid = doc_emb.mean(axis=0); centroid /= np.linalg.norm(centroid)
cand_emb = enc(display, 128)
kb = cand_emb @ centroid
# stratum's TF-IDF tokenizer drops stop words / short tokens -> look up the same form
from sklearn.feature_extraction.text import TfidfVectorizer
an = TfidfVectorizer(ngram_range=(1, 5), stop_words="english").build_analyzer()
tf = np.array([tfidf_map.get(" ".join(an(x)), 0.0) if an(x) else 0.0 for x in display])
comb = (1 - ALPHA) * kb + ALPHA * tf

# ---- MMR (stratum: relevance = KeyBERT sim; red = max sim to already selected)
# tier 1 (recurring phrases) is exhausted first, then tier 2 fills the remaining slots
n = len(cands); k = min(TOP_N, n)
selected, max_red = [], np.full(n, -np.inf)
avail = recurring.copy()
for step in range(k):
    if not avail.any():
        avail = ~recurring & ~np.isin(np.arange(n), selected)
    if step == 0:
        score = kb.copy()
    else:
        score = (1 - DIVERSITY) * kb - DIVERSITY * max_red
    score[~avail] = -np.inf
    i = int(np.argmax(score)); selected.append(i); avail[i] = False
    max_red = np.maximum(max_red, cand_emb @ cand_emb[i])

rows = [{"keyword": display[i], "concept_key": cands[i], "n_words": len(cands[i].split()),
         "combined_score": round(float(comb[i]), 4), "keybert_score": round(float(kb[i]), 4),
         "tfidf_score": round(float(tf[i]), 4), "mmr_pick_order": r + 1,
         "anchor_doc_freq": anchor_df[cands[i]], "corpus_doc_freq": cdf[cands[i]],
         "recurring": bool(recurring[i])}
        for r, i in enumerate(selected)]
out = pd.DataFrame(rows).sort_values(["recurring", "combined_score"], ascending=[False, False])
out.insert(0, "rank", range(1, len(out) + 1))
out.to_csv(os.path.join(OUT, "top3000_keywords.csv"), index=False, encoding="utf-8")
print(out.head(40).keyword.tolist())
print(out.n_words.value_counts().to_dict(), "| corpus df==0:", int((out.corpus_doc_freq == 0).sum()))
