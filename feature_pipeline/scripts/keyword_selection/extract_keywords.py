#!/usr/bin/env python3
"""
Per-paper keyword / concept extraction for the AI x Quantum corpus.

Method = Gu & Krenn (2024), "Forecasting high-impact research topics via machine
learning on evolving knowledge graphs" (Impact4Cast), adapted to our corpus,
+ the hybrid KeyBERT/TF-IDF scorer from stratum-core (api/scripts/tfidf_keybert.py).

Pipeline
  1. Text      : title + ". " + abstract (title only when abstract is missing); LaTeX/HTML stripped.
  2. RAKE      : split every sentence at punctuation + stop words -> candidate phrases
                 (paper, Appendix B). Per-phrase RAKE score = sum(degree(w)/freq(w)).
  3. Vocabulary: keep phrases that recur across the corpus (paper: 2-word >= 9 papers,
                 3+-word >= 6 papers on 2.44M papers; thresholds scaled to our 30k corpus,
                 see --min-df-2 / --min-df-3). Automatic clean-up of non-conceptual phrases
                 (verbs, numbers, generic words). Curated OpenAlex/imputed keywords that recur
                 are added to the vocabulary (single-word concepts such as "qubit" only enter
                 this way, since the paper keeps multi-word concepts).
  4. Concepts  : for every paper, all vocabulary concepts that occur in its title/abstract
                 (paper: this is the per-paper concept set that forms knowledge-graph edges).
  5. Ranking   : stratum-core hybrid score per paper
                     combined = (1-alpha) * KeyBERT + alpha * TF-IDF   (alpha = 0.10)
                 KeyBERT = cosine(SPECTER(concept), SPECTER(paper)); TF-IDF over the corpus,
                 normalised per paper. MMR (diversity 0.5) removes near-duplicate picks.

Outputs (in --out):
  paper_keywords.csv       one row per paper: top-N keywords + full concept set
  paper_keywords_long.csv  one row per (paper, ranked keyword) with all scores
  concept_vocabulary.csv   the concept list (= knowledge-graph vertices) with doc freq / years
  run_config.json          parameters used
"""
import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------------------------------------------------- text
LATEX_RE = re.compile(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]")
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-+']*[a-z0-9+]|[a-z0-9]")
SPLIT_RE = re.compile(r"[.,;:!?()\[\]{}\"“”‘’/|=<>•·–—]|\s-\s|\n")

# Words that act as phrase delimiters in RAKE (in addition to standard stop words):
# reporting verbs, generic academic words, adverbs. They never appear inside a concept.
EXTRA_STOP = set("""
using use used uses based proposed propose proposes present presents presented show shows shown
demonstrate demonstrates demonstrated investigate investigates investigated study studies studied
paper article work works result results finding findings approach approaches introduce introduces
introduced develop develops developed provide provides provided achieve achieves achieved obtain
obtained obtains allow allows allowed enable enables enabled consider considers considered discuss
discussed discusses explore explores explored compare compared compares perform performs performed
apply applies applied review reviews reviewed analyze analyzes analyzed analyse analysed report
reported reports suggest suggests suggested find finds found able possible significantly
significant new novel various different several many various potential important key main
recent recently currently current also however furthermore moreover therefore thus hence
respectively including include includes included particular particularly specific specifically
high-quality well first second third one two three four five here herein within without via
can could may might will would shall should must need needs needed require requires required
help helps make makes made order case cases way ways number numbers type types kind kinds lot
good better best large small higher lower improved improve improves improving efficient efficiently
effective effectively accurate accurately compared comparable existing based-on etc et al i.e e.g ie eg
able achieved overall great greater strong strongly simple simply yet still often typically
""".split())
STOP = set(ENGLISH_STOP_WORDS) | EXTRA_STOP

# Concepts may not start/end with these (cleans RAKE leftovers like "proposed quantum").
BAD_EDGE = STOP | set("""
model models method methods technique techniques system systems process framework analysis data
problem problems performance task tasks set sets level levels value values area areas field fields
aspect aspects application applications experiment experiments simulation simulations
""".split())
# ...but these generic heads are fine at the END of a multi-word phrase ("neural network model").
OK_AT_END = set("""
model models method methods technique techniques system systems process framework analysis data
problem problems performance task tasks simulation simulations algorithm algorithms
""".split())


def clean_text(s):
    s = "" if not isinstance(s, str) else s
    s = URL_RE.sub(" ", s)
    s = TAG_RE.sub(" ", s)
    s = LATEX_RE.sub(" , ", s)  # formula = phrase break
    s = s.replace("\\", " ")
    return s.lower()


def tokenize(s):
    return TOKEN_RE.findall(s)


def singular(w):
    """Very light plural folding for the last word of a phrase (networks -> network)."""
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith(("ss", "us", "is", "ics", "sis", "ous", "ws")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def canon(tokens):
    toks = list(tokens)
    toks[-1] = singular(toks[-1])
    return " ".join(toks)


def valid_token(t):
    if t.isdigit() or len(t) == 1:
        return False
    if re.fullmatch(r"[0-9\-+']+", t):
        return False
    if sum(c.isdigit() for c in t) > len(t) / 2:
        return False
    return True


# --------------------------------------------------------------------------- RAKE
def rake_phrases(text, max_words=5):
    """Return {canonical_phrase: rake_score} and surface forms for one document."""
    phrases = []
    for chunk in SPLIT_RE.split(text):
        cur = []
        for tok in tokenize(chunk):
            if tok in STOP or not valid_token(tok):
                if cur:
                    phrases.append(cur)
                cur = []
            else:
                cur.append(tok)
        if cur:
            phrases.append(cur)
    # word scores (degree / frequency)
    freq, deg = Counter(), Counter()
    for p in phrases:
        for w in p:
            freq[w] += 1
            deg[w] += len(p)
    out, surface = {}, {}
    for p in phrases:
        if len(p) > max_words:
            continue
        sc = sum(deg[w] / freq[w] for w in p)
        c = canon(p)
        if sc > out.get(c, -1):
            out[c] = sc
            surface[c] = " ".join(p)
    return out, surface


# Generic, domain-independent phrases (stand-in for the paper's manual review step).
GENERIC_FIRST = set("""
future further wide broad promising practical real-world numerical theoretical experimental
low lower higher fundamental general common traditional conventional standard typical classical-only
excellent superior remarkable extensive comprehensive detailed deep-insight key-role
""".split())
GENERIC_PHRASES = set("""
physical model|computational cost|computational time|computational resources|state of the art|
proof of concept|case study|real world|wide range|large number|small number|long time|short time|
research direction|research area|research field|research community|open problem|open question|
experimental evidence|experimental data|experimental result|experimental measurement|
numerical result|numerical simulation|numerical calculation|scientific community|growing interest|
great potential|great interest|considerable attention|recent year|past decade|last decade|recent advance|
paper review|literature review|systematic review|comparative analysis|detailed analysis|in-depth analysis
""".replace("\n", "").split("|"))


def is_conceptual(c):
    toks = c.split()
    if c in GENERIC_PHRASES or (len(toks) > 1 and toks[0] in GENERIC_FIRST):
        return False
    if toks[0] in BAD_EDGE:
        return False
    if toks[-1] in BAD_EDGE and not (len(toks) > 1 and toks[-1] in OK_AT_END):
        return False
    if any(t.endswith("ly") and len(t) > 4 for t in toks):  # adverbs
        return False
    if len(c) < 4:
        return False
    return True


# --------------------------------------------------------------------------- n-gram matching
def doc_ngrams(tokens, max_n):
    grams = Counter()
    n_tok = len(tokens)
    for n in range(1, max_n + 1):
        for i in range(n_tok - n + 1):
            grams[canon(tokens[i:i + n])] += 1
    return grams


# --------------------------------------------------------------------------- scoring
def mmr_select(rel, cand_emb, k, diversity):
    if len(rel) == 0:
        return []
    selected, remaining = [], list(range(len(rel)))
    while remaining and len(selected) < k:
        if not selected or diversity <= 0:
            idx = remaining[int(np.argmax(rel[remaining]))]
        else:
            red = (cand_emb[remaining] @ cand_emb[selected].T).max(axis=1)
            mmr = (1 - diversity) * rel[remaining] - diversity * red
            idx = remaining[int(np.argmax(mmr))]
        selected.append(idx)
        remaining.remove(idx)
    return selected


def encode_cached(model, texts, path, batch_size, label):
    if os.path.exists(path):
        emb = np.load(path)
        if emb.shape[0] == len(texts):
            print(f"  [{label}] loaded cached embeddings {emb.shape} from {path}")
            return emb
    t0 = time.time()
    print(f"  [{label}] encoding {len(texts)} texts ...", flush=True)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(path, emb)
    print(f"  [{label}] done in {time.time() - t0:.0f}s")
    return emb


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/ai_quantum_imputed.csv")
    ap.add_argument("--out", default="output/keywords")
    ap.add_argument("--sample", type=int, default=0, help="only process first N papers (testing)")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.10, help="TF-IDF share in hybrid score (stratum default 0.10)")
    ap.add_argument("--diversity", type=float, default=0.5, help="MMR diversity (stratum default 0.5)")
    ap.add_argument("--min-df-2", type=int, default=3, help="min #papers for 2-word concepts (paper: 9 on 2.44M)")
    ap.add_argument("--min-df-3", type=int, default=2, help="min #papers for 3+-word concepts (paper: 6 on 2.44M)")
    ap.add_argument("--min-df-kw", type=int, default=2, help="min #papers for curated keywords to enter vocab")
    ap.add_argument("--max-df", type=float, default=0.50, help="drop concepts in more than this share of papers")
    ap.add_argument("--max-words", type=int, default=5)
    ap.add_argument("--model", default="sentence-transformers/allenai-specter")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=320)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "_cache")
    os.makedirs(cache, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(args.input, low_memory=False)
    if args.sample:
        df = df.head(args.sample).copy()
    df = df.reset_index(drop=True)
    N = len(df)
    print(f"Loaded {N} papers")

    title = df["title"].fillna("").astype(str)
    abstract = df["abstract"].fillna("").astype(str)
    has_abs = abstract.str.strip().str.len() > 0
    raw_docs = (title + ". " + abstract).str.strip()
    docs = [clean_text(d) for d in raw_docs]
    doc_tokens = [tokenize(d) for d in docs]
    years = pd.to_numeric(df["publication_year"], errors="coerce").fillna(0).astype(int).tolist()

    # ---- 1. RAKE candidates per paper + corpus document frequency
    print("RAKE candidate extraction ...")
    rake_per_doc, surf_counter = [], defaultdict(Counter)
    rake_df = Counter()
    for d in docs:
        sc, surf = rake_phrases(d, args.max_words)
        sc = {c: v for c, v in sc.items() if len(c.split()) >= 2 and is_conceptual(c)}
        rake_per_doc.append(sc)
        rake_df.update(sc.keys())
        for c in sc:
            surf_counter[c][surf[c]] += 1

    # ---- 2. curated keywords (OpenAlex / imputed `keywords` column)
    kw_df = Counter()
    if "keywords" in df.columns:
        for s in df["keywords"].fillna(""):
            seen = set()
            for k in str(s).split(";"):
                toks = tokenize(clean_text(k))
                toks = [t for t in toks]
                if not toks or len(toks) > args.max_words:
                    continue
                if not all(valid_token(t) or t in STOP for t in toks):
                    continue
                if toks[0] in STOP or toks[-1] in STOP:
                    continue
                c = canon(toks)
                if c in seen:
                    continue
                seen.add(c)
                kw_df[c] += 1
                surf_counter[c][" ".join(toks)] += 1

    # ---- 3. vocabulary
    vocab_src = {}
    for c, n in rake_df.items():
        nw = len(c.split())
        if (nw == 2 and n >= args.min_df_2) or (nw >= 3 and n >= args.min_df_3):
            vocab_src[c] = "rake"
    for c, n in kw_df.items():
        if n >= args.min_df_kw:
            if len(c.split()) == 1 and (c in BAD_EDGE or len(c) < 3):
                continue
            vocab_src[c] = "rake+keyword" if c in vocab_src else "keyword"
    max_n = max(len(c.split()) for c in vocab_src)

    # ---- 4. per-paper concept matching (exact n-gram match on title+abstract)
    print(f"Vocabulary candidates: {len(vocab_src)}; matching concepts in papers ...")
    concept_df, concept_years = Counter(), defaultdict(list)
    paper_concepts = []
    for toks, y in zip(doc_tokens, years):
        grams = doc_ngrams(toks, max_n)
        found = {c: cnt for c, cnt in grams.items() if c in vocab_src}
        paper_concepts.append(found)
        concept_df.update(found.keys())
        if y:
            for c in found:
                concept_years[c].append(y)

    # drop over-general concepts (appear in > max_df of papers) and drop subsumed ones per paper
    too_common = {c for c, n in concept_df.items() if n > args.max_df * N}
    def keep_concept(c):
        if concept_df.get(c, 0) == 0 or c in too_common:
            return False
        return is_conceptual(c) if len(c.split()) > 1 else c not in BAD_EDGE

    vocab = sorted(c for c in vocab_src if keep_concept(c))
    vocab_idx = {c: i for i, c in enumerate(vocab)}
    print(f"Final concept vocabulary: {len(vocab)} (removed {len(too_common)} over-general)")
    if too_common:
        print("  over-general:", sorted(too_common, key=lambda c: -concept_df[c])[:25])

    def display(c):
        return surf_counter[c].most_common(1)[0][0] if surf_counter[c] else c

    # ---- 5. embeddings (SPECTER, as in stratum-core)
    from sentence_transformers import SentenceTransformer
    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq
    tag = re.sub(r"\W+", "_", args.model.split("/")[-1])
    doc_emb = encode_cached(model, raw_docs.tolist(), os.path.join(cache, f"doc_{tag}_{N}.npy"),
                            args.batch_size, "papers")
    cand_texts = [display(c) for c in vocab]
    cand_emb = encode_cached(model, cand_texts, os.path.join(cache, f"vocab_{tag}_{len(vocab)}.npy"),
                             args.batch_size * 4, "concepts")

    # ---- 6. per-paper hybrid ranking
    print("Ranking keywords per paper ...")
    idf = {c: math.log((1 + N) / (1 + concept_df[c])) + 1 for c in vocab}
    summary_rows, long_rows = [], []
    for i in range(N):
        found = {c: n for c, n in paper_concepts[i].items() if c in vocab_idx}
        # paper-level subsumption: drop "neural" style fragments contained in a longer found concept
        cands = list(found)
        if cands:
            longer = sorted(cands, key=lambda c: -len(c))
            keep = []
            for c in cands:
                if any(c != o and f" {c} " in f" {o} " and found[o] >= found[c] for o in longer):
                    continue
                keep.append(c)
            cands = keep
        if not cands:
            summary_rows.append((i, "", "", 0))
            continue
        idx = np.array([vocab_idx[c] for c in cands])
        ce = cand_emb[idx]
        kb = ce @ doc_emb[i]
        tf = np.array([(1 + math.log(found[c])) * idf[c] for c in cands])
        tf = tf / tf.max() if tf.max() > 0 else tf
        comb = (1 - args.alpha) * kb + args.alpha * tf
        sel = mmr_select(comb, ce, args.top_n, args.diversity)
        sel = sorted(sel, key=lambda j: -comb[j])
        top = []
        for rank, j in enumerate(sel, 1):
            c = cands[j]
            top.append(display(c))
            long_rows.append({
                "id": df.at[i, "id"], "rank": rank, "keyword": display(c), "concept_key": c,
                "combined_score": round(float(comb[j]), 4), "keybert_score": round(float(kb[j]), 4),
                "tfidf_score": round(float(tf[j]), 4), "rake_score": round(rake_per_doc[i].get(c, 0.0), 3),
                "source": vocab_src[c], "corpus_doc_freq": concept_df[c],
            })
        all_c = sorted(cands, key=lambda c: -float(comb[cands.index(c)]))
        summary_rows.append((i, "; ".join(top), "; ".join(display(c) for c in all_c), len(cands)))

    # ---- 7. save
    out = pd.DataFrame({
        "id": df["id"], "doi": df["doi"], "title": df["title"], "publication_year": df["publication_year"],
        "text_source": np.where(has_abs, "title+abstract", "title_only"),
        "top_keywords": [r[1] for r in summary_rows],
        "n_concepts": [r[3] for r in summary_rows],
        "all_concepts": [r[2] for r in summary_rows],
        "original_keywords": df["keywords"] if "keywords" in df.columns else "",
    })
    out.to_csv(os.path.join(args.out, "paper_keywords.csv"), index=False, encoding="utf-8")
    pd.DataFrame(long_rows).to_csv(os.path.join(args.out, "paper_keywords_long.csv"), index=False, encoding="utf-8")

    vrows = []
    for c in vocab:
        ys = concept_years.get(c, [])
        vrows.append({"concept": display(c), "concept_key": c, "n_words": len(c.split()),
                      "source": vocab_src[c], "doc_freq": concept_df[c],
                      "first_year": min(ys) if ys else None, "last_year": max(ys) if ys else None})
    pd.DataFrame(vrows).sort_values("doc_freq", ascending=False).to_csv(
        os.path.join(args.out, "concept_vocabulary.csv"), index=False, encoding="utf-8")

    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({**vars(args), "n_papers": N, "n_vocab": len(vocab),
                   "papers_without_concepts": int((out["n_concepts"] == 0).sum()),
                   "runtime_sec": round(time.time() - t0)}, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
