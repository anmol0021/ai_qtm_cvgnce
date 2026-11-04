#!/usr/bin/env python3
"""
Step 2: evolving, citation-augmented knowledge graph + 141 features per unconnected
keyword pair + impact labels, following Gu & Krenn (2024), Appendix C/D, Tables I-II.

Knowledge graph
  vertices : the 3000 keywords (output/top3000_keywords.csv)
  edges    : two keywords co-occur in the title/abstract of a paper of the AI x Quantum
             corpus (30,053 OpenAlex papers); an edge exists "until year t" if such a paper
             was published <= t.
  citations: per paper yearly citations (OpenAlex counts_by_year, 2012-2026).
             Cumulative citations of a paper up to year t = cited_by_count - sum(c_{t'>t}).

For a base year y (features use y, y-1, y-2) and every keyword pair (u, v), u < v, that is
UNCONNECTED in year y:
  0-19    node network features   (degree, new neighbours, their ranks, PageRank)
  20-77   node citation features  (yearly / total / 3-year citations, #papers, averages,
                                   new citations & papers and their ranks)
  78-98   pair network features   (shared neighbours, geometric, cosine, Simpson,
                                   preferential attachment, Sorensen-Dice, Jaccard)
  99-140  pair citation features  (ratios, sums, min / max of node citation quantities)
Label (paper: "3-year citation count of the new pair"): impact = total citations, up to the
end of y+3, of all papers published in (y, y+3] that mention both u and v.
  ir_ge_10 = impact >= 10, ir_ge_50 = impact >= 50, ir_ge_100 = impact >= 100

Outputs per base year (output/pairs_<y>/):
  features_full/part-*.parquet  every unconnected pair, 141 features + impact + 3 labels
  balanced_ir10.csv             paper's training recipe: all IR>=10 positives + equal number
                                of random negatives
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd
import scipy.sparse as sp
import networkx as nx
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
CIT_YEARS = list(range(2003, 2027))

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-+']*[a-z0-9+]|[a-z0-9]")


def singular(w):
    if len(w) <= 3 or not w.isalpha() or w.endswith(("ss", "us", "is", "ics", "sis", "ous", "ws")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    return w[:-1] if w.endswith("s") else w


def canon_tokens(toks):
    toks = list(toks)
    toks[-1] = singular(toks[-1])
    return " ".join(toks)


def tokenize(s):
    s = re.sub(r"\$[^$]*\$|<[^>]+>", " ", s or "").lower()
    return TOKEN_RE.findall(s)


# ----------------------------------------------------------------------------- incidence
def build_incidence(texts, keywords):
    """paper x keyword binary matrix: keyword occurs as contiguous token sequence
    (last word singular-folded, so 'neural networks' == 'neural network')."""
    keys = [canon_tokens(tokenize(k)) for k in keywords]
    idx = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, i)
    max_n = max(len(k.split()) for k in keys)
    rows, cols = [], []
    for r, t in enumerate(texts):
        toks = tokenize(t)
        found = set()
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                j = idx.get(canon_tokens(toks[i:i + n]))
                if j is not None:
                    found.add(j)
        rows += [r] * len(found)
        cols += list(found)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), len(keywords)))


def rank(x):
    """rank among all keywords, 1 = largest (ties share the best rank)."""
    return pd.Series(x).rank(method="min", ascending=False).to_numpy(np.float32)


def safe_div(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


# ----------------------------------------------------------------------------- per-year state
class YearState:
    """Everything needed from 'the graph until year t'."""

    def __init__(self, M, pyear, cit, cited_total, t):
        K = M.shape[1]
        mask = pyear <= t
        Mt = M[mask]
        co = (Mt.T @ Mt).tocsr()
        co.setdiag(0); co.eliminate_zeros()
        A = (co > 0).astype(np.float32)
        self.A = A
        self.N = np.asarray(A.sum(1)).ravel()                                  # degree
        G = nx.from_scipy_sparse_array(A)
        pr = nx.pagerank(G) if G.number_of_edges() else {}
        self.PR = np.array([pr.get(i, 1.0 / K) for i in range(K)], np.float32)
        ci = lambda yy: cit[:, CIT_YEARS.index(yy)] if yy in CIT_YEARS else np.zeros(len(pyear))
        c_t = ci(t) * mask
        c_3 = (ci(t) + ci(t - 1) + ci(t - 2)) * mask
        later = sum(ci(yy) for yy in CIT_YEARS if yy > t)
        cum_t = np.clip(cited_total - later, 0, None) * mask
        self.C = M.T @ c_t                                                      # yearly citations in t
        self.Ct = M.T @ cum_t                                                   # total citations up to t
        self.Ct3 = M.T @ c_3                                                    # citations in t-2..t
        self.Pn = np.asarray(M[mask].sum(0)).ravel()                            # papers mentioning
        self.Cm, self.Ctm, self.Ctm3 = safe_div(self.C, self.Pn), safe_div(self.Ct, self.Pn), safe_div(self.Ct3, self.Pn)
        self.Ns = (A @ A).toarray().astype(np.float32)                          # shared neighbours


def node_feature_table(S):
    """S = {0: state(y), 1: state(y-1), 2: state(y-2), 3: state(y-3), 4: state(y-4)};
    returns list of (name, per-node array) in the paper's order, u/v interleaved later."""
    y0, y1, y2 = S[0], S[1], S[2]
    T = []
    # 0-5 degree
    for k, s in enumerate([y0, y1, y2]):
        T.append((f"N_y-{k}", s.N))
    # 6-9 new neighbours, 10-13 ranks
    T.append(("newN_1y", y0.N - y1.N)); T.append(("newN_2y", y0.N - y2.N))
    T.append(("rank_newN_1y", rank(y0.N - y1.N))); T.append(("rank_newN_2y", rank(y0.N - y2.N)))
    # 14-19 PageRank
    for k, s in enumerate([y0, y1, y2]):
        T.append((f"PR_y-{k}", s.PR))
    # ---- node citation features 20-77
    for nm, attr in [("C", "C"), ("Ct", "Ct"), ("Ct3", "Ct3"), ("Pn", "Pn"), ("Cm", "Cm"), ("Ctm", "Ctm"), ("Ctm3", "Ctm3")]:
        for k, s in enumerate([y0, y1, y2]):
            T.append((f"{nm}_y-{k}", getattr(s, attr)))
    T.append(("newCt_1y", y0.Ct - y1.Ct)); T.append(("newCt_2y", y0.Ct - y2.Ct))
    T.append(("rank_newCt_1y", rank(y0.Ct - y1.Ct))); T.append(("rank_newCt_2y", rank(y0.Ct - y2.Ct)))
    T.append(("newPn_1y", y0.Pn - y1.Pn)); T.append(("newPn_2y", y0.Pn - y2.Pn))
    T.append(("rank_newPn_1y", rank(y0.Pn - y1.Pn))); T.append(("rank_newPn_2y", rank(y0.Pn - y2.Pn)))
    return T


def feature_names():
    names = []
    # node features are emitted per entry as (u, v); the paper interleaves u,v per year
    dummy = {"N": None}
    return names


def pair_features(u, v, S, NODE):
    """Compute the 141 features for arrays of pairs (u, v). Returns (names, matrix)."""
    cols, names = [], []

    def add(name, arr):
        names.append(name); cols.append(np.asarray(arr, np.float32))

    y = [S[0], S[1], S[2]]
    # ---- 0-19 node network features (paper order: u_y, v_y, u_y-1, v_y-1, ...)
    for k in range(3):
        add(f"f_N_u_y-{k}", y[k].N[u]); add(f"f_N_v_y-{k}", y[k].N[v])
    for nm in ["newN_1y", "newN_2y", "rank_newN_1y", "rank_newN_2y"]:
        add(f"f_{nm}_u", NODE[nm][u]); add(f"f_{nm}_v", NODE[nm][v])
    for k in range(3):
        add(f"f_PR_u_y-{k}", y[k].PR[u]); add(f"f_PR_v_y-{k}", y[k].PR[v])
    # ---- 20-77 node citation features
    for attr in ["C", "Ct", "Ct3", "Pn", "Cm", "Ctm", "Ctm3"]:
        for k in range(3):
            a = getattr(y[k], attr)
            add(f"f_{attr}_u_y-{k}", a[u]); add(f"f_{attr}_v_y-{k}", a[v])
    for nm in ["newCt_1y", "newCt_2y", "rank_newCt_1y", "rank_newCt_2y",
               "newPn_1y", "newPn_2y", "rank_newPn_1y", "rank_newPn_2y"]:
        add(f"f_{nm}_u", NODE[nm][u]); add(f"f_{nm}_v", NODE[nm][v])
    # ---- 78-98 pair network features
    Ns = [y[k].Ns[u, v] for k in range(3)]
    Nu = [y[k].N[u].astype(np.float64) for k in range(3)]
    Nv = [y[k].N[v].astype(np.float64) for k in range(3)]
    for k in range(3): add(f"f_Ns_y-{k}", Ns[k])
    geo = [safe_div(Ns[k] ** 2, Nu[k] * Nv[k]) for k in range(3)]
    for k in range(3): add(f"f_Geo_y-{k}", geo[k])
    for k in range(3): add(f"f_Cos_y-{k}", np.sqrt(geo[k]))
    for k in range(3): add(f"f_Sim_y-{k}", safe_div(Ns[k], np.minimum(Nu[k], Nv[k])))
    for k in range(3): add(f"f_Pre_y-{k}", Nu[k] * Nv[k])
    for k in range(3): add(f"f_Sor_y-{k}", safe_div(2 * Ns[k], Nu[k] + Nv[k]))
    for k in range(3): add(f"f_Jac_y-{k}", safe_div(Ns[k], Nu[k] + Nv[k] - Ns[k]))
    # ---- 99-140 pair citation features
    g = lambda attr, k, n: getattr(y[k], attr)[n].astype(np.float64)
    for k in range(3): add(f"f_r1_y-{k}", safe_div(g("Ct", k, u) + g("Ct", k, v), g("Pn", k, u) + g("Pn", k, v)))
    for k in range(3): add(f"f_r2_y-{k}", safe_div(g("Ct", k, u) * g("Ct", k, v), g("Pn", k, u) + g("Pn", k, v)))
    for attr, nm in [("Cm", "s"), ("Ctm", "st"), ("Ct3", "st3"), ("Ctm3", "stm3")]:
        for k in range(3): add(f"f_{nm}_y-{k}", g(attr, k, u) + g(attr, k, v))
    for attr in ["C", "Ct", "Ct3"]:
        for fn, nm in [(np.minimum, "min"), (np.maximum, "max")]:
            for k in range(3): add(f"f_{nm}{attr}_y-{k}", fn(g(attr, k, u), g(attr, k, v)))
    for fn, nm in [(np.minimum, "min"), (np.maximum, "max")]:
        for k in range(3): add(f"f_{nm}Pn_y-{k}", fn(g("Pn", k, u), g("Pn", k, v)))
    assert len(cols) == 141, len(cols)
    return names, np.column_stack(cols)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", default=os.path.join(OUT, "top3000_keywords.csv"))
    ap.add_argument("--years", default="2019,2022", help="base years y (label window y+1..y+3)")
    ap.add_argument("--chunk", type=int, default=60, help="u-nodes per parquet part")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    kw = pd.read_csv(args.keywords)["keyword"].astype(str).tolist()
    pap = pd.read_parquet(os.path.join(OUT, "_work", "papers_citations.parquet"))
    pyear = pap["year"].fillna(0).astype(int).to_numpy()
    cit = pap[[f"c{y}" for y in CIT_YEARS]].to_numpy(np.float64)
    cited_total = pap["cited_total"].to_numpy(np.float64)
    print(f"{len(kw)} keywords, {len(pap)} papers")

    M = build_incidence(pap["text"].tolist(), kw)
    sp.save_npz(os.path.join(OUT, "_work", "paper_keyword_incidence.npz"), M)
    dfreq = np.asarray(M.sum(0)).ravel()
    print(f"incidence built: {int(M.sum())} mentions; keywords present in corpus: {(dfreq > 0).sum()}/{len(kw)}"
          f"; papers with >=2 keywords: {(np.asarray(M.sum(1)).ravel() >= 2).sum()}  [{time.time()-t0:.0f}s]")

    summary = {}
    rng = np.random.default_rng(args.seed)
    for y in [int(s) for s in args.years.split(",")]:
        print(f"\n=== base year y={y}: features from {y-2}..{y}, label from papers in {y+1}..{y+3} ===")
        S = {k: YearState(M, pyear, cit, cited_total, y - k) for k in range(3)}
        NODE = dict(node_feature_table(S))
        # label: citations up to y+3 of papers published in (y, y+3] that mention both
        fmask = (pyear > y) & (pyear <= y + 3)
        later = sum(cit[:, CIT_YEARS.index(yy)] for yy in CIT_YEARS if yy > y + 3)
        cum = np.clip(cited_total - later, 0, None) * fmask
        Mf = M[fmask]
        W = (Mf.T @ sp.diags(cum[fmask]) @ Mf).toarray().astype(np.float32)
        Wcnt = (Mf.T @ Mf).toarray()
        A0 = S[0].A.toarray() > 0
        K = len(kw)
        iu, ju = np.triu_indices(K, 1)
        unconn = ~A0[iu, ju]
        imp_all = W[iu, ju][unconn]
        n_pairs = int(unconn.sum())
        pos10 = int((imp_all >= 10).sum())
        neg_p = pos10 / max(1, n_pairs - pos10)
        print(f"pairs total {len(iu)}, unconnected in {y}: {n_pairs}; newly connected by {y+3}: "
              f"{int((Wcnt[iu, ju][unconn] > 0).sum())}; IR>=10: {pos10}, IR>=50: {int((imp_all>=50).sum())}, "
              f"IR>=100: {int((imp_all>=100).sum())}")

        odir = os.path.join(OUT, f"pairs_{y}"); fdir = os.path.join(odir, "features_full")
        os.makedirs(fdir, exist_ok=True)
        for f in os.listdir(fdir): os.remove(os.path.join(fdir, f))
        bal_path = os.path.join(odir, "balanced_ir10.csv")
        if os.path.exists(bal_path): os.remove(bal_path)
        part = 0; names = None
        for start in range(0, K, args.chunk):
            sel = (iu >= start) & (iu < start + args.chunk)
            sel &= ~A0[iu, ju]
            u, v = iu[sel], ju[sel]
            if len(u) == 0: continue
            names, X = pair_features(u, v, S, NODE)
            impact = W[u, v]
            df = pd.DataFrame(X, columns=names)
            df.insert(0, "keyword_v", np.array(kw, dtype=object)[v]); df.insert(0, "keyword_u", np.array(kw, dtype=object)[u])
            df.insert(0, "v", v.astype(np.int32)); df.insert(0, "u", u.astype(np.int32))
            df["impact_3y_citations"] = impact
            df["ir_ge_10"] = (impact >= 10).astype(np.int8)
            df["ir_ge_50"] = (impact >= 50).astype(np.int8)
            df["ir_ge_100"] = (impact >= 100).astype(np.int8)
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                           os.path.join(fdir, f"part-{part:04d}.parquet"), compression="zstd")
            part += 1
            keep = (df["ir_ge_10"] == 1) | (rng.random(len(df)) < neg_p)
            df[keep].to_csv(bal_path, mode="a", header=not os.path.exists(bal_path), index=False)
            del df, X
            print(f"  part {part}: u in [{start},{start+args.chunk}) {len(u)} pairs  [{time.time()-t0:.0f}s]", flush=True)
        summary[y] = {"unconnected_pairs": n_pairs, "ir_ge_10": pos10, "ir_ge_50": int((imp_all >= 50).sum()),
                      "ir_ge_100": int((imp_all >= 100).sum()), "parts": part,
                      "feature_window": f"{y-2}-{y}", "label_window": f"{y+1}-{y+3}"}
        del S, NODE, W, Wcnt, A0

    # feature dictionary (index -> name -> paper group)
    groups = ["node_network"] * 20 + ["node_citation"] * 58 + ["pair_network"] * 21 + ["pair_citation"] * 42
    pd.DataFrame({"index": range(141), "feature": names, "group": groups}).to_csv(
        os.path.join(OUT, "feature_dictionary.csv"), index=False)
    json.dump(summary, open(os.path.join(OUT, "pairs_summary.json"), "w"), indent=2)
    print("\nsummary:", json.dumps(summary, indent=1), f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
