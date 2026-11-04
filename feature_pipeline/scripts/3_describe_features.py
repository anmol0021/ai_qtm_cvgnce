"""Adds a plain-English description + paper index to output/feature_dictionary.csv."""
import os, re
import pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
fd = pd.read_csv(os.path.join(OUT, "feature_dictionary.csv"))
BASE = {
 "N": "degree = number of distinct keywords it has co-occurred with",
 "PR": "PageRank score in the keyword graph",
 "C": "citations received in that single year by all papers mentioning it",
 "Ct": "total citations received so far (since first paper) by all papers mentioning it",
 "Ct3": "citations received in the 3 years ending that year by papers mentioning it",
 "Pn": "number of papers mentioning it so far",
 "Cm": "average yearly citations per paper = C / Pn",
 "Ctm": "average total citations per paper = Ct / Pn",
 "Ctm3": "average 3-year citations per paper = Ct3 / Pn",
}
DELTA = {
 "newN": "new neighbours gained = N(y) - N(y-{k})",
 "rank_newN": "rank of new neighbours gained over last {k} year(s) among all 3000 keywords (1 = most)",
 "newCt": "new citations gained = Ct(y) - Ct(y-{k})",
 "rank_newCt": "rank of new citations gained over last {k} year(s) among all keywords (1 = most)",
 "newPn": "new papers mentioning it = Pn(y) - Pn(y-{k})",
 "rank_newPn": "rank of new papers over last {k} year(s) among all keywords (1 = most)",
}
PAIR = {
 "Ns": "shared neighbours of u and v (keywords linked to both)",
 "Geo": "geometric similarity = Ns^2 / (Nu * Nv)",
 "Cos": "cosine similarity = Ns / sqrt(Nu * Nv)",
 "Sim": "Simpson similarity = Ns / min(Nu, Nv)",
 "Pre": "preferential attachment = Nu * Nv",
 "Sor": "Sorensen-Dice = 2 Ns / (Nu + Nv)",
 "Jac": "Jaccard = Ns / (Nu + Nv - Ns)",
 "r1": "(Ct_u + Ct_v) / (Pn_u + Pn_v): citations per paper of the two keywords combined",
 "r2": "(Ct_u * Ct_v) / (Pn_u + Pn_v)",
 "s": "Cm_u + Cm_v (sum of average yearly citations)",
 "st": "Ctm_u + Ctm_v (sum of average total citations)",
 "st3": "Ct3_u + Ct3_v (sum of 3-year citations)",
 "stm3": "Ctm3_u + Ctm3_v (sum of average 3-year citations)",
}
for a in ["C", "Ct", "Ct3", "Pn"]:
    PAIR["min" + a] = f"min({a}_u, {a}_v)"; PAIR["max" + a] = f"max({a}_u, {a}_v)"
YR = {"0": "year y", "1": "year y-1", "2": "year y-2"}
desc = []
for n in fd.feature:
    m = re.fullmatch(r"f_(\w+?)_([uv])_y-(\d)", n)
    if m:
        b, node, k = m.groups(); desc.append(f"[{node}] {BASE[b]} | until {YR[k]}"); continue
    m = re.fullmatch(r"f_(\w+?)_(\d)y_([uv])", n)
    if m:
        b, k, node = m.groups(); desc.append(f"[{node}] " + DELTA[b].format(k=k)); continue
    m = re.fullmatch(r"f_(\w+?)_y-(\d)", n)
    b, k = m.groups(); desc.append(f"[pair] {PAIR[b]} | at {YR[k]}")
fd["description"] = desc
fd.to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)
print(fd.to_string(index=False, max_colwidth=95))
