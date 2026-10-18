# Export the deterministic fold assignments used in the paper (same seeds as exp.py / exp_fam.py).
import re, os, numpy as np, pandas as pd
L=["activation","inhibition","phosphorylation"]
A=pd.read_csv("all_raw.csv",dtype=str); B=A[A.relation.isin(L)]
P=B.drop_duplicates(["starter","receiver","relation"]).groupby(["starter","receiver"]).relation.apply(set).reset_index()
for l in L: P[l]=P.relation.apply(lambda s:int(l in s))
pw=B.groupby(["starter","receiver"]).pathway.apply(lambda s:";".join(sorted(set(s)))).reset_index()
P=P.drop(columns="relation").merge(pw,on=["starter","receiver"]); P["upair"]=P.apply(lambda r:"|".join(sorted([r.starter,r.receiver])),axis=1)
os.makedirs("splits",exist_ok=True)
for s in (0,1):
    ug=np.unique(P.upair.values); perm=np.random.default_rng(s).permutation(ug); f={g:i%5 for i,g in enumerate(perm)}
    P[f"pair_fold_seed{s}"]=P.upair.map(f)
P.to_csv("splits/pairs_multilabel.csv",index=False)
genes=np.array(sorted(set(P.starter)|set(P.receiver)))
fam=lambda g:(re.sub(r'[0-9]+[A-Z]?[0-9]*$','',g) or "?")
G=pd.DataFrame({"gene":genes,"family":[fam(g) for g in genes]}); fams=sorted(set(G.family))
for s in (0,1):
    G[f"gene_fold_seed{s}"]=G.gene.map({g:i%5 for i,g in enumerate(np.random.default_rng(s).permutation(genes))})
    G[f"family_fold_seed{s}"]=G.family.map({x:i%5 for i,x in enumerate(np.random.default_rng(s).permutation(fams))})
G.to_csv("splits/gene_and_family_folds.csv",index=False)
print(len(P),"pairs;",len(G),"genes;",len(fams),"families")
