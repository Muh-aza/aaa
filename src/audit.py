import pandas as pd, numpy as np
A=pd.read_csv("all_raw.csv",dtype=str)
M=["activation","inhibition","phosphorylation"]
print("rows",len(A),"non-main rows",(~A.relation.isin(M)).sum())
B=A[A.relation.isin(M)]
print("main rows",len(B))
# exact duplicate rows within same pathway
print("dup rows within pathway", B.duplicated(["starter","receiver","relation","pathway"]).sum())
print("dup triples overall", B.duplicated(["starter","receiver","relation"]).sum())
P=B.drop_duplicates(["starter","receiver","relation"]).groupby(["starter","receiver"]).relation.apply(lambda s: tuple(sorted(s))).reset_index()
print("unique pairs",len(P)); print(P.relation.value_counts())
npw=B.groupby(["starter","receiver"]).pathway.nunique()
print("pairs in >1 pathway",(npw>1).sum(), "max",npw.max())
# reverse pairs
s=set(zip(P.starter,P.receiver)); print("pairs whose reverse also exists",sum((r,q) in s for q,r in s))
self_=(P.starter==P.receiver).sum(); print("self loops",self_)
# random row-split leakage simulation 70/20/10
rng=np.random.default_rng(42); leak=[]
for seed in range(20):
    idx=np.random.default_rng(seed).permutation(len(B)); tr=B.iloc[idx[:int(.7*len(B))]]; te=B.iloc[idx[int(.7*len(B)):int(.9*len(B))]]
    trp=set(zip(tr.starter,tr.receiver)); trt=set(zip(tr.starter,tr.receiver,tr.relation))
    leak.append(( np.mean([ (a,b) in trp for a,b in zip(te.starter,te.receiver)]), np.mean([ (a,b,c) in trt for a,b,c in zip(te.starter,te.receiver,te.relation)])))
leak=np.array(leak); print("random split: test rows whose pair in train %.3f±%.3f, whose exact triple in train %.3f±%.3f"%(leak[:,0].mean(),leak[:,0].std(),leak[:,1].mean(),leak[:,1].std()))
genes=set(B.starter)|set(B.receiver); print("genes",len(genes))
deg=pd.concat([B.starter,B.receiver]).value_counts(); print(deg.head(10))
print("pathway sizes\n",B.groupby("pathway").size().to_string())
