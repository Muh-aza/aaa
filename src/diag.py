import pandas as pd, numpy as np, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
warnings.filterwarnings("ignore")
L=["activation","inhibition","phosphorylation"]
A=pd.read_csv("all_raw.csv",dtype=str); B=A[A.relation.isin(L)].reset_index(drop=True)
out=[]
for seed in range(10):
    idx=np.random.default_rng(seed).permutation(len(B)); n=len(B); tr=B.iloc[idx[:int(.7*n)]]; te=B.iloc[idx[int(.7*n):int(.9*n)]]
    trl=tr.groupby(["starter","receiver"]).relation.apply(set).to_dict()
    v1=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False); v2=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False)
    m=LogisticRegression(max_iter=3000,class_weight="balanced",C=3.0).fit(hstack([v1.fit_transform(tr.starter),v2.fit_transform(tr.receiver)]),tr.relation)
    p=m.predict(hstack([v1.transform(te.starter),v2.transform(te.receiver)]))
    cat=[]
    for a,b,y in zip(te.starter,te.receiver,te.relation):
        s=trl.get((a,b))
        cat.append("unseen" if s is None else ("seen-same" if y in s else "seen-conflict"))
    d=pd.DataFrame(dict(cat=cat,ok=(p==te.relation.values)))
    out.append(d.groupby("cat").ok.agg(["mean","size"]).assign(seed=seed))
O=pd.concat(out).reset_index(); print(O.groupby("cat")[["mean","size"]].mean().round(3))
