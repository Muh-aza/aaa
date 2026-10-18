import pandas as pd, numpy as np, warnings, json
from collections import Counter
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
warnings.filterwarnings("ignore")
L=["activation","inhibition","phosphorylation"]
A=pd.read_csv("all_raw.csv",dtype=str); B=A[A.relation.isin(L)].reset_index(drop=True)
# ceiling for single-label prediction: for each (pair) row group, best constant prediction
g=B.groupby(["starter","receiver"]).relation.apply(list)
ceil=sum(Counter(v).most_common(1)[0][1] for v in g)/len(B)
print("single-label accuracy ceiling (pair-deterministic predictor): %.4f"%ceil)
def mk(tr):
    pair={k:Counter(v).most_common(1)[0][0] for k,v in tr.groupby(["starter","receiver"]).relation.apply(list).items()}
    st={k:Counter(v).most_common(1)[0][0] for k,v in tr.groupby("starter").relation.apply(list).items()}
    rc={k:Counter(v).most_common(1)[0][0] for k,v in tr.groupby("receiver").relation.apply(list).items()}
    glob=tr.relation.mode()[0]; return pair,st,rc,glob
res=[]
for seed in range(10):
  for regime in ["random-row","pair-disjoint"]:
    rng=np.random.default_rng(seed)
    if regime=="random-row":
        idx=rng.permutation(len(B)); n=len(B); tr=B.iloc[idx[:int(.7*n)]]; te=B.iloc[idx[int(.7*n):int(.9*n)]]
    else:
        up=B.apply(lambda r:"|".join(sorted([r.starter,r.receiver])),axis=1); ug=up.unique(); rng.shuffle(ug)
        cut1,cut2=int(.7*len(ug)),int(.9*len(ug)); trg=set(ug[:cut1]); teg=set(ug[cut1:cut2])
        tr=B[up.isin(trg)]; te=B[up.isin(teg)]
    pair,st,rc,glob=mk(tr)
    preds={}
    preds["Majority"]=[glob]*len(te)
    preds["Pair-lookup"]=[pair.get((a,b), st.get(a, rc.get(b,glob))) for a,b in zip(te.starter,te.receiver)]
    preds["Starter-lookup"]=[st.get(a, rc.get(b,glob)) for a,b in zip(te.starter,te.receiver)]
    v1=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False); v2=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False)
    Xtr=hstack([v1.fit_transform(tr.starter),v2.fit_transform(tr.receiver)]); Xte=hstack([v1.transform(te.starter),v2.transform(te.receiver)])
    m=LogisticRegression(max_iter=3000,class_weight="balanced",C=3.0).fit(Xtr,tr.relation); preds["LR-Lex"]=m.predict(Xte)
    for k,p in preds.items():
        res.append(dict(seed=seed,regime=regime,model=k,macroF1=f1_score(te.relation,p,average="macro"),MCC=matthews_corrcoef(te.relation,p),acc=accuracy_score(te.relation,p),ntest=len(te)))
R=pd.DataFrame(res); R.to_csv("res/row_results.csv",index=False)
print(R.groupby(["regime","model"])[["macroF1","MCC","acc","ntest"]].agg(["mean","std"]).round(3).to_string())
