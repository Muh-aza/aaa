import re, json
src=open("exp.py").read(); src=src[:src.index("rng=np.random.default_rng(0); allrows=[]")]
exec(src)
from scipy.stats import wilcoxon
out={}
def gkf(groups,n,seed):
    ug=np.unique(groups); perm=np.random.default_rng(seed).permutation(ug); fold={g:i%n for i,g in enumerate(perm)}
    f=np.array([fold[g] for g in groups]); return [(np.where(f!=i)[0],np.where(f==i)[0]) for i in range(n)]
def mauc(Y,Pp):
    a=[roc_auc_score(Y[:,j],Pp[:,j]) for j in range(3) if 0<Y[:,j].sum()<len(Y)]; return float(np.mean(a)) if a else np.nan
# ---- A: Park-Marcotte strata within pair-disjoint
rows=[]
for s in (0,1):
    for tri,tei in gkf(P.upair.values,5,s):
        tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True)
        seen=set(tr.starter)|set(tr.receiver)
        k=te.starter.isin(seen).astype(int)+te.receiver.isin(seen).astype(int)
        for m in ["Lookup","LR-Lex","GBDT-Graph"]:
            pp=fit_predict(m,tr,te)
            for c,lab in [(2,"C1"),(1,"C2"),(0,"C3")]:
                idx=(k==c).values
                if idx.sum()>=10: rows.append(dict(model=m,stratum=lab,n=int(idx.sum()),AUROC=mauc(te[L].values[idx],pp[idx])))
A_=pd.DataFrame(rows); A_.to_csv("res/strata.csv",index=False)
print(A_.groupby(["stratum","model"]).agg(AUROC=("AUROC","mean"),sd=("AUROC","std"),n=("n","mean")).round(3))
# ---- B: learning curve, gene-disjoint
genes=np.array(sorted(set(P.starter)|set(P.receiver))); rows=[]
for s in (0,1):
    gf={g:i%5 for i,g in enumerate(np.random.default_rng(s).permutation(genes))}
    for i in range(5):
        tg={g for g,f in gf.items() if f==i}; a=P.starter.isin(tg).values; b=P.receiver.isin(tg).values
        trall=P.iloc[np.where(~a&~b)[0]].reset_index(drop=True); te=P.iloc[np.where(a|b)[0]].reset_index(drop=True)
        for frac in [0.1,0.2,0.4,0.6,0.8,1.0]:
            tr=trall.sample(frac=frac,random_state=s*10+i).reset_index(drop=True)
            for m in ["Lookup","LR-Lex","LR-Graph"]:
                rows.append(dict(model=m,frac=frac,ntrain=len(tr),AUROC=mauc(te[L].values,fit_predict(m,tr,te))))
B_=pd.DataFrame(rows); B_.to_csv("res/learning_curve.csv",index=False)
print(B_.groupby(["model","frac"]).AUROC.agg(["mean","std"]).round(3).unstack(0))
# ---- C: cross-pathway label agreement
Bm=B.drop_duplicates(["starter","receiver","relation","pathway"])
G=Bm.groupby(["starter","receiver","pathway"]).relation.apply(frozenset).reset_index()
multi=G.groupby(["starter","receiver"]).filter(lambda d:len(d)>=2)
res={}
for lab in L+["sign"]:
    agree=[];
    for _,d in multi.groupby(["starter","receiver"]):
        if lab=="sign": v=[frozenset(x&{"activation","inhibition"}) for x in d.relation]
        else: v=[lab in x for x in d.relation]
        agree.append(len(set(v))==1)
    res[lab]=(float(np.mean(agree)),len(agree))
# chance agreement baseline: shuffle pathway label sets among the same pathway? report raw only
out["crosspath"]=res; print("cross-pathway full agreement:",res)
# also: per multi-pathway pair, does sign flip (act in one, inh-only in another)?
flip=0;n=0
for _,d in multi.groupby(["starter","receiver"]):
    sg=[x&{"activation","inhibition"} for x in d.relation]; sg=[x for x in sg if x]
    if len(sg)>=2: n+=1; flip+= any("activation" in x for x in sg) and any("inhibition" in x for x in sg)
out["signflip"]=(flip,n); print("pairs with opposite signs across pathways:",flip,"of",n)
# ---- D: n-gram coefficients
from sklearn.feature_extraction.text import TfidfVectorizer
v1=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False); v2=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False)
X=hstack([v1.fit_transform(P.starter),v2.fit_transform(P.receiver)]).tocsr()
names=["S:"+t for t in v1.get_feature_names_out()]+["R:"+t for t in v2.get_feature_names_out()]
coef={}
for j,l in enumerate(L):
    m=LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced").fit(X,P[l].values); c=m.coef_[0]
    o=np.argsort(-c)[:10]; coef[l]=[(names[i],round(float(c[i]),3)) for i in o]
    print(l,coef[l])
out["coef"]=coef
# ---- E: significance + bootstrap
R=pd.concat([pd.read_csv("res/pair_results.csv",dtype={"pathway":str}),pd.read_csv("res/fam_results.csv")])
sig={}
for s in ["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]:
    a=R[(R.split==s)&(R.model=="LR-Lex")].sort_values(["fold"]).AUROC.values; b=R[(R.split==s)&(R.model=="GBDT-Graph")].sort_values(["fold"]).AUROC.values
    mk=~(np.isnan(a)|np.isnan(b)); st=wilcoxon(a[mk],b[mk]); sig[s]=dict(diff=float(np.mean(a[mk]-b[mk])),p=float(st.pvalue),n=int(mk.sum()))
print(sig); out["wilcoxon"]=sig
ph=R[(R.split=="pathway-held-out")].dropna(subset=["AUROC"]); boot={}
rng=np.random.default_rng(0)
for m in ["Lookup","LR-Lex","LR-Graph","LR-Lex+Graph","GBDT-Graph"]:
    d=ph[ph.model==m]; v=d.AUROC.values; w=d.ntest.values; bs=[]
    for _ in range(5000):
        i=rng.integers(0,len(v),len(v)); bs.append(np.average(v[i],weights=w[i]))
    boot[m]=(float(np.average(v,weights=w)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
print(boot); out["boot"]=boot
json.dump(out,open("res/more.json","w"),indent=1)
