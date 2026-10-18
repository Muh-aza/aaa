import re,time
src=open("exp.py").read(); src=src[:src.index("rng=np.random.default_rng(0); allrows=[]")]
exec(src)
def fam(s):
    s=re.sub(r'[0-9]+[A-Z]?[0-9]*$','',s); return s if s else "?"
genes=sorted(set(P.starter)|set(P.receiver)); F={g:fam(g) for g in genes}
fams=sorted(set(F.values())); print("genes",len(genes),"families",len(fams))
import collections; print(collections.Counter(F.values()).most_common(12))
folds=[]
for s in (0,1):
    ff={f:i%5 for i,f in enumerate(np.random.default_rng(s).permutation(fams))}
    for i in range(5):
        a=P.starter.map(lambda g:ff[F[g]]==i).values; b=P.receiver.map(lambda g:ff[F[g]]==i).values
        folds.append((np.where(~a&~b)[0],np.where(a|b)[0]))
t=time.time(); r,_,_=run_split("family-disjoint",folds)
R=pd.DataFrame(r); R.to_csv("res/fam_results.csv",index=False)
print(R.groupby("model")[["macroF1","microF1","AUROC","exact","ntest","ntrain"]].agg(["mean","std"]).round(3).to_string()); print(time.time()-t)
