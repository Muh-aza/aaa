# Extra analyses: (A) symbol scrambling, (B) direction swap, (C) calibration, (D) overlap vs pathway AUROC
import re, json, string
src=open("exp.py").read(); src=src[:src.index("rng=np.random.default_rng(0); allrows=[]")]
exec(src)
from scipy.stats import spearmanr
def mauc(Y,Pp):
    a=[roc_auc_score(Y[:,j],Pp[:,j]) for j in range(3) if 0<Y[:,j].sum()<len(Y)]; return float(np.mean(a)) if a else np.nan
def gkf(groups,n,seed):
    ug=np.unique(groups); perm=np.random.default_rng(seed).permutation(ug); fold={g:i%n for i,g in enumerate(perm)}
    f=np.array([fold[g] for g in groups]); return [(np.where(f!=i)[0],np.where(f==i)[0]) for i in range(n)]
def fam(s):
    s=re.sub(r'[0-9]+[A-Z]?[0-9]*$','',s); return s if s else "?"
genes=np.array(sorted(set(P.starter)|set(P.receiver))); F={g:fam(g) for g in genes}; fams=sorted(set(F.values()))
SPL={}
SPL["pair-disjoint"]=gkf(P.upair.values,5,0)+gkf(P.upair.values,5,1)
f=[]
for s in (0,1):
    gf={g:i%5 for i,g in enumerate(np.random.default_rng(s).permutation(genes))}
    for i in range(5):
        tg={g for g,x in gf.items() if x==i}; a=P.starter.isin(tg).values; b=P.receiver.isin(tg).values
        f.append((np.where(~a&~b)[0],np.where(a|b)[0]))
SPL["gene-disjoint"]=f; f=[]
for s in (0,1):
    ff={x:i%5 for i,x in enumerate(np.random.default_rng(s).permutation(fams))}
    for i in range(5):
        a=P.starter.map(lambda g:ff[F[g]]==i).values; b=P.receiver.map(lambda g:ff[F[g]]==i).values
        f.append((np.where(~a&~b)[0],np.where(a|b)[0]))
SPL["family-disjoint"]=f; f=[]
pws=sorted(set(B.pathway))
for p in pws:
    te=np.where(P.pathway.apply(lambda s:p in s))[0]; trm=np.ones(len(P),bool); trm[te]=False
    trm&=~P.upair.isin(set(P.upair.iloc[te])).values; f.append((np.where(trm)[0],te))
SPL["pathway-held-out"]=f

from sklearn.metrics import average_precision_score
def mauprc(Y,Pp):
    a=[average_precision_score(Y[:,j],Pp[:,j]) for j in range(3) if 0<Y[:,j].sum()<len(Y)]; return float(np.mean(a)) if a else np.nan
def conformal(tr,te,alpha=0.1,seed=0):
    # split conformal per label with LR-Lex; calibration = 20% of training pairs
    rs=np.random.default_rng(seed); idx=rs.permutation(len(tr)); nc=len(tr)//5
    cal=tr.iloc[idx[:nc]].reset_index(drop=True); fit=tr.iloc[idx[nc:]].reset_index(drop=True)
    pc=fit_predict("LR-Lex",fit,cal); pt=fit_predict("LR-Lex",fit,te); cov=[];amb=[]
    for j in range(3):
        yc=cal[L].values[:,j]; s=1-np.where(yc==1,pc[:,j],1-pc[:,j]); n=len(s)
        q=np.quantile(s,min(1,np.ceil((n+1)*(1-alpha))/n),method="higher")
        in1=(1-pt[:,j])<=q; in0=pt[:,j]<=q; yt=te[L].values[:,j]
        cov.append(np.mean(np.where(yt==1,in1,in0))); amb.append(np.mean(in1&in0))
    return float(np.mean(cov)),float(np.mean(amb))
def patk(y,p,k=10):
    k=min(k,len(y)); o=np.argsort(-p)[:k]; return float(y[o].mean()), float(y.mean())
rows=[]
for sname,folds in SPL.items():
    for k,(tri,tei) in enumerate(folds):
        tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True)
        if len(te)==0: continue
        Yt=te[L].values; r=dict(split=sname,fold=k,ntest=len(te)); pr={}
        for m in ["Lookup","LR-Lex","LR-Graph","GBDT-Graph"]:
            pr[m]=fit_predict(m,tr,te)
        pr["Ensemble"]=(pr["LR-Lex"]+pr["GBDT-Graph"]+pr["LR-Graph"])/3
        for m,pp in pr.items():
            r[m+"_AUROC"]=mauc(Yt,pp); r[m+"_AUPRC"]=mauprc(Yt,pp)
        r["prev_AUPRC"]=float(np.mean([Yt[:,j].mean() for j in range(3) if 0<Yt[:,j].sum()<len(Yt)])) if any(0<Yt[:,j].sum()<len(Yt) for j in range(3)) else np.nan
        for j,l in enumerate(L):
            if Yt[:,j].sum()>0:
                a,b=patk(Yt[:,j],pr["LR-Lex"][:,j]); r["P10_"+l]=a; r["prev_"+l]=b
        r["conf_cov"],r["conf_amb"]=conformal(tr,te)
        # label-permutation negative control
        trp=tr.copy(); trp[L]=tr[L].values[np.random.default_rng(k).permutation(len(tr))]
        r["Perm_AUROC"]=mauc(Yt,fit_predict("LR-Lex",trp,te))
        _GC.clear(); rows.append(r); print(sname,k,{x:round(v,3) for x,v in r.items() if isinstance(v,float)},flush=True)
R=pd.DataFrame(rows); R.to_csv("res/plus_results.csv",index=False)
print(R.groupby("split").mean(numeric_only=True).round(3).T.to_string())
