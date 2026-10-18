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
# scrambled symbol map: random 6-letter token per gene (identity kept, spelling destroyed)
rs=np.random.default_rng(7); letters=np.array(list(string.ascii_uppercase+string.digits))
SC={g:"".join(rs.choice(letters,6)) for g in genes}
def scr(df): d=df.copy(); d["starter"]=d.starter.map(SC); d["receiver"]=d.receiver.map(SC); return d
def swap(df): d=df.copy(); d["starter"],d["receiver"]=df.receiver.values,df.starter.values; return d
def ece(y,p,nb=10):
    b=np.clip((p*nb).astype(int),0,nb-1); e=0
    for k in range(nb):
        m=b==k
        if m.any(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
rows=[]
for sname,folds in SPL.items():
    for k,(tri,tei) in enumerate(folds):
        tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True)
        if len(te)==0: continue
        Yt=te[L].values
        seen=set(tr.starter)|set(tr.receiver); ov=float(np.mean([g in seen for g in pd.concat([te.starter,te.receiver])]))
        famseen={F[g] for g in seen}; fov=float(np.mean([F[g] in famseen for g in pd.concat([te.starter,te.receiver])]))
        r=dict(split=sname,fold=k,ntest=len(te),gene_overlap=ov,family_overlap=fov,pathway=pws[k] if sname=="pathway-held-out" else "")
        for m in ["LR-Lex","GBDT-Graph"]:
            pp=fit_predict(m,tr,te); r[m+"_AUROC"]=mauc(Yt,pp)
            r[m+"_ECE"]=np.mean([ece(Yt[:,j],pp[:,j]) for j in range(3)])
            r[m+"_Brier"]=float(np.mean((pp-Yt)**2))
            ps=fit_predict(m,tr,swap(te)) if m=="LR-Lex" else None
            if ps is not None: r[m+"_swapAUROC"]=mauc(Yt,ps)
        _GC.clear()
        ps=fit_predict("LR-Lex",scr(tr),scr(te)); r["LR-Lex-scr_AUROC"]=mauc(Yt,ps)
        rows.append(r); print(sname,k,{x:round(v,3) for x,v in r.items() if isinstance(v,float)},flush=True)
R=pd.DataFrame(rows); R.to_csv("res/extra_results.csv",index=False)
cols=[c for c in R.columns if c.endswith(("AUROC","ECE","Brier"))]+["gene_overlap","family_overlap"]
print(R.groupby("split")[cols].agg(["mean","std"]).round(3).T.to_string())
ph=R[R.split=="pathway-held-out"].dropna(subset=["LR-Lex_AUROC"])
out={}
for c in ["gene_overlap","family_overlap","ntest"]:
    for m in ["LR-Lex","GBDT-Graph"]:
        rho,p=spearmanr(ph[c],ph[m+"_AUROC"]); out[f"{m}~{c}"]=(float(rho),float(p))
print(out); json.dump(out,open("res/extra_corr.json","w"),indent=1)
