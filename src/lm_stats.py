# Summaries for the four frozen biomedical language models (test-size-weighted for pathway folds)
import json, pandas as pd, numpy as np
from scipy.stats import wilcoxon
FILES={"BioBERT":"res/lm_frozen.csv","BioBERT v1.0":"res/lm_frozen_biobert10.csv","SciBERT":"res/lm_frozen_scibert.csv","SPECTER":"res/lm_frozen_specter.csv"}
SP=["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]; L=["activation","inhibition","phosphorylation"]; S={}
def wm(d,c):
    d=d.dropna(subset=[c]); 
    if d.split.iloc[0]=="pathway-held-out": return float(np.average(d[c],weights=d.ntest))
    return float(d[c].mean())
def boot(d,c,n=5000):
    d=d.dropna(subset=[c]); rs=np.random.default_rng(0); v=[]
    for _ in range(n):
        i=rs.integers(0,len(d),len(d)); v.append(np.average(d[c].values[i],weights=d.ntest.values[i]))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
for m,f in FILES.items():
    D=pd.read_csv(f); S[m]={}
    for s in SP:
        a=D[D.split==s]; o={k:wm(a,c) for k,c in [("AUROC","BioBERT-Frozen_AUROC"),("AUPRC","BioBERT-Frozen_AUPRC"),("Scr","BioBERT-Frozen-Scr_AUROC"),("LR-Lex","LR-Lex_AUROC")]+[(l,"Frozen_"+l) for l in L]}
        x=a.dropna(subset=["LR-Lex_AUROC","BioBERT-Frozen_AUROC","BioBERT-Frozen-Scr_AUROC"])
        o["p_vs_lex"]=float(wilcoxon(x["BioBERT-Frozen_AUROC"],x["LR-Lex_AUROC"]).pvalue)
        o["p_vs_scr"]=float(wilcoxon(x["BioBERT-Frozen_AUROC"],x["BioBERT-Frozen-Scr_AUROC"]).pvalue)
        if s=="pathway-held-out": o["CI"]=boot(a,"BioBERT-Frozen_AUROC")
        S[m][s]=o
json.dump(S,open("res/lm_summary.json","w"),indent=1)
for m in S: print(m, pd.DataFrame(S[m]).drop("CI",errors="ignore").round(3).to_string(),"\n")
# fine-tuned BioBERT (if complete) and LR-Lex per-label pathway AUROC from the main results
import os
R=pd.read_csv("res/pair_results.csv",dtype={"pathway":str}); r=R[(R.split=="pathway-held-out")&(R.model=="LR-Lex")]
S["LexLabels"]={l:float(np.average(r.dropna(subset=["AUC_"+l])["AUC_"+l],weights=r.dropna(subset=["AUC_"+l]).ntest)) for l in L}
if os.path.exists("res/lm_ft.csv"):
    T=pd.read_csv("res/lm_ft.csv"); Z=pd.read_csv("res/lm_frozen.csv")
    if set(SP)<=set(T.split):
        S["BioBERT-FT"]={}
        for s in SP:
            a=T[T.split==s]; o={"AUROC":wm(a,"BioBERT-FT_AUROC"),"AUPRC":wm(a,"BioBERT-FT_AUPRC"),"nfold":len(a)}
            for l in L: o[l]=wm(a,"FT_"+l)
            m=a.merge(Z[Z.split==s][["fold","LR-Lex_AUROC"]],on="fold").dropna(subset=["BioBERT-FT_AUROC","LR-Lex_AUROC"])
            o["LR-Lex"]=wm(m.assign(split=s),"LR-Lex_AUROC"); o["p_vs_lex"]=float(wilcoxon(m["BioBERT-FT_AUROC"],m["LR-Lex_AUROC"]).pvalue)
            if s=="pathway-held-out": o["CI"]=boot(a,"BioBERT-FT_AUROC")
            S["BioBERT-FT"][s]=o
json.dump(S,open("res/lm_summary.json","w"),indent=1)
print(json.dumps({k:S[k] for k in ("LexLabels","BioBERT-FT") if k in S},indent=0)[:3000])
