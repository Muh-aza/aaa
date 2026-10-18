import pandas as pd, numpy as np, networkx as nx, json, warnings, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
from scipy.sparse import hstack, csr_matrix
from scipy.sparse.linalg import svds
warnings.filterwarnings("ignore")
L=["activation","inhibition","phosphorylation"]
A=pd.read_csv("all_raw.csv",dtype=str); B=A[A.relation.isin(L)].reset_index(drop=True)
def pairs_of(df):
    g=df.drop_duplicates(["starter","receiver","relation"]).groupby(["starter","receiver"])
    P=g.relation.apply(set).reset_index()
    for l in L: P[l]=P.relation.apply(lambda s:int(l in s))
    return P.drop(columns="relation")
P=pairs_of(B)
pw=B.groupby(["starter","receiver"]).pathway.apply(set).reset_index(); P=P.merge(pw,on=["starter","receiver"])
P["upair"]=P.apply(lambda r:"|".join(sorted([r.starter,r.receiver])),axis=1)
Y=P[L].values

# ---------------- features computed from training pairs only ----------------
_GC={}
def graph_feats(tr, te, alpha=1.0, k=16):
    key=(len(tr),len(te),tr.starter.iloc[0]+tr.receiver.iloc[-1]+te.starter.iloc[0]+te.receiver.iloc[-1])
    if key in _GC: return _GC[key]
    _GC[key]=_graph_feats(tr,te,alpha,k); return _GC[key]
def _graph_feats(tr, te, alpha=1.0, k=16):
    prior=tr[L].mean().values
    out_ct=tr.groupby("starter")[L].agg(["sum","count"]); in_ct=tr.groupby("receiver")[L].agg(["sum","count"])
    def enc(df,ct,col):
        d={g:([(ct.loc[g,(l,"sum")]+alpha*prior[i])/(ct.loc[g,(L[0],"count")]+alpha) for i,l in enumerate(L)]+[np.log1p(ct.loc[g,(L[0],"count")])]) for g in ct.index}
        dflt=list(prior)+[0.0]
        return np.array([d.get(g,dflt) for g in df[col]])
    # any-role label profile of each gene
    G=nx.DiGraph(); G.add_edges_from(zip(tr.starter,tr.receiver))
    pr=nx.pagerank(G) if len(G) else {}
    nodes=sorted(G.nodes()); ix={n:i for i,n in enumerate(nodes)}
    Adj=csr_matrix((np.ones(len(tr)),([ix[a] for a in tr.starter],[ix[b] for b in tr.receiver])),shape=(len(nodes),len(nodes)))
    kk=min(k,len(nodes)-2); U,S,Vt=svds(Adj.astype(float),k=kk); Uo=U*np.sqrt(S); Vi=Vt.T*np.sqrt(S)
    def topo(df):
        F=[]; succ={n:set(G.successors(n)) for n in G}; pred={n:set(G.predecessors(n)) for n in G}
        for a,b in zip(df.starter,df.receiver):
            fa=[G.out_degree(a) if a in G else 0, G.in_degree(a) if a in G else 0, pr.get(a,0)*100]
            fb=[G.out_degree(b) if b in G else 0, G.in_degree(b) if b in G else 0, pr.get(b,0)*100]
            ea=Uo[ix[a]] if a in ix else np.zeros(kk); eb=Vi[ix[b]] if b in ix else np.zeros(kk)
            cn=len(succ[a]&pred[b]) if (a in G and b in G) else 0
            F.append(fa+fb+[cn, float(ea@eb)]+list(ea)+list(eb))
        return np.array(F)
    Xtr=np.hstack([enc(tr,out_ct,"starter"),enc(tr,in_ct,"receiver"),topo(tr)])
    Xte=np.hstack([enc(te,out_ct,"starter"),enc(te,in_ct,"receiver"),topo(te)])
    return Xtr,Xte
def lex_feats(tr,te):
    v1=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False,min_df=1); v2=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False,min_df=1)
    a=v1.fit_transform(tr.starter); b=v2.fit_transform(tr.receiver)
    return hstack([a,b]).tocsr(), hstack([v1.transform(te.starter),v2.transform(te.receiver)]).tocsr()

def fit_predict(name,tr,te):
    ytr=tr[L].values; out=np.zeros((len(te),3))
    if name=="Prior":
        out[:]=ytr.mean(0); return out
    if name=="Lookup":  # memorise starter->label profile; strongest non-learned heuristic
        Xtr,Xte=graph_feats(tr,te); return Xte[:,0:3]*0.5+Xte[:,4:7]*0.5
    if name in ("LR-Lex","LR-Graph","LR-Lex+Graph"):
        mats_tr=[];mats_te=[]
        if "Lex" in name: a,b=lex_feats(tr,te); mats_tr.append(a); mats_te.append(b)
        if "Graph" in name:
            a,b=graph_feats(tr,te); mu,sd=a.mean(0),a.std(0)+1e-6; mats_tr.append(csr_matrix((a-mu)/sd)); mats_te.append(csr_matrix((b-mu)/sd))
        Xtr=hstack(mats_tr).tocsr(); Xte=hstack(mats_te).tocsr()
        for j in range(3):
            if ytr[:,j].min()==ytr[:,j].max(): out[:,j]=ytr[:,j].mean(); continue
            m=LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced"); m.fit(Xtr,ytr[:,j]); out[:,j]=m.predict_proba(Xte)[:,1]
        return out
    if name=="GBDT-Graph":
        Xtr,Xte=graph_feats(tr,te)
        for j in range(3):
            if ytr[:,j].min()==ytr[:,j].max(): out[:,j]=ytr[:,j].mean(); continue
            m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0,class_weight="balanced",random_state=0); m.fit(Xtr,ytr[:,j]); out[:,j]=m.predict_proba(Xte)[:,1]
        return out
MODELS=["Prior","Lookup","LR-Lex","LR-Graph","LR-Lex+Graph","GBDT-Graph"]

def thresholds(prob,prior):  # threshold at training prevalence-matched 0.5 for balanced models; use 0.5
    return (prob>=0.5).astype(int)
def metrics(Yt,Pp,prior):
    Yh=(Pp>=0.5).astype(int)
    # Prior/Lookup are uncalibrated w.r.t. 0.5 -> threshold at training prevalence
    return Yh
def score(Yt,Pp,thr):
    Yh=(Pp>=thr).astype(int); r={}
    r["macroF1"]=f1_score(Yt,Yh,average="macro",zero_division=0); r["microF1"]=f1_score(Yt,Yh,average="micro",zero_division=0)
    r["exact"]=accuracy_score(Yt,Yh)
    au=[]
    for j in range(3):
        au.append(roc_auc_score(Yt[:,j],Pp[:,j]) if 0<Yt[:,j].sum()<len(Yt) else np.nan)
    r["AUROC"]=np.nanmean(au)
    for j,l in enumerate(L): r["F1_"+l]=f1_score(Yt[:,j],Yh[:,j],zero_division=0); r["AUC_"+l]=au[j]
    return r
def thr_for(name,tr):
    return tr[L].values.mean(0) if name in ("Prior","Lookup","GBDT-Graph_x") else np.full(3,0.5)

def run_split(splitname, folds):
    rows=[]; preds={m:[] for m in MODELS}; ys=[]
    for k,(tri,tei) in enumerate(folds):
        tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True)
        if len(te)==0: continue
        ys.append(te[L].values)
        for m in MODELS:
            pp=fit_predict(m,tr,te); thr=thr_for(m,tr); preds[m].append((pp>=thr).astype(int)); preds[m+"_p"]=preds.get(m+"_p",[])+[pp]
            r=score(te[L].values,pp,thr); r.update(model=m,fold=k,split=splitname,ntest=len(te),ntrain=len(tr)); rows.append(r)
    return rows, preds, ys

rng=np.random.default_rng(0); allrows=[]
# S2 pair-disjoint (group by unordered pair), 5 folds x 2 repeats
from sklearn.model_selection import GroupKFold
def gkf(groups,n,seed):
    ug=np.unique(groups); perm=np.random.default_rng(seed).permutation(ug); fold={g:i%n for i,g in enumerate(perm)}
    f=np.array([fold[g] for g in groups]); return [(np.where(f!=i)[0],np.where(f==i)[0]) for i in range(n)]
folds=[]; 
for s in (0,1): folds+=gkf(P.upair.values,5,s)
r,_,_=run_split("pair-disjoint",folds); allrows+=r
# S3 leave-one-pathway-out
pws=sorted(set(B.pathway)); folds=[]
for p in pws:
    te=np.where(P.pathway.apply(lambda s:p in s))[0]; trm=np.ones(len(P),bool); trm[te]=False
    # also drop reverse-direction duplicates of test pairs
    teu=set(P.upair.iloc[te]); trm&=~P.upair.isin(teu).values
    folds.append((np.where(trm)[0],te))
r,_,_=run_split("pathway-held-out",folds); 
for x,p in zip([q for q in r], [pws[i//len(MODELS)] for i in range(len(r))]): x["pathway"]=p
allrows+=r
# S4 gene-disjoint (cold start): genes into 5 folds; test pairs with both genes in test fold? too few -> at least one unseen gene
genes=np.array(sorted(set(P.starter)|set(P.receiver))); folds=[]
for s in (0,1):
    gf={g:i%5 for i,g in enumerate(np.random.default_rng(s).permutation(genes))}
    for i in range(5):
        tg={g for g,f in gf.items() if f==i}
        a=P.starter.isin(tg).values; b=P.receiver.isin(tg).values
        folds.append((np.where(~a&~b)[0], np.where(a|b)[0]))
r,_,_=run_split("gene-disjoint",folds); allrows+=r
R=pd.DataFrame(allrows); R.to_csv("res/pair_results.csv",index=False)
summ=R[R.split!="pathway-held-out"].groupby(["split","model"])[["macroF1","microF1","AUROC","exact","ntest","ntrain"]].agg(["mean","std"]).round(3)
print(summ.to_string())
# pathway-held-out: pooled weighted mean
ph=R[R.split=="pathway-held-out"]
w=ph.groupby("model").apply(lambda d: pd.Series({c:np.average(d[c].fillna(d[c].mean()),weights=d.ntest) for c in ["macroF1","microF1","AUROC","exact"]})).round(3)
print("pathway-held-out (test-size weighted)\n",w.to_string())
print(ph[ph.model=="LR-Lex+Graph"][["pathway","ntest","macroF1","AUROC"]].round(3).to_string())
