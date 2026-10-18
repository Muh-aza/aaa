import sys, time
src=open("exp.py").read(); src=src[:src.index("rng=np.random.default_rng(0); allrows=[]")]
exec(src)
import torch, torch.nn as nn, torch.nn.functional as Fnn
from sklearn.decomposition import TruncatedSVD
torch.set_num_threads(4)
CH="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"; CI={c:i+1 for i,c in enumerate(CH)}; MAXL=12
def enc_sym(s): v=[CI.get(c,len(CH)+1) for c in s.upper()[:MAXL]]; return v+[0]*(MAXL-len(v))
class CharEnc(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.emb=nn.Embedding(len(CH)+2,24,padding_idx=0)
        s.convs=nn.ModuleList([nn.Conv1d(24,d,k,padding=k//2) for k in (2,3,4)]); s.out=d*3
    def forward(s,x):
        e=s.emb(x).transpose(1,2); return torch.cat([c(e).relu().max(-1).values for c in s.convs],-1)
class CharCNN(nn.Module):  # symbol-only neural pair classifier
    def __init__(s):
        super().__init__(); s.es=CharEnc(); s.er=CharEnc(); d=s.es.out
        s.mlp=nn.Sequential(nn.Linear(2*d,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,3))
    def forward(s,xs,xr): return s.mlp(torch.cat([s.es(xs),s.er(xr)],-1))
class SAGE(nn.Module):  # inductive GraphSAGE encoder + pairwise edge decoder
    def __init__(s,din,h=64):
        super().__init__(); s.l1s=nn.Linear(din,h); s.l1n=nn.Linear(din,h); s.l2s=nn.Linear(h,h); s.l2n=nn.Linear(h,h)
        s.dec=nn.Sequential(nn.Linear(4*h,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,3)); s.dp=nn.Dropout(0.3)
    def forward(s,X,A,src,dst):
        h=(s.l1s(X)+s.l1n(torch.sparse.mm(A,X))).relu(); h=s.dp(h)
        h=s.l2s(h)+s.l2n(torch.sparse.mm(A,h))
        hs,hr=h[src],h[dst]; return s.dec(torch.cat([hs,hr,hs*hr,(hs-hr).abs()],-1))
def train(model,fwd,Y,epochs,lr=3e-3,wd=1e-4,seed=0):
    pw=torch.tensor(((len(Y)-Y.sum(0))/np.maximum(Y.sum(0),1)),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd); Yt=torch.tensor(Y,dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss=Fnn.binary_cross_entropy_with_logits(fwd(),Yt,pos_weight=pw); loss.backward(); opt.step()
    model.eval()
def predict_ai(name,tr,te,seed=0):
    torch.manual_seed(seed); np.random.seed(seed); ytr=tr[L].values.astype(float)
    if name=="CharCNN":
        xs=torch.tensor([enc_sym(g) for g in tr.starter]); xr=torch.tensor([enc_sym(g) for g in tr.receiver])
        m=CharCNN(); train(m,lambda:m(xs,xr),ytr,epochs=250)
        with torch.no_grad(): return torch.sigmoid(m(torch.tensor([enc_sym(g) for g in te.starter]),torch.tensor([enc_sym(g) for g in te.receiver]))).numpy()
    if name=="GNN":
        nodes=sorted(set(tr.starter)|set(tr.receiver)|set(te.starter)|set(te.receiver)); ix={g:i for i,g in enumerate(nodes)}
        trg=sorted(set(tr.starter)|set(tr.receiver))
        v=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),lowercase=False).fit(trg)
        Z=TruncatedSVD(64,random_state=seed).fit(v.transform(trg)); X=Z.transform(v.transform(nodes))
        deg=np.zeros(len(nodes)); 
        for a,b in zip(tr.starter,tr.receiver): deg[ix[a]]+=1; deg[ix[b]]+=1
        X=np.hstack([X/ (np.linalg.norm(X,axis=1,keepdims=True)+1e-9), np.log1p(deg)[:,None]/3]); X=torch.tensor(X,dtype=torch.float32)
        s=[ix[a] for a in tr.starter]; d=[ix[b] for b in tr.receiver]
        r=np.array(s+d); c=np.array(d+s); w=1/np.maximum(np.bincount(r,minlength=len(nodes)),1)[r]
        A=torch.sparse_coo_tensor(torch.tensor(np.vstack([r,c])),torch.tensor(w,dtype=torch.float32),(len(nodes),len(nodes))).coalesce()
        m=SAGE(X.shape[1]); st=torch.tensor(s); dt=torch.tensor(d)
        # edge dropout on supervision edges not needed; train edges are message edges (transductive within train)
        train(m,lambda:m(X,A,st,dt),ytr,epochs=300,lr=5e-3)
        with torch.no_grad(): return torch.sigmoid(m(X,A,torch.tensor([ix[a] for a in te.starter]),torch.tensor([ix[b] for b in te.receiver]))).numpy()
AI=["CharCNN","GNN"]
def run(splitname,folds,tag=None):
    rows=[]
    for k,(tri,tei) in enumerate(folds):
        tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True)
        for m in AI:
            pp=predict_ai(m,tr,te,seed=k); r=score(te[L].values,pp,np.full(3,0.5)); r.update(model=m,fold=k,split=splitname,ntest=len(te),ntrain=len(tr))
            if tag is not None: r["pathway"]=tag[k]
            rows.append(r)
        print(splitname,k,[round(x["AUROC"],3) for x in rows[-2:]],flush=True)
    return rows
def gkf(groups,n,seed):
    ug=np.unique(groups); perm=np.random.default_rng(seed).permutation(ug); fold={g:i%n for i,g in enumerate(perm)}
    f=np.array([fold[g] for g in groups]); return [(np.where(f!=i)[0],np.where(f==i)[0]) for i in range(n)]
allr=[]
folds=[]; [folds.extend(gkf(P.upair.values,5,s)) for s in (0,1)]; allr+=run("pair-disjoint",folds)
genes=np.array(sorted(set(P.starter)|set(P.receiver))); folds=[]
for s in (0,1):
    gf={g:i%5 for i,g in enumerate(np.random.default_rng(s).permutation(genes))}
    for i in range(5):
        tg={g for g,f in gf.items() if f==i}; a=P.starter.isin(tg).values; b=P.receiver.isin(tg).values; folds.append((np.where(~a&~b)[0],np.where(a|b)[0]))
allr+=run("gene-disjoint",folds)
import re
def fam(s): s2=re.sub(r'[0-9]+[A-Z]?[0-9]*$','',s); return s2 if s2 else "?"
F={g:fam(g) for g in genes}; fams=sorted(set(F.values())); folds=[]
for s in (0,1):
    ff={f:i%5 for i,f in enumerate(np.random.default_rng(s).permutation(fams))}
    for i in range(5):
        a=P.starter.map(lambda g:ff[F[g]]==i).values; b=P.receiver.map(lambda g:ff[F[g]]==i).values; folds.append((np.where(~a&~b)[0],np.where(a|b)[0]))
allr+=run("family-disjoint",folds)
pws=sorted(set(B.pathway)); folds=[]
for p in pws:
    te=np.where(P.pathway.apply(lambda s:p in s))[0]; trm=np.ones(len(P),bool); trm[te]=False; trm&=~P.upair.isin(set(P.upair.iloc[te])).values; folds.append((np.where(trm)[0],te))
allr+=run("pathway-held-out",folds,tag=pws)
R=pd.DataFrame(allr); R.to_csv("res/ai_results.csv",index=False)
for s in ["pair-disjoint","gene-disjoint","family-disjoint"]:
    print(s, R[R.split==s].groupby("model")[["AUROC","macroF1"]].agg(["mean","std"]).round(3).to_string())
ph=R[R.split=="pathway-held-out"]
print(ph.groupby("model").apply(lambda d: pd.Series({c:np.average(d[c].dropna(),weights=d.ntest[d[c].notna()]) for c in ["AUROC","macroF1","AUC_activation","AUC_inhibition","AUC_phosphorylation"]})).round(3))
