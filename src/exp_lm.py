# Language-model methodology: BioBERT v1.1 (PubMed) frozen embeddings and fine-tuning under the four splits
import sys, time
src=open("exp_plus.py").read(); src=src[:src.index("\nrows=[]")]; exec(src)
import torch, torch.nn as nn
from transformers import BertModel, BertTokenizer
torch.set_num_threads(4); torch.manual_seed(0)
MODE=sys.argv[1] if len(sys.argv)>1 else "frozen"
LM=sys.argv[2] if len(sys.argv)>2 else "lm/biobert_pt"; TAG=sys.argv[3] if len(sys.argv)>3 else ""
tok=BertTokenizer.from_pretrained(LM, do_lower_case=("scibert" in LM or "specter" in LM))
rng=np.random.default_rng(7); SCR={g:"".join(rng.choice(list(string.ascii_uppercase),6)) for g in genes}
@torch.no_grad()
def embed(names):
    m=BertModel.from_pretrained(LM, output_hidden_states=True).eval(); E={}
    for i in range(0,len(names),64):
        b=names[i:i+64]; t=tok(b,padding=True,return_tensors="pt"); h=m(**t).hidden_states
        H=torch.stack(h[-4:]).mean(0); msk=t["attention_mask"].clone(); msk[:,0]=0
        msk[torch.arange(len(b)),t["attention_mask"].sum(1)-1]=0
        e=(H*msk[...,None]).sum(1)/msk.sum(1,keepdim=True)
        for n,v in zip(b,e.numpy()): E[n]=v
    return E
def pairX(df,E,mp=None):
    mp=mp or (lambda g:g); s=np.stack([E[mp(g)] for g in df.starter]); r=np.stack([E[mp(g)] for g in df.receiver])
    return np.hstack([s,r,s*r])
def lr_emb(tr,te,E,mp=None):
    from sklearn.preprocessing import StandardScaler
    Xtr,Xte=pairX(tr,E,mp),pairX(te,E,mp); sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    ytr=tr[L].values; out=np.zeros((len(te),3))
    for j in range(3):
        if ytr[:,j].min()==ytr[:,j].max(): out[:,j]=ytr[:,j].mean(); continue
        m=LogisticRegression(C=0.05,max_iter=3000,class_weight="balanced"); m.fit(Xtr,ytr[:,j]); out[:,j]=m.predict_proba(Xte)[:,1]
    return out
class Clf(nn.Module):
    def __init__(s): super().__init__(); s.b=BertModel.from_pretrained(LM); s.d=nn.Dropout(0.1); s.o=nn.Linear(768,3)
    def forward(s,**t): return s.o(s.d(s.b(**t).last_hidden_state[:,0]))
def finetune(tr,te,epochs=4,bs=32,lr=3e-5,seed=0):
    torch.manual_seed(seed); m=Clf(); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    y=torch.tensor(tr[L].values,dtype=torch.float32); pos=y.sum(0).clamp(min=1); pw=((len(y)-pos)/pos).clamp(max=20)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw); S,R=list(tr.starter),list(tr.receiver)
    steps=epochs*int(np.ceil(len(tr)/bs)); sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:max(0.0,1-i/steps)); g=torch.Generator().manual_seed(seed)
    m.train()
    for ep in range(epochs):
        for idx in torch.randperm(len(tr),generator=g).split(bs):
            t=tok([S[i] for i in idx],[R[i] for i in idx],padding=True,return_tensors="pt")
            loss=lossf(m(**t),y[idx]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(te),128):
            t=tok(list(te.starter[i:i+128]),list(te.receiver[i:i+128]),padding=True,return_tensors="pt"); out.append(torch.sigmoid(m(**t)).numpy())
    return np.vstack(out)
rows=[]
if MODE=="frozen":
    E=embed(list(genes)); E.update(embed([SCR[g] for g in genes]))
    for sname,folds in SPL.items():
        for k,(tri,tei) in enumerate(folds):
            tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True); Yt=te[L].values
            r=dict(split=sname,fold=k,ntest=len(te))
            r["LR-Lex_AUROC"]=mauc(Yt,fit_predict("LR-Lex",tr,te))
            p=lr_emb(tr,te,E); r["BioBERT-Frozen_AUROC"]=mauc(Yt,p); r["BioBERT-Frozen_AUPRC"]=mauprc(Yt,p)
            r["BioBERT-Frozen-Scr_AUROC"]=mauc(Yt,lr_emb(tr,te,E,lambda g:SCR[g]))
            for j,l in enumerate(L):
                if 0<Yt[:,j].sum()<len(Yt): r["Frozen_"+l]=roc_auc_score(Yt[:,j],p[:,j])
            _GC.clear(); rows.append(r); print(sname,k,{x:round(v,3) for x,v in r.items() if isinstance(v,float)},flush=True)
    out=f"res/lm_frozen{TAG}.csv"
else:
    sub=lambda s,f: f[:5] if s!="pathway-held-out" else f
    import os
    fn=f"res/lm_ft{TAG}.csv"; done=set()
    if os.path.exists(fn):
        prev=pd.read_csv(fn); rows=prev.to_dict("records"); done={(r["split"],int(r["fold"])) for r in rows}
    for sname,folds in SPL.items():
        for k,(tri,tei) in enumerate(sub(sname,folds)):
            if (sname,k) in done: continue
            tr=P.iloc[tri].reset_index(drop=True); te=P.iloc[tei].reset_index(drop=True); Yt=te[L].values
            t0=time.time(); p=finetune(tr,te); r=dict(split=sname,fold=k,ntest=len(te))
            r["BioBERT-FT_AUROC"]=mauc(Yt,p); r["BioBERT-FT_AUPRC"]=mauprc(Yt,p)
            for j,l in enumerate(L):
                if 0<Yt[:,j].sum()<len(Yt): r["FT_"+l]=roc_auc_score(Yt[:,j],p[:,j])
            r["sec"]=time.time()-t0; rows.append(r); print(sname,k,{x:round(v,3) for x,v in r.items() if isinstance(v,float)},flush=True)
            pd.DataFrame(rows).to_csv(f"res/lm_ft{TAG}.csv",index=False)
    out=f"res/lm_ft{TAG}.csv"
R=pd.DataFrame(rows); R.to_csv(out,index=False); print(R.groupby("split").mean(numeric_only=True).round(3).T.to_string())
