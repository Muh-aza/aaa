import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
R=pd.concat([pd.read_csv("res/pair_results.csv",dtype={"pathway":str}),pd.read_csv("res/fam_results.csv"),pd.read_csv("res/ai_results.csv",dtype={"pathway":str})])
FIX={"0414":"04014","0521":"01521"}
order=["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]; xt=["Pair-\ndisjoint","Gene-\ndisjoint","Family-\ndisjoint","Pathway\nheld-out"]
models=list(nstyle.MOD)
def agg(d,c):
    if d.split.iloc[0]=="pathway-held-out":
        v=d[c].values; w=d.ntest.values; m=~np.isnan(v); return np.average(v[m],weights=w[m]),np.nan
    return d[c].mean(), d[c].std()
fig=plt.figure(figsize=(7.1,2.6)); gs=fig.add_gridspec(1,3,width_ratios=[1.3,0.9,1],wspace=0.95)
x=np.arange(4)
ax=fig.add_subplot(gs[0]); MK={"Lookup":"o","LR-Lex":"o","LR-Graph":"s","LR-Lex+Graph":"D","GBDT-Graph":"^","CharCNN":"o","GNN":"s"}
ends=[]
for m in models:
    mu=np.array([agg(R[(R.split==s)&(R.model==m)],"AUROC")[0] for s in order])
    lw=1.4 if m in ("LR-Lex","CharCNN","GNN") else 0.8
    ax.plot(x,mu,color=nstyle.MOD[m],lw=lw,marker=MK[m],ms=3.2,mec="white",mew=0.4,zorder=3 if lw>1 else 2)
    ends.append([mu[-1],m])
ends.sort(); ys=[e[0] for e in ends]
for i in range(1,len(ys)):
    if ys[i]-ys[i-1]<0.021: ys[i]=ys[i-1]+0.021
for (v,m),y in zip(ends,ys): ax.text(3.15,y,m,color=nstyle.MOD[m],fontsize=5.6,va="center")
ax.axhline(0.5,ls=(0,(2,2)),lw=0.5,color="#888888"); ax.text(0,0.508,"chance",fontsize=5.4,color="#888888")
ax.set_xticks(x); ax.set_xticklabels(["Pair","Gene","Family","Pathway"]); ax.set_xlabel("Held-out level"); ax.set_ylabel("Macro AUROC")
ax.set_ylim(0.45,1.0); ax.set_xlim(-0.25,3.9); nstyle.letter(ax,"a",x=-0.24,y=1.03)
def heat(ax,M,rows,cols_,cbl):
    im=ax.imshow(M,cmap="PuBuGn",vmin=0.5,vmax=1.0,aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=5.8,color="white" if M[i,j]>0.8 else "#222222")
    ax.set_xticks(range(len(cols_))); ax.set_xticklabels(cols_); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]; return im
L=["activation","inhibition","phosphorylation"]
ax=fig.add_subplot(gs[1]); mm=[m for m in models]
M=np.array([[agg(R[(R.split=="pathway-held-out")&(R.model==m)],"AUC_"+l)[0] for l in L] for m in mm])
heat(ax,M,mm,["Act.","Inh.","Phos."],""); ax.set_xlabel("Pathway held out"); nstyle.letter(ax,"b",x=-1.0,y=1.03)
ax=fig.add_subplot(gs[2])
M=np.array([[agg(R[(R.split==s)&(R.model=="LR-Lex")],"AUC_"+l)[0] for s in order] for l in L])
im=heat(ax,M,["Act.","Inh.","Phos."],["Pair","Gene","Fam.","Path."],""); ax.set_xlabel("Held-out level (LR-Lex)")
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.04); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5,width=0.4,length=2); cb.set_label("AUROC",fontsize=6)
nstyle.letter(ax,"c",x=-0.42,y=1.03)
fig.savefig("fig/fig_splits.pdf",bbox_inches="tight")
# per-pathway lollipop
ph=R[(R.split=="pathway-held-out")&(R.model=="LR-Lex")].dropna(subset=["AUROC"]).sort_values("AUROC")
names={"01522":"Endocrine resist.","01524":"Platinum resist.","04010":"MAPK","04012":"ErbB","04015":"Rap1","04022":"cGMP–PKG","04024":"cAMP","04114":"Oocyte meiosis","04115":"p53","0414":"Ras","04140":"Autophagy","04141":"ER protein proc.","04144":"Endocytosis","04150":"mTOR","04152":"AMPK","04210":"Apoptosis","04211":"Longevity","04215":"Apoptosis (multi-sp.)","04261":"Adrenergic (cardiac)","0521":"EGFR-TKI resist."}
lab=[nstyle.pname(p) for p in ph.pathway]
fig,ax=plt.subplots(figsize=(3.45,3.0)); y=np.arange(len(ph))
ax.hlines(y,0.5,ph.AUROC,color="#BBBBBB",lw=0.8)
sz=12+ph.ntest.values*0.35
ax.scatter(ph.AUROC,y,s=sz,color=nstyle.MOD["LR-Lex"],edgecolor="white",linewidth=0.4,zorder=3)
ax.axvline(0.5,ls=(0,(2,2)),lw=0.5,color="#777777")
ax.set_yticks(y); ax.set_yticklabels(lab,fontsize=5.8); ax.tick_params(axis="y",length=0)
ax.set_xlabel("Macro AUROC on held-out pathway (LR-Lex)"); ax.set_xlim(0.4,0.9); ax.spines["left"].set_visible(False)
for n in (25,100,250): ax.scatter([],[],s=12+n*0.35,color=nstyle.MOD["LR-Lex"],label=f"{n} pairs")
ax.legend(loc="lower right",title="Test size",title_fontsize=6,labelspacing=0.6,borderpad=0.2)
fig.savefig("fig/fig_pathway.pdf",bbox_inches="tight")
