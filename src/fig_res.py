import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
R=pd.concat([pd.read_csv("res/pair_results.csv",dtype={"pathway":str}),pd.read_csv("res/fam_results.csv"),pd.read_csv("res/ai_results.csv",dtype={"pathway":str})])
order=nstyle.SPLITS; models=["Lookup","LR-Graph","LR-Lex+Graph","GBDT-Graph","LR-Lex","CharCNN","GNN"]
CM=LinearSegmentedColormap.from_list("nb",["#FFFFFF","#DCE9F5","#9CC3E4","#4A90C8","#155A9C","#08306B"])
def agg(d,c):
    if d.split.iloc[0]=="pathway-held-out":
        v=d[c].values; w=d.ntest.values; m=~np.isnan(v); return np.average(v[m],weights=w[m])
    return d[c].mean()
fig=plt.figure(figsize=(nstyle.W2,2.5)); gs=fig.add_gridspec(1,2,width_ratios=[2.05,0.85],wspace=0.42)
# a fold-level AUROC distributions (means are in Table II; this shows spread across folds and pathways)
ax=fig.add_subplot(gs[0]); nm=len(models); w=0.82/nm
for i,s_ in enumerate(order):
    for k,m in enumerate(models):
        d=R[(R.split==s_)&(R.model==m)].AUROC.dropna().values; xc=i+(k-(nm-1)/2)*w
        ax.scatter(xc+np.random.default_rng(i*10+k).uniform(-w*0.28,w*0.28,len(d)),d,s=3.5,color=nstyle.MOD[m],alpha=0.55,lw=0)
        q1,med,q3=np.percentile(d,[25,50,75]); ax.plot([xc,xc],[q1,q3],color=nstyle.MOD[m],lw=1.6,solid_capstyle="butt")
        ax.plot([xc-w*0.38,xc+w*0.38],[med,med],color="#1A1A1A",lw=0.8)
for i in range(1,4): ax.axvline(i-0.5,color="#E3E3E3",lw=0.6,zorder=0)
nstyle.chance(ax,x=-0.45); ax.set_xticks(range(4)); ax.set_xticklabels(["Pair-disjoint","Gene-disjoint","Family-disjoint","Pathway held out"]); ax.set_xlim(-0.5,3.5)
ax.set_ylim(0.3,1.01); ax.set_ylabel("Macro AUROC per fold or pathway")
ax.legend(handles=[plt.Line2D([],[],color=nstyle.MOD[m],lw=2.2,label=m) for m in models],loc="lower left",ncol=4,fontsize=5.8,handlelength=1.0,columnspacing=1.0)
nstyle.letter(ax,"a",x=-0.07,y=1.02)
def heat(ax,M,rows,cols):
    im=ax.imshow(M,cmap=CM,vmin=0.5,vmax=1.0,aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=5.8,color="white" if M[i,j]>0.8 else "#1A1A1A")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(np.arange(-0.5,len(cols)),minor=True); ax.set_yticks(np.arange(-0.5,len(rows)),minor=True)
    ax.grid(which="minor",color="white",lw=1.2); ax.tick_params(which="minor",length=0); return im
L=["activation","inhibition","phosphorylation"]
ax=fig.add_subplot(gs[1]); M=np.array([[agg(R[(R.split=="pathway-held-out")&(R.model==m)],"AUC_"+l) for l in L] for m in models])
im=heat(ax,M,models,["Act.","Inh.","Phos."]); ax.set_xlabel("Label (pathway held out)"); nstyle.letter(ax,"b",x=-0.95,y=1.02)
cb=fig.colorbar(im,ax=ax,fraction=0.06,pad=0.05,ticks=[0.5,0.75,1.0]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.8,width=0.4,length=2); cb.set_label("AUROC",fontsize=6.5)
nstyle.save(fig,"fig_splits")
# per-pathway dot plot, coloured by the fraction of test-gene occurrences seen in training
from matplotlib.colors import Normalize
ph=R[(R.split=="pathway-held-out")&(R.model=="LR-Lex")].dropna(subset=["AUROC"])
X=pd.read_csv("res/extra_results.csv"); X=X[X.split=="pathway-held-out"][["fold","gene_overlap","ntest"]]
ph=ph.merge(X,on="fold",suffixes=("","_x")); assert (ph.ntest==ph.ntest_x).all()
ph=ph.sort_values("AUROC"); lab=[nstyle.pname(p) for p in ph.pathway]
OV=LinearSegmentedColormap.from_list("ov",["#C6DBEF","#6BAED6","#2171B5","#08306B"]); nrm=Normalize(0,1)
fig,ax=plt.subplots(figsize=(nstyle.W1,3.1)); y=np.arange(len(ph))
for yi in y: ax.axhline(yi,color="#F0F0F0",lw=0.5,zorder=0)
ax.hlines(y,0.5,ph.AUROC,color="#BDBDBD",lw=0.7)
sc=ax.scatter(ph.AUROC,y,s=8+ph.ntest.values*0.3,c=ph.gene_overlap,cmap=OV,norm=nrm,edgecolor="#1A1A1A",linewidth=0.3,zorder=3)
ax.axvline(0.5,ls=(0,(2.5,2)),lw=0.5,color="#9A9A9A")
ax.set_yticks(y); ax.set_yticklabels(lab,fontsize=6); ax.tick_params(axis="y",length=0); ax.spines["left"].set_visible(False)
ax.set_xlabel("Macro AUROC on held-out pathway (LR-Lex)"); ax.set_xlim(0.4,0.9); ax.set_ylim(-0.7,len(ph)-0.3)
for n in (25,100,250): ax.scatter([],[],s=8+n*0.3,color="#BDBDBD",edgecolor="#1A1A1A",linewidth=0.3,label=f"{n}")
ax.legend(loc="lower right",title="Test pairs",title_fontsize=6,labelspacing=0.7,borderpad=0.3)
cb=fig.colorbar(sc,ax=ax,fraction=0.04,pad=0.02,ticks=[0,0.5,1]); cb.outline.set_visible(False); cb.ax.set_yticklabels(["0%","50%","100%"]); cb.ax.tick_params(labelsize=5.8,width=0.4,length=2)
cb.set_label("Test genes seen in training",fontsize=6.2)
nstyle.save(fig,"fig_pathway")
