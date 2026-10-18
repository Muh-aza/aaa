import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
R=pd.read_csv("res/plus_results.csv"); S=json.load(open("res/plus_summary.json")); SP=nstyle.SPLITS; x=np.arange(4)
fig,axs=plt.subplots(1,4,figsize=(nstyle.W2,1.95)); fig.subplots_adjust(wspace=0.58)
# a AUPRC
ax=axs[0]
for m in ["Lookup","LR-Graph","GBDT-Graph","LR-Lex","Ensemble"]:
    v=[S[s][m+"_AUPRC"] for s in SP]; hi=m in ("LR-Lex","Ensemble")
    ax.plot(x,v,marker="o",color=nstyle.MOD[m],ms=2.4,lw=1.1 if hi else 0.8,ls=(0,(3,1.5)) if m=="Ensemble" else "-",label=m,zorder=3 if hi else 2,mec="white",mew=0.3)
ax.plot(x,[S[s]["prev_AUPRC"] for s in SP],color="#9A9A9A",lw=0.6,ls=(0,(2,2)))
ax.text(0.05,S[SP[0]]["prev_AUPRC"]-0.05,"label prevalence",fontsize=5.6,color="#8A8A8A")
ax.set_ylim(0.3,1.02); ax.set_ylabel("Macro AUPRC"); nstyle.splitaxis(ax,label=False,short=True); nstyle.title(ax,"Precision–recall"); nstyle.letter(ax,"a",x=-0.38,y=1.04)
ax.legend(loc="lower left",fontsize=5.6,bbox_to_anchor=(0,0.13),handlelength=1.4)
# b conformal coverage
ax=axs[1]
for i,s in enumerate(SP):
    d=R[(R.split==s)].dropna(subset=["LR-Lex_AUROC"]).conf_cov; c=nstyle.MOD["LR-Lex"] if i==0 else nstyle.ACC
    ax.scatter(np.full(len(d),i)+np.random.default_rng(i).uniform(-0.13,0.13,len(d)),d,s=5,color=c,alpha=0.4,lw=0)
    ax.errorbar(i,S[s]["conf_cov"],yerr=d.std(),fmt="o",ms=3.6,color=c,mec="white",mew=0.4,elinewidth=0.8,capsize=0)
ax.axhline(0.9,color="#1A1A1A",lw=0.6,ls=(0,(3,1.5))); ax.text(3.4,0.915,"90% target",fontsize=5.6,ha="right")
ax.set_ylim(0.2,1.0); ax.set_ylabel("Empirical coverage"); nstyle.splitaxis(ax,label=False,short=True); nstyle.title(ax,"Conformal coverage"); nstyle.letter(ax,"b",x=-0.38,y=1.04)
# c precision@10 lift
ax=axs[2]; w=0.25
for j,l in enumerate(nstyle.REL):
    v=[S[s]["P10_"+l]/S[s]["prev_"+l] for s in SP]
    ax.bar(x+(j-1)*w,v,width=w*0.9,color=nstyle.REL[l],label=l.capitalize())
ax.axhline(1,color="#9A9A9A",lw=0.6,ls=(0,(2,2))); ax.set_ylabel("Precision@10 / prevalence"); ax.set_ylim(0,6.0); ax.set_yticks([0,1,2,3,4]); ax.spines["left"].set_bounds(0,4); nstyle.splitaxis(ax,label=False,short=True)
nstyle.title(ax,"Ranking for curation"); nstyle.letter(ax,"c",x=-0.38,y=1.04); ax.legend(loc="upper left",fontsize=5.6,handlelength=0.8,ncol=1,bbox_to_anchor=(0,1.02))
# d permutation control
ax=axs[3]; a=[S[s]["LR-Lex_AUROC"] for s in SP]; b=[S[s]["Perm_AUROC"] for s in SP]
for i in x: ax.plot([i,i],[b[i],a[i]],color="#CFCFCF",lw=2.2,zorder=1,solid_capstyle="round")
ax.scatter(x,a,s=18,color=nstyle.MOD["LR-Lex"],zorder=3,ec="white",lw=0.4,label="True labels")
ax.scatter(x,b,s=18,color=nstyle.GREY,zorder=3,ec="white",lw=0.4,label="Permuted labels")
nstyle.chance(ax,text=False); ax.set_ylim(0.4,1.04); ax.set_ylabel("LR-Lex macro AUROC"); nstyle.splitaxis(ax,label=False,short=True)
nstyle.title(ax,"Negative control"); nstyle.letter(ax,"d",x=-0.38,y=1.04); ax.legend(loc="upper right",fontsize=5.6)
nstyle.save(fig,"fig_plus")
