import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
R=pd.read_csv("res/plus_results.csv"); S=json.load(open("res/plus_summary.json"))
SP=["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]; XL=["Pair","Gene","Family","Pathway"]; x=np.arange(4)
COL=dict(nstyle.MOD); COL["Ensemble"]="#1F1F1F"
fig,axs=plt.subplots(1,4,figsize=(7.1,2.15)); fig.subplots_adjust(left=0.065,right=0.985,bottom=0.25,top=0.86,wspace=0.6)
def xt(ax): ax.set_xticks(x); ax.set_xticklabels(XL,rotation=35,ha="right",rotation_mode="anchor"); ax.set_xlim(-0.4,3.4)
def ttl(ax,t,l): ax.set_title(t,fontsize=7,loc="left",fontweight="bold"); nstyle.letter(ax,l,x=-0.36,y=1.06)
# a AUPRC
ax=axs[0]
for m in ["Lookup","LR-Graph","GBDT-Graph","LR-Lex","Ensemble"]:
    v=[S[s][m+"_AUPRC"] for s in SP]; ax.plot(x,v,marker="o",color=COL[m],ms=2.6,lw=1.1 if m in("LR-Lex","Ensemble") else 0.8,ls="--" if m=="Ensemble" else "-",label=m,zorder=3 if m=="Ensemble" else 2)
ax.plot(x,[S[s]["prev_AUPRC"] for s in SP],color="#999999",lw=0.7,ls=(0,(2,2)))
ax.text(0,S[SP[0]]["prev_AUPRC"]-0.045,"label prevalence",fontsize=5.3,color="#777777")
ax.set_ylim(0.3,1.02); ax.set_ylabel("Macro AUPRC"); xt(ax); ttl(ax,"Precision–recall","a")
ax.legend(loc="lower left",fontsize=5.2,frameon=False,handlelength=1.6,borderaxespad=0.1,bbox_to_anchor=(0,0.15))
# b conformal
ax=axs[1]
for i,s in enumerate(SP):
    d=R[(R.split==s)].dropna(subset=["LR-Lex_AUROC"]).conf_cov
    ax.scatter(np.full(len(d),i)+np.random.default_rng(i).uniform(-0.12,0.12,len(d)),d,s=6,color="#D1495B" if i else "#5F6B76",alpha=0.45,lw=0)
    ax.errorbar(i,S[s]["conf_cov"],yerr=d.std(),fmt="o",ms=4,color="#D1495B" if i else "#5F6B76",mec="white",mew=0.5,elinewidth=0.8,capsize=0)
ax.axhline(0.9,color="#1F1F1F",lw=0.7,ls=(0,(3,1.5))); ax.text(3.35,0.915,"90% target",fontsize=5.3,ha="right")
ax.set_ylim(0.2,1.0); ax.set_ylabel("Empirical coverage"); xt(ax); ttl(ax,"Conformal guarantee","b")
# c precision@10 lift
ax=axs[2]; labs=["activation","inhibition","phosphorylation"]; w=0.24
for j,l in enumerate(labs):
    v=[S[s]["P10_"+l]/S[s]["prev_"+l] for s in SP]
    ax.bar(x+(j-1)*w,v,width=w*0.92,color=nstyle.REL[l],label=l.capitalize())
ax.axhline(1,color="#999999",lw=0.7,ls=(0,(2,2))); ax.set_ylabel("Precision@10 / prevalence"); ax.set_ylim(0,5.0); xt(ax); ttl(ax,"Ranking for curation","c")
ax.legend(loc="upper left",fontsize=5.2,frameon=False,handlelength=1,borderaxespad=0.1)
# d permutation control
ax=axs[3]
a=[S[s]["LR-Lex_AUROC"] for s in SP]; b=[S[s]["Perm_AUROC"] for s in SP]
for i in x: ax.plot([i,i],[b[i],a[i]],color="#BBBBBB",lw=1.6,zorder=1,solid_capstyle="round")
ax.scatter(x,a,s=22,color=nstyle.MOD["LR-Lex"],zorder=3,ec="white",lw=0.5,label="True labels")
ax.scatter(x,b,s=22,color="#9A9A9A",zorder=3,ec="white",lw=0.5,label="Permuted labels")
ax.axhline(0.5,color="#999999",lw=0.6,ls=(0,(2,2)))
ax.set_ylim(0.4,1.04); ax.set_ylabel("LR-Lex macro AUROC"); xt(ax); ttl(ax,"Negative control","d")
ax.legend(loc="upper right",fontsize=5.2,frameon=False,handletextpad=0.1,borderaxespad=0.1)
for e in ("pdf","png"): fig.savefig(f"fig/fig_plus.{e}",dpi=400)
