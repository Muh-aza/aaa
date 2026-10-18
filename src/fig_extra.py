import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
R=pd.read_csv("res/extra_results.csv"); C=json.load(open("res/extra_corr.json")); S=C["summary"]
SP=["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]; XL=["Pair","Gene","Family","Pathway"]
fig,axs=plt.subplots(1,4,figsize=(7.1,2.05)); fig.subplots_adjust(left=0.07,right=0.99,bottom=0.27,top=0.86,wspace=0.55)
def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "n.s."
def dumb(ax,key2,lab2,c2,pkey,title,let,lg=dict(loc="upper right")):
    x=np.arange(4); a=[S[s]["LR-Lex_AUROC"] for s in SP]; b=[S[s][key2] for s in SP]
    for i in x:
        ax.plot([i,i],[b[i],a[i]],color="#BBBBBB",lw=1.6,zorder=1,solid_capstyle="round")
        ax.text(i,max(a[i],b[i])+0.03,stars(S[SP[i]][pkey]),ha="center",fontsize=6)
    ax.scatter(x,a,s=22,color=nstyle.MOD["LR-Lex"],zorder=3,ec="white",lw=0.5,label="Original")
    ax.scatter(x,b,s=22,color=c2,zorder=3,ec="white",lw=0.5,label=lab2)
    ax.axhline(0.5,color="#999999",lw=0.6,ls=(0,(2,2)),zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(XL,rotation=35,ha="right",rotation_mode="anchor"); ax.set_ylim(0.45,1.06); ax.set_xlim(-0.5,3.5)
    ax.set_ylabel("LR-Lex macro AUROC"); ax.set_title(title,fontsize=7,loc="left",fontweight="bold")
    ax.legend(**lg,fontsize=5.5,frameon=False,handletextpad=0.1,borderaxespad=0.2); nstyle.letter(ax,let,x=-0.36,y=1.06)
dumb(axs[0],"LR-Lex-scr_AUROC","Scrambled names","#B5446E","p_scr","Symbol scrambling","a",lg=dict(loc="lower left",bbox_to_anchor=(0.0,0.08)))
dumb(axs[1],"LR-Lex_swapAUROC","Roles swapped","#C4692B","p_swap","Direction swap","b")
ax=axs[2]; x=np.arange(4)
for m in ["LR-Lex","GBDT-Graph"]:
    v=[S[s][m+"_ECE"] for s in SP]; ax.plot(x,v,"-o",color=nstyle.MOD[m],ms=3,lw=1.0)
    ax.text(3.12,v[-1],m,color=nstyle.MOD[m],fontsize=5.8,va="center")
ax.set_xticks(x); ax.set_xticklabels(XL,rotation=35,ha="right",rotation_mode="anchor"); ax.set_xlim(-0.3,4.3); ax.set_ylim(0,0.42)
ax.set_ylabel("Expected calibration error"); ax.set_title("Calibration under shift",fontsize=7,loc="left",fontweight="bold"); nstyle.letter(ax,"c",x=-0.36,y=1.06)
ax=axs[3]; d=R[R.split=="pathway-held-out"].dropna(subset=["LR-Lex_AUROC"])
ax.scatter(d.gene_overlap,d["LR-Lex_AUROC"],s=np.clip(d.ntest/2,6,60),color=nstyle.MOD["LR-Lex"],alpha=0.75,ec="white",lw=0.4)
ax.axhline(0.5,color="#999999",lw=0.6,ls=(0,(2,2)))
rho,p=C["LR-Lex~gene_overlap"]; ax.text(0.04,0.95,f"Spearman ρ = {rho:.2f}\nP = {p:.2f}",transform=ax.transAxes,va="top",fontsize=5.8)
ax.set_xlabel("Test genes seen in training"); ax.set_ylabel("LR-Lex macro AUROC"); ax.set_xlim(0,0.9); ax.set_ylim(0.4,0.9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v*100:.0f}%"))
ax.set_title("Held-out pathways",fontsize=7,loc="left",fontweight="bold"); nstyle.letter(ax,"d",x=-0.36,y=1.06)
for e in ("pdf","png"): fig.savefig(f"fig/fig_extra.{e}",dpi=400)
