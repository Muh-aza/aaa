import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
R=pd.read_csv("res/extra_results.csv"); C=json.load(open("res/extra_corr.json")); S=C["summary"]; SP=nstyle.SPLITS
fig,axs=plt.subplots(1,3,figsize=(nstyle.W2*0.78,1.95)); fig.subplots_adjust(wspace=0.55)
def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "n.s."
def dumb(ax,key2,lab2,c2,pkey,title,let,lg):
    x=np.arange(4); a=[S[s]["LR-Lex_AUROC"] for s in SP]; b=[S[s][key2] for s in SP]
    for i in x:
        ax.plot([i,i],[b[i],a[i]],color="#CFCFCF",lw=2.2,zorder=1,solid_capstyle="round")
        ax.text(i,max(a[i],b[i])+0.025,stars(S[SP[i]][pkey]),ha="center",fontsize=6)
    ax.scatter(x,a,s=18,color=nstyle.MOD["LR-Lex"],zorder=3,ec="white",lw=0.4,label="Original")
    ax.scatter(x,b,s=18,color=c2,zorder=3,ec="white",lw=0.4,label=lab2)
    nstyle.chance(ax,text=False); nstyle.splitaxis(ax,label=False,short=True); ax.set_ylim(0.45,1.06)
    ax.set_ylabel("LR-Lex macro AUROC"); nstyle.title(ax,title); ax.legend(**lg); nstyle.letter(ax,let,x=-0.38,y=1.04)
dumb(axs[0],"LR-Lex-scr_AUROC","Scrambled names",nstyle.ACC,"p_scr","Symbol scrambling","a",dict(loc="lower left",bbox_to_anchor=(0,0.06)))
dumb(axs[1],"LR-Lex_swapAUROC","Roles swapped","#E69F00","p_swap","Direction swap","b",dict(loc="lower left",bbox_to_anchor=(0,0.06)))
ax=axs[2]; x=np.arange(4)
for m in ["LR-Lex","GBDT-Graph"]:
    v=[S[s][m+"_ECE"] for s in SP]; ax.plot(x,v,"-o",color=nstyle.MOD[m],ms=2.8,lw=1.0,mec="white",mew=0.3)
    ax.text(*({"LR-Lex":(2.15,0.06),"GBDT-Graph":(1.55,0.30)}[m]),m,color=nstyle.MOD[m],fontsize=5.8,va="center",ha="left" if m=="LR-Lex" else "right")
nstyle.splitaxis(ax,label=False,short=True); ax.set_ylim(0,0.42)
ax.set_ylabel("Expected calibration error"); nstyle.title(ax,"Calibration under shift"); nstyle.letter(ax,"c",x=-0.38,y=1.04)
nstyle.save(fig,"fig_stress")
