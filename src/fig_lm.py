import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Means are reported in Table VII; this figure shows fold-level differences, the scrambled-name control and per-pathway results.
S=json.load(open("res/lm_summary.json")); SP=nstyle.SPLITS
LMS=["BioBERT","BioBERT v1.0","SciBERT","SPECTER"]; ms=LMS+["BioBERT-FT"]
FILES={"BioBERT":"res/lm_frozen.csv","BioBERT v1.0":"res/lm_frozen_biobert10.csv","SciBERT":"res/lm_frozen_scibert.csv","SPECTER":"res/lm_frozen_specter.csv"}
NM={"BioBERT":"BioBERT v1.1 (frozen)","BioBERT v1.0":"BioBERT v1.0 (frozen)","SciBERT":"SciBERT (frozen)","SPECTER":"SPECTER (frozen)","BioBERT-FT":"BioBERT v1.1 (fine-tuned)"}
Z=pd.read_csv(FILES["BioBERT"])[["split","fold","LR-Lex_AUROC","ntest"]]
D={}
for m in LMS:
    d=pd.read_csv(FILES[m]); D[m]=d.assign(delta=d["BioBERT-Frozen_AUROC"]-d["LR-Lex_AUROC"])
T=pd.read_csv("res/lm_ft.csv").merge(Z,on=["split","fold"],suffixes=("","_z")); D["BioBERT-FT"]=T.assign(delta=T["BioBERT-FT_AUROC"]-T["LR-Lex_AUROC"])
fig,axs=plt.subplots(1,2,figsize=(nstyle.W2,2.3),gridspec_kw=dict(width_ratios=[1.75,1])); fig.subplots_adjust(wspace=0.42,top=0.8)
fig.legend([Line2D([],[],color=nstyle.MOD[m],marker="s" if m=="BioBERT-FT" else "o",ms=3.2,lw=0) for m in ms],[NM[m] for m in ms],
           loc="upper center",ncol=5,fontsize=5.8,bbox_to_anchor=(0.5,1.0),handletextpad=0.2,columnspacing=1.2)
# a fold-level difference from LR-Lex
ax=axs[0]; w=0.8/len(ms)
for i,s in enumerate(SP):
    for k,m in enumerate(ms):
        d=D[m][D[m].split==s].delta.dropna().values; xc=i+(k-(len(ms)-1)/2)*w
        ax.scatter(xc+np.random.default_rng(i*7+k).uniform(-w*0.25,w*0.25,len(d)),d,s=4,color=nstyle.MOD[m],alpha=0.5,lw=0,marker="s" if m=="BioBERT-FT" else "o")
        ax.plot([xc-w*0.38,xc+w*0.38],[np.median(d)]*2,color=nstyle.MOD[m],lw=1.4)
for i in range(1,4): ax.axvline(i-0.5,color="#E3E3E3",lw=0.6,zorder=0)
ax.axhline(0,color="#1A1A1A",lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(["Pair","Gene","Family","Pathway"]); ax.set_xlim(-0.5,3.5)
ax.set_ylim(-0.3,0.35); ax.set_ylabel("AUROC minus LR-Lex AUROC"); ax.set_xlabel("Held-out level (folds or pathways; bar, median)")
ax.text(-0.42,0.33,"better than gene-name model",fontsize=5.6,color="#555555",va="top"); ax.text(-0.42,-0.28,"worse",fontsize=5.6,color="#555555",va="bottom")
nstyle.letter(ax,"a",x=-0.12,y=1.03)
# b scrambled names on unseen genes
ax=axs[1]; s="gene-disjoint"; y=np.arange(len(LMS))
for i,m in enumerate(LMS):
    a,b=S[m][s]["AUROC"],S[m][s]["Scr"]; ax.plot([b,a],[i,i],color="#CFCFCF",lw=2.2,solid_capstyle="round",zorder=1)
    ax.scatter(a,i,s=18,color=nstyle.MOD[m],ec="white",lw=0.4,zorder=3); ax.scatter(b,i,s=18,color="white",ec=nstyle.MOD[m],lw=0.9,zorder=3)
    p=S[m][s]["p_vs_scr"]; ax.text((a+b)/2,i-0.22,f"−{a-b:.2f}"+(" **" if p<0.01 else " *" if p<0.05 else ""),fontsize=5.6,ha="center",va="bottom")
ax.set_yticks(y); ax.set_yticklabels(["BioBERT v1.1","BioBERT v1.0","SciBERT","SPECTER"]); ax.set_ylim(len(LMS)-0.5,-0.9); ax.tick_params(axis="y",length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0.65,1.0); ax.set_xticks([0.7,0.8,0.9,1.0]); ax.set_xlabel("Macro AUROC (unseen genes)")
ax.scatter(0.68,-0.7,s=14,color="#4D4D4D"); ax.text(0.695,-0.7,"real names",fontsize=5.6,va="center"); ax.scatter(0.83,-0.7,s=14,color="white",ec="#4D4D4D",lw=0.9); ax.text(0.845,-0.7,"scrambled",fontsize=5.6,va="center")
nstyle.letter(ax,"b",x=-0.55,y=1.03)
nstyle.save(fig,"fig_lm")
