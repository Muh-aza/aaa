import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
L=list(nstyle.REL); C=nstyle.REL
A=pd.read_csv("all_raw.csv",dtype=str); B=A[A.relation.isin(L)]
ct=B.groupby(["pathway","relation"]).size().unstack(fill_value=0)[L]; ct=ct.loc[ct.sum(axis=1).sort_values().index]
lab=[nstyle.pname(p) for p in ct.index]
fig=plt.figure(figsize=(nstyle.W2,3.0)); gs=fig.add_gridspec(2,2,width_ratios=[1.45,1],height_ratios=[1,0.42],wspace=0.42,hspace=0.08)
# a records per pathway
ax=fig.add_subplot(gs[:,0]); left=np.zeros(len(ct))
for l in L: ax.barh(np.arange(len(ct)),ct[l],left=left,color=C[l],label=l.capitalize(),height=0.68); left+=ct[l].values
ax.set_yticks(np.arange(len(ct))); ax.set_yticklabels(lab,fontsize=6); ax.tick_params(axis="y",length=0); ax.spines["left"].set_visible(False)
ax.set_xlabel("Interaction records"); ax.set_ylim(-0.6,len(ct)-0.4); ax.set_xlim(0,310)
ax.legend(loc="lower right",handlelength=0.9,handleheight=0.9)
nstyle.letter(ax,"a",x=-0.42,y=1.0)
# b UpSet-style label combinations
P=B.drop_duplicates(["starter","receiver","relation"]).groupby(["starter","receiver"]).relation.apply(frozenset); vc=P.value_counts()
order=sorted(vc.index,key=lambda s:(len(s),-vc[s])); vals=[vc[s] for s in order]; x=np.arange(len(order))
ax2=fig.add_subplot(gs[0,1]); cols=[C[next(iter(s))] if len(s)==1 else "#4D4D4D" for s in order]
ax2.bar(x,vals,color=cols,width=0.62)
for i,v in enumerate(vals): ax2.text(i,v+18,f"{v:,}",ha="center",fontsize=5.8)
ax2.set_ylabel("Unique gene pairs"); ax2.set_ylim(0,1080); ax2.set_xlim(-0.6,len(order)-0.4); ax2.set_xticks([])
ax2.spines["bottom"].set_visible(False)
ax2.axvspan(2.5,len(order)-0.4,color=nstyle.LIGHT,lw=0,zorder=0); ax2.text(len(order)-0.5,1040,f"multi-label pairs: {sum(vals[3:])}",ha="right",va="top",fontsize=6)
nstyle.letter(ax2,"b",x=-0.3,y=1.0)
ax3=fig.add_subplot(gs[1,1],sharex=ax2)
for j,l in enumerate(L):
    for i,s in enumerate(order):
        ax3.scatter(i,j,s=16,color=C[l] if l in s else "#E2E2E2",zorder=3,lw=0)
for i,s in enumerate(order):
    js=[j for j,l in enumerate(L) if l in s]
    if len(js)>1: ax3.plot([i,i],[min(js),max(js)],color="#4D4D4D",lw=0.9,zorder=2)
ax3.set_yticks(range(3)); ax3.set_yticklabels(["Activation","Inhibition","Phosphorylation"],fontsize=6); ax3.set_ylim(2.6,-0.6)
ax3.tick_params(length=0); [s.set_visible(False) for s in ax3.spines.values()]; ax3.set_xticks([])
nstyle.save(fig,"fig_audit")
