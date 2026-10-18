import sys; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
L=list(nstyle.REL); C=nstyle.REL
A=pd.read_csv("all_raw.csv",dtype=str); B=A[A.relation.isin(L)]
names={"01522":"Endocrine resistance","01524":"Platinum resistance","04010":"MAPK","04012":"ErbB","04015":"Rap1","04022":"cGMP–PKG","04024":"cAMP","04060":"Cytokine–receptor","04061":"Viral–cytokine","04114":"Oocyte meiosis","04115":"p53","0414":"Ras","04140":"Autophagy","04141":"ER protein processing","04144":"Endocytosis","04150":"mTOR","04152":"AMPK","04210":"Apoptosis","04211":"Longevity","04215":"Apoptosis (multi-sp.)","04261":"Adrenergic (cardiac)","0521":"EGFR-TKI resistance"}
FIX={"0414":"04014","0521":"01521"}
ct=B.groupby(["pathway","relation"]).size().unstack(fill_value=0)[L]; ct=ct.loc[ct.sum(axis=1).sort_values().index]
lab=[nstyle.pname(p) for p in ct.index]
fig=plt.figure(figsize=(7.1,3.1)); gs=fig.add_gridspec(1,2,width_ratios=[1.55,1],wspace=0.55)
ax=fig.add_subplot(gs[0]); left=np.zeros(len(ct))
for l in L: ax.barh(lab,ct[l],left=left,color=C[l],label=l.capitalize(),height=0.72,edgecolor="white",linewidth=0.3); left+=ct[l].values
ax.set_xlabel("Interaction records"); ax.tick_params(axis="y",length=0,labelsize=6); ax.set_ylim(-0.6,len(ct)-0.4)
ax.legend(loc="lower right",handlelength=1,handleheight=0.8); nstyle.letter(ax,"a",x=-0.42,y=1.0)
ax2=fig.add_subplot(gs[1])
P=B.drop_duplicates(["starter","receiver","relation"]).groupby(["starter","receiver"]).relation.apply(frozenset)
vc=P.value_counts()
ab={"activation":"Act","inhibition":"Inh","phosphorylation":"Phos"}
order=sorted(vc.index,key=lambda s:(len(s),-vc[s]))
xl=[" + ".join(ab[x] for x in L if x in s) for s in order]; vals=[vc[s] for s in order]
cols=[C[next(iter(s))] if len(s)==1 else "#555555" for s in order]
ax2.bar(range(len(order)),vals,color=cols,width=0.7)
for i,v in enumerate(vals): ax2.text(i,v+15,f"{v:,}",ha="center",fontsize=6)
ax2.set_xticks(range(len(order))); ax2.set_xticklabels(xl,rotation=45,ha="right")
ax2.set_ylabel("Unique directed gene pairs"); ax2.set_ylim(0,1100)
ax2.axvline(2.5,ls=(0,(2,2)),lw=0.5,color="#999999"); ax2.text(2.6,1050,"multi-label (426)",fontsize=6,color="#555555",va="top")
nstyle.letter(ax2,"b",x=-0.3,y=1.0)
fig.savefig("fig/fig_audit.pdf",bbox_inches="tight")
