import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
M=json.load(open("res/more.json")); S=pd.read_csv("res/strata.csv"); LC=pd.read_csv("res/learning_curve.csv")
fig=plt.figure(figsize=(7.1,4.3)); gs=fig.add_gridspec(2,3,hspace=0.75,wspace=0.55,height_ratios=[1,1.1])
# a strata
ax=fig.add_subplot(gs[0,0]); mods=["Lookup","LR-Lex","GBDT-Graph"]; col={"Lookup":nstyle.MOD["Lookup"],"LR-Lex":nstyle.MOD["LR-Lex"],"GBDT-Graph":nstyle.MOD["GBDT-Graph"]}
for j,m in enumerate(mods):
    for k,st in enumerate(["C1","C2"]):
        d=S[(S.model==m)&(S.stratum==st)].AUROC; x=k+(j-1)*0.22
        ax.scatter(np.full(len(d),x)+np.random.default_rng(j+k).uniform(-0.05,0.05,len(d)),d,s=5,color=col[m],alpha=0.45,lw=0)
        ax.errorbar(x,d.mean(),yerr=d.std(),fmt="o",ms=3.5,color=col[m],mec="white",mew=0.4,elinewidth=0.7,label=m if k==0 else None)
n1=S[S.stratum=="C1"].n.mean(); n2=S[S.stratum=="C2"].n.mean()
ax.set_xticks([0,1]); ax.set_xticklabels([f"Both seen\n(n≈{n1:.0f})",f"One seen\n(n≈{n2:.0f})"]); ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.9,1.0); ax.set_xlim(-0.5,1.5)
ax.legend(loc="lower left",fontsize=5.5,handletextpad=0.1); nstyle.letter(ax,"a",x=-0.36)
# b learning curve
ax=fig.add_subplot(gs[0,1])
for m,c in [("LR-Lex",nstyle.MOD["LR-Lex"]),("Lookup",nstyle.MOD["Lookup"]),("LR-Graph",nstyle.MOD["LR-Graph"])]:
    g=LC[LC.model==m].groupby("frac").agg(mu=("AUROC","mean"),sd=("AUROC","std"),n=("ntrain","mean"))
    ax.fill_between(g.n,g.mu-g.sd,g.mu+g.sd,color=c,alpha=0.15,lw=0); ax.plot(g.n,g.mu,"-o",color=c,ms=2.5,lw=0.9,label=m)
ax.set_xlabel("Training pairs (gene-disjoint)"); ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.6,1.0); ax.legend(loc="lower right",fontsize=5.5)
nstyle.letter(ax,"b",x=-0.36)
# c cross-pathway agreement
ax=fig.add_subplot(gs[0,2]); cp=M["crosspath"]
labs=[("activation","Activation"),("inhibition","Inhibition"),("phosphorylation","Phosphorylation"),("sign","Sign (act/inh)")]
vals=[cp[k][0] for k,_ in labs]; cols=[nstyle.REL.get(k,"#555555") for k,_ in labs]
ax.barh(range(4),vals,color=cols,height=0.6); ax.set_yticks(range(4)); ax.set_yticklabels([l for _,l in labs]); ax.invert_yaxis()
for i,v in enumerate(vals): ax.text(v+0.01,i,f"{v*100:.0f}%",va="center",fontsize=5.8)
ax.set_xlim(0,1.12); ax.set_xlabel("Pairs with identical label\nin every pathway (n=244)"); ax.tick_params(axis="y",length=0)
nstyle.letter(ax,"c",x=-0.75)
# d n-gram weights
L=["activation","inhibition","phosphorylation"]
for j,l in enumerate(L):
    ax=fig.add_subplot(gs[1,j]); items=M["coef"][l]; seen=set(); rows=[]
    for name,cf in items:
        if cf in seen: continue
        seen.add(cf); role,ng=name.split(":",1); rows.append(("starter" if role=="S" else "receiver", ng.replace(" ","^"), cf))
    rows=rows[:6][::-1]
    ax.barh(range(len(rows)),[r[2] for r in rows],color=nstyle.REL[l],height=0.62)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[1]}  ({r[0][0].upper()})" for r in rows],family="monospace",fontsize=6)
    ax.tick_params(axis="y",length=0); ax.set_xlabel("LR-Lex coefficient"); ax.set_title(l.capitalize(),fontsize=7,color=nstyle.REL[l],fontweight="bold",loc="left")
    if j==0: nstyle.letter(ax,"d",x=-0.62,y=1.12)
fig.text(0.5,0.005,"^ marks the start of a symbol; (S) starter gene, (R) receiver gene",ha="center",fontsize=5.6,color="#555555")
fig.savefig("fig/fig_more.pdf",bbox_inches="tight")
