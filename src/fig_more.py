import sys,json; sys.path.insert(0,".")
import pandas as pd, numpy as np, nstyle; nstyle.apply(); import matplotlib.pyplot as plt
M=json.load(open("res/more.json")); S=pd.read_csv("res/strata.csv"); LC=pd.read_csv("res/learning_curve.csv")
fig=plt.figure(figsize=(nstyle.W2,4.1)); gs=fig.add_gridspec(2,3,hspace=0.72,wspace=0.62,height_ratios=[1,1.05])
# a strata
ax=fig.add_subplot(gs[0,0]); mods=["Lookup","LR-Lex","GBDT-Graph"]
for j,m in enumerate(mods):
    for k,st in enumerate(["C1","C2"]):
        d=S[(S.model==m)&(S.stratum==st)].AUROC; x=k+(j-1)*0.24
        ax.scatter(np.full(len(d),x)+np.random.default_rng(j+k).uniform(-0.05,0.05,len(d)),d,s=4,color=nstyle.MOD[m],alpha=0.4,lw=0)
        ax.errorbar(x,d.mean(),yerr=d.std(),fmt="o",ms=3.4,color=nstyle.MOD[m],mec="white",mew=0.4,elinewidth=0.7,capsize=0,label=m if k==0 else None)
n1=S[S.stratum=="C1"].n.mean(); n2=S[S.stratum=="C2"].n.mean()
ax.set_xticks([0,1]); ax.set_xticklabels([f"Both seen\n(n ≈ {n1:.0f})",f"One seen\n(n ≈ {n2:.0f})"]); ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.9,1.0); ax.set_xlim(-0.5,1.5)
ax.legend(loc="lower left"); nstyle.letter(ax,"a",x=-0.36)
# b learning curve
ax=fig.add_subplot(gs[0,1])
for m in ["LR-Lex","Lookup","LR-Graph"]:
    c=nstyle.MOD[m]; g=LC[LC.model==m].groupby("frac").agg(mu=("AUROC","mean"),sd=("AUROC","std"),n=("ntrain","mean"))
    ax.fill_between(g.n,g.mu-g.sd,g.mu+g.sd,color=c,alpha=0.15,lw=0); ax.plot(g.n,g.mu,"-o",color=c,ms=2.4,lw=1.0)
    ax.text(g.n.iloc[-1]+40,g.mu.iloc[-1]+{"LR-Lex":0,"Lookup":0.012,"LR-Graph":-0.014}[m],m,color=c,fontsize=5.8,va="center")
ax.set_xlabel("Training pairs (gene-disjoint)"); ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.6,1.0); ax.set_xlim(0,1750)
nstyle.letter(ax,"b",x=-0.36)
# c cross-pathway agreement
ax=fig.add_subplot(gs[0,2]); cp=M["crosspath"]
labs=[("activation","Activation"),("inhibition","Inhibition"),("phosphorylation","Phosphorylation"),("sign","Sign (act./inh.)")]
vals=[cp[k][0] for k,_ in labs]; cols=[nstyle.REL.get(k,"#4D4D4D") for k,_ in labs]
ax.barh(range(4),[1]*4,color=nstyle.LIGHT,height=0.55); ax.barh(range(4),vals,color=cols,height=0.55)
ax.set_yticks(range(4)); ax.set_yticklabels([l for _,l in labs]); ax.invert_yaxis()
for i,v in enumerate(vals): ax.text(v-0.02,i,f"{v*100:.0f}%",va="center",ha="right",fontsize=6,color="white")
ax.set_xlim(0,1); ax.set_xticks([0,0.5,1]); ax.set_xticklabels(["0","50","100"]); ax.set_xlabel("Pairs with the same label in\nevery pathway (%, n = 244)")
ax.tick_params(axis="y",length=0); ax.spines["left"].set_visible(False)
nstyle.letter(ax,"c",x=-0.78)
# d n-gram weights
L=["activation","inhibition","phosphorylation"]
for j,l in enumerate(L):
    ax=fig.add_subplot(gs[1,j]); items=M["coef"][l]; seen=set(); rows=[]
    for name,cf in items:
        if cf in seen: continue
        seen.add(cf); role,ng=name.split(":",1); rows.append(("starter" if role=="S" else "receiver", ng.replace(" ","^"), cf))
    rows=rows[:6][::-1]
    ax.barh(range(len(rows)),[r[2] for r in rows],color=[nstyle.REL[l] if r[0]=="starter" else nstyle.REL[l]+"99" for r in rows],height=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[1]}" for r in rows],fontsize=6.3)
    for i,r in enumerate(rows): ax.text(0.03,i,"S" if r[0]=="starter" else "R",va="center",fontsize=5.6,color="white",fontweight="bold")
    ax.tick_params(axis="y",length=0); ax.spines["left"].set_visible(False); ax.set_xlabel("LR-Lex coefficient")
    ax.set_title(l.capitalize(),loc="left",fontsize=7,color=nstyle.REL[l],pad=4)
    if j==0: nstyle.letter(ax,"d",x=-0.5,y=1.1)
fig.text(0.5,-0.01,"^ marks the start of a symbol; S, starter gene; R, receiver gene (lighter bars)",ha="center",fontsize=5.8,color="#555555")
nstyle.save(fig,"fig_more")
