import sys,json; sys.path.insert(0,".")
import nstyle; nstyle.apply()
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
ACT,INH,PHO=nstyle.REL["activation"],nstyle.REL["inhibition"],nstyle.REL["phosphorylation"]
INK="#1F1F1F"; SUB="#666666"; LG="#C9C9C9"; RED=nstyle.ACC; BLUE=nstyle.MOD["LR-Lex"]; CARD="#F6F7F9"; EDGE="#D5D9DF"
NUM=json.load(open("res/fig1_numbers.json"))
W,H=180,98
fig=plt.figure(figsize=(7.1,7.1*H/W)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.set_aspect("equal"); ax.axis("off")
def T(x,y,s,**k): k.setdefault("fontsize",6); k.setdefault("color",INK); return ax.text(x,y,s,**k)
def arrow(p,q,c=INK,lw=0.7,hl=1.3,hw=0.6,sA=0,sB=0,ls="-",bar=False,z=5):
    p=np.array(p,float); q=np.array(q,float); u=(q-p)/np.linalg.norm(q-p); n=np.array([-u[1],u[0]]); a=p+u*sA; b=q-u*sB
    if bar:
        ax.add_line(Line2D(*zip(a,b),color=c,lw=lw,solid_capstyle="butt",zorder=z)); ax.add_line(Line2D(*zip(b-n*hw*1.4,b+n*hw*1.4),color=c,lw=lw*1.6,zorder=z)); return
    e=b-u*hl*0.85; ax.add_line(Line2D(*zip(a,e),color=c,lw=lw,ls=ls,dash_capstyle="butt",solid_capstyle="butt",zorder=z))
    ax.add_patch(Polygon([b,b-u*hl+n*hw,b-u*hl-n*hw],fc=c,ec="none",zorder=z))
def rbox(x,y,w,h,fc="white",ec=EDGE,lw=0.6,r=1.2,z=1): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=z))
CW,CH=57,45; X=[1.5,61.5,121.5]; Y=[51.5,2]
cards=[("a","KEGG pathway data","Curated human signaling maps"),("b","Hidden shortcuts","What random splits do not show"),("c","Leakage-aware splits","Four levels of held-out knowledge"),
       ("d","AI and language models","Seven models, symbol and graph inputs"),("e","Stress tests","Probing what the models rely on"),("f","Key finding","Names transfer; signaling sign does not")]
for i,(l,t,s) in enumerate(cards):
    x0=X[i%3]; y0=Y[i//3]
    T(x0+0.5,y0+CH-4.0,l,fontsize=9,fontweight="bold",va="top"); T(x0+5.2,y0+CH-4.35,t,fontsize=6.8,va="top")

# ---- flow connectors (a->b->c, c->d, d->e->f)
FC="#8C96A3"
for yc in (Y[0]+21,Y[1]+20):
    for xa,xb in ((55.8,62.6),(118.0,122.6)):
        arrow((xa,yc),(xb,yc),c=FC,lw=0.9,hl=1.5,hw=0.75,z=8)
ax.add_line(Line2D([150,150,30,30],[Y[0]+2.2,49.6,49.6,47.2],color=FC,lw=0.9,zorder=8,solid_joinstyle="round",solid_capstyle="round"))
ax.add_patch(Polygon([(30,45.6),(29.25,47.3),(30.75,47.3)],fc=FC,ec="none",zorder=8))
# ================= a: data
x0,y0=X[0],Y[0]
for k in range(3):
    rbox(x0+3+k*1.5,y0+5+k*1.5,25,26,fc="white" if k==2 else "#F2F4F7",ec="#C9CED6",lw=0.45,r=1.0,z=1+k)
ox,oy=x0+7.5,y0+11
P={"A":(2,15),"B":(2,4),"C":(9,15),"D":(9,4),"E":(16,15),"F":(16,4),"G":(19.5,9.5)}
for s,t,k in [("A","C","a"),("B","D","a"),("A","D","a"),("C","E","p"),("D","F","p"),("D","E","a"),("B","C","i"),("E","G","i"),("F","G","a")]:
    p=np.array(P[s])+(ox,oy); q=np.array(P[t])+(ox,oy); c={"a":ACT,"p":PHO,"i":INH}[k]
    arrow(p,q,c=c,lw=0.65,hl=1.0,hw=0.5,sA=1.4,sB=1.5,ls=(0,(2,1)) if k=="p" else "-",bar=(k=="i"))
for v in P.values(): ax.add_patch(Circle(np.array(v)+(ox,oy),1.1,fc="white",ec="#5F6B76",lw=0.55,zorder=6))
for i,(c,lab,ls,bar) in enumerate([(ACT,"activation","-",False),(INH,"inhibition","-",True),(PHO,"phosphorylation",(0,(2,1)),False)]):
    yy=y0+33.5-i*4.3 if False else None
sx=x0+36
for i,(v,lab) in enumerate([("23","pathway maps"),("3,207","typed relations"),("860","genes"),("2,176","unique gene pairs")]):
    yy=y0+31.5-i*7.2; T(sx,yy,v,fontsize=7.6,fontweight="bold",va="center"); T(sx,yy-3,lab,fontsize=5.2,color=SUB,va="center")
lx=x0+4
for i,(c,lab,ls,bar) in enumerate([(ACT,"act.","-",False),(INH,"inh.","-",True),(PHO,"phos.",(0,(2,1)),False)]):
    xx=lx+i*10.5; arrow((xx,y0+2.6),(xx+4,y0+2.6),c=c,lw=0.7,hl=1.0,hw=0.5,ls=ls,bar=bar); T(xx+4.8,y0+2.6,lab,fontsize=5,color=c,va="center")
# ================= b: shortcuts
x0,y0=X[1],Y[0]; bx=x0+3.5
def row(yy,big,lab,lab2):
    T(bx+17,yy+1.6,big,fontsize=7.6,fontweight="bold",va="center"); T(bx+17,yy-1.6,lab,fontsize=5.2,color=SUB,va="center")
    if lab2: T(bx+17,yy-4.2,lab2,fontsize=5.2,color=SUB,va="center")
# 1 co-annotation
yy=y0+32.5
for xx in (bx+1.5,bx+12): ax.add_patch(Circle((xx,yy),1.4,fc="white",ec="#5F6B76",lw=0.55,zorder=6))
arrow((bx+1.5,yy+0.65),(bx+12,yy+0.65),c=ACT,lw=0.7,hl=1,hw=0.5,sA=1.6,sB=1.6); arrow((bx+1.5,yy-0.65),(bx+12,yy-0.65),c=PHO,lw=0.7,hl=1,hw=0.5,sA=1.6,sB=1.6,ls=(0,(2,1)))
row(yy,"19.6%","of pairs carry two labels",None)
# 2 overlap
yy=y0+24.3
for j in range(10):
    ax.add_patch(Rectangle((bx+j*1.4,yy-1.4),1.15,2.8,fc=RED if j<4 else "#DADDE1",ec="none"))
row(yy,"41%","of random-split test rows",None); T(bx+17,yy-4.2,"reuse a training pair",fontsize=5.2,color=SUB,va="center")
# 3 families
yy=y0+14.2
T(bx,yy+1.5,"MAPK",fontsize=5.6,fontweight="bold",color=PHO,family="monospace",va="center"); T(bx+6.4,yy+1.5,"1",fontsize=5.6,color="#9A9A9A",family="monospace",va="center")
T(bx,yy-1.7,"MAPK",fontsize=5.6,fontweight="bold",color=PHO,family="monospace",va="center"); T(bx+6.4,yy-1.7,"3",fontsize=5.6,color="#9A9A9A",family="monospace",va="center")
row(yy,"432","gene-name families",None); T(bx+17,yy-4.2,"share informative stems",fontsize=5.2,color=SUB,va="center")
# 4 ceiling
yy=y0+4.6
ax.add_patch(Rectangle((bx,yy-1.4),14,2.8,fc="#DADDE1",ec="none")); ax.add_patch(Rectangle((bx,yy-1.4),14*0.842,2.8,fc="#5F6B76",ec="none"))
ax.add_line(Line2D([bx+14*0.842]*2,[yy-2.3,yy+2.3],color=RED,lw=0.8))
row(yy,"0.842","ceiling on single-label accuracy",None)
# ================= c: splits ladder
x0,y0=X[2],Y[0]; cx=x0+4
rows=[("Pair-disjoint","unseen pairs, seen genes",1,"10 folds"),("Gene-disjoint","unseen genes",2,"10 folds"),("Family-disjoint","unseen name families",3,"10 folds"),("Pathway held out","unseen pathways",4,"23 folds")]
for i,(nm,ds,lev,fd) in enumerate(rows):
    yy=y0+32-i*8.6
    for j in range(4):
        ax.add_patch(FancyBboxPatch((cx+4+j*2.3,yy-1.0),1.8,2.0,boxstyle="round,pad=0,rounding_size=0.35",fc=RED if j<lev else "#DADDE1",ec="none",zorder=3))
    T(cx+15,yy+1.3,nm,fontsize=6.2,fontweight="bold",va="center"); T(cx+15,yy-1.7,ds,fontsize=5.2,color=SUB,va="center")
    T(x0+CW-2.5,yy+1.3,fd,fontsize=5,color=SUB,va="center",ha="right")
arrow((cx+1,y0+34),(cx+1,y0+4.5),c=RED,lw=0.9,hl=1.5,hw=0.75)
T(cx-0.6,y0+19,"stricter",fontsize=5.2,color=RED,rotation=90,va="center",ha="center")
# ================= d: predictors
x0,y0=X[0],Y[1]
ix=x0+3.5; iy=y0+19.5
for k,ch in enumerate("MAPK1"):
    rbox(ix+k*2.6,iy+3.2,2.3,3.0,fc="white",ec=BLUE,lw=0.55,r=0.4,z=3); T(ix+k*2.6+1.15,iy+4.7,ch,fontsize=5,family="monospace",ha="center",va="center",color=BLUE,zorder=4)
for k,ch in enumerate("ELK1"):
    rbox(ix+k*2.6,iy-6.2,2.3,3.0,fc="white",ec=BLUE,lw=0.55,r=0.4,z=3); T(ix+k*2.6+1.15,iy-4.7,ch,fontsize=5,family="monospace",ha="center",va="center",color=BLUE,zorder=4)
arrow((ix+5.5,iy+2.7),(ix+5.5,iy-2.7),c="#5F6B76",lw=0.6,hl=0.9,hw=0.45)
T(ix+6.5,iy+0.1,"?",fontsize=7,fontweight="bold",color=RED,va="center")
T(ix,iy+8.3,"gene pair",fontsize=5.2,color=SUB)
mx=x0+20.5
rbox(mx,y0+25.3,17.5,8.6,fc="#F2F4F7",ec="none",lw=0,r=1.0,z=2)
T(mx+8.75,y0+32.3,"Symbol models",fontsize=5.4,fontweight="bold",ha="center",va="center",zorder=4)
for k,(nm,c) in enumerate([("LR-Lex",nstyle.MOD["LR-Lex"]),("CharCNN",nstyle.MOD["CharCNN"])]):
    ax.add_patch(Circle((mx+2.4,y0+29.4-k*2.5),0.7,fc=c,ec="none",zorder=4)); T(mx+4,y0+29.4-k*2.5,nm,fontsize=5.0,va="center",zorder=4)
rbox(mx,y0+15.2,17.5,8.9,fc="#F2F4F7",ec="none",lw=0,r=1.0,z=2)
T(mx+8.75,y0+22.5,"Language models",fontsize=5.4,fontweight="bold",ha="center",va="center",zorder=4)
for k,(nm,c) in enumerate([("BioBERT v1.1, v1.0",nstyle.MOD["BioBERT"]),("SciBERT, SPECTER",nstyle.MOD["SciBERT"])]):
    ax.add_patch(Circle((mx+2.4,y0+19.6-k*2.5),0.7,fc=c,ec="none",zorder=4)); T(mx+4,y0+19.6-k*2.5,nm,fontsize=5.0,va="center",zorder=4)
rbox(mx,y0+1.2,17.5,12.8,fc="#F2F4F7",ec="none",lw=0,r=1.0,z=2)
T(mx+8.75,y0+12.4,"Graph models",fontsize=5.4,fontweight="bold",ha="center",va="center",zorder=4)
for k,(nm,c) in enumerate([("Lookup",nstyle.MOD["Lookup"]),("LR-Graph",nstyle.MOD["LR-Graph"]),("GBDT-Graph",nstyle.MOD["GBDT-Graph"]),("GraphSAGE GNN",nstyle.MOD["GNN"])]):
    ax.add_patch(Circle((mx+2.4,y0+9.6-k*2.4),0.7,fc=c,ec="none",zorder=4)); T(mx+4,y0+9.6-k*2.4,nm,fontsize=5.0,va="center",zorder=4)
for yy in (y0+29.6,y0+19.6,y0+7.6): arrow((ix+13.5,iy),(mx-0.3,yy),c="#9AA3AE",lw=0.55,hl=0.9,hw=0.45)
ox=x0+44
for yy in (y0+29.6,y0+19.6,y0+7.6): arrow((mx+17.8,yy),(ox-0.5,y0+19),c="#9AA3AE",lw=0.55,hl=0.9,hw=0.45)
for k,(lab,c,v) in enumerate([("act.",ACT,0.78),("inh.",INH,0.12),("phos.",PHO,0.91)]):
    yy=y0+25-k*5.5; ax.add_patch(Rectangle((ox,yy-1),8*v,2,fc=c,ec="none")); ax.add_patch(Rectangle((ox,yy-1),8,2,fc="none",ec="#BFC5CC",lw=0.4))
    T(ox,yy+2.4,lab,fontsize=5,color=c,va="center")
T(ox+4,y0+6.5,"multi-label\nprobabilities",fontsize=5,color=SUB,ha="center",va="center",linespacing=1.1)
# ================= e: stress tests (real values, four mini charts)
EX=json.load(open("res/extra_corr.json"))["summary"]; PL=json.load(open("res/plus_summary.json"))
SPL=["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]; SL=["P","G","F","Pw"]
x0,y0=X[1],Y[1]; tw,th=26.0,17.6
tiles=[(x0+2,y0+19.0),(x0+30,y0+19.0),(x0+2,y0+0.4),(x0+30,y0+0.4)]
for (tx,ty) in tiles: rbox(tx,ty,tw,th,fc="#F2F4F7",ec="none",lw=0,r=1.0,z=1)
def frame(tx,ty,title,lo,hi,ticks,fmt):
    T(tx+1.4,ty+th-2.1,title,fontsize=5.5,fontweight="bold",va="center",zorder=3)
    px,py,pw,ph=tx+5.2,ty+3.4,tw-7.0,th-9.6
    ax.add_line(Line2D([px,px],[py,py+ph],color="#5F6B76",lw=0.4,zorder=3))
    ax.add_line(Line2D([px,px+pw],[py,py],color="#5F6B76",lw=0.4,zorder=3))
    for t in ticks:
        yy=py+(t-lo)/(hi-lo)*ph; ax.add_line(Line2D([px-0.5,px],[yy,yy],color="#5F6B76",lw=0.4,zorder=3)); T(px-0.8,yy,fmt(t),fontsize=3.9,ha="right",va="center",color=SUB,zorder=3)
    xs=[px+pw*(k+0.5)/4 for k in range(4)]
    for xx,l in zip(xs,SL): T(xx,py-1.4,l,fontsize=4.1,ha="center",va="center",color=SUB,zorder=3)
    return px,py,pw,ph,xs,(lambda v:py+(v-lo)/(hi-lo)*ph)
def pairbars(tx,ty,title,key2,c2,lab2):
    px,py,pw,ph,xs,Yv=frame(tx,ty,title,0.5,1.0,[0.5,1.0],lambda t:f"{t:g}")
    bw=pw/4*0.32
    for k,s in enumerate(SPL):
        for off,v,c in ((-bw*0.55,EX[s]["LR-Lex_AUROC"],nstyle.MOD["LR-Lex"]),(bw*0.55,EX[s][key2],c2)):
            ax.add_patch(Rectangle((xs[k]+off-bw/2,py),bw,Yv(v)-py,fc=c,ec="none",zorder=4))
    lx=tx+tw-1.2
    for k,(lab,c) in enumerate((("original",nstyle.MOD["LR-Lex"]),(lab2,c2))):
        yy=ty+th-2.1-k*2.2 if False else ty+th-2.1
    T(tx+tw-1.2,ty+th-4.9,lab2,fontsize=4.1,color=c2,ha="right",va="center",zorder=3); T(tx+tw-11.0,ty+th-4.9,"original",fontsize=4.1,color=nstyle.MOD["LR-Lex"],ha="right",va="center",zorder=3)
pairbars(*tiles[0],"Scrambled names","LR-Lex-scr_AUROC",nstyle.ACC,"scrambled")
pairbars(*tiles[1],"Swapped roles","LR-Lex_swapAUROC",nstyle.MOD["GBDT-Graph"],"swapped")
# calibration: ECE lines
tx,ty=tiles[2]; px,py,pw,ph,xs,Yv=frame(tx,ty,"Calibration (ECE)",0,0.4,[0,0.2,0.4],lambda t:f"{t:g}")
for m,c,lab in (("LR-Lex",nstyle.MOD["LR-Lex"],"LR-Lex"),("GBDT-Graph",nstyle.MOD["GBDT-Graph"],"GBDT")):
    ys=[Yv(EX[s][m+"_ECE"]) for s in SPL]; ax.add_line(Line2D(xs,ys,color=c,lw=0.9,zorder=4))
    for xx,yy in zip(xs,ys): ax.add_patch(Circle((xx,yy),0.5,fc=c,ec="white",lw=0.3,zorder=5))
T(tx+tw-1.2,ty+th-4.9,"GBDT-Graph",fontsize=4.3,color=nstyle.MOD["GBDT-Graph"],ha="right",va="center",zorder=3)
T(tx+tw-12.6,ty+th-4.9,"LR-Lex",fontsize=4.3,color=nstyle.MOD["LR-Lex"],ha="right",va="center",zorder=3)
# conformal coverage bars
tx,ty=tiles[3]; px,py,pw,ph,xs,Yv=frame(tx,ty,"Conformal coverage",0,1.0,[0,0.5,1.0],lambda t:f"{t:g}")
bw=pw/4*0.68
for k,s in enumerate(SPL):
    v=PL[s]["conf_cov"]; ax.add_patch(Rectangle((xs[k]-bw/2,py),bw,Yv(v)-py,fc=nstyle.MOD["LR-Lex"] if k==0 else RED,ec="none",zorder=4))
    T(xs[k],Yv(v)-0.5,f"{v*100:.0f}",fontsize=4.0,ha="center",va="top",color="white",fontweight="bold",zorder=7)
ax.add_line(Line2D([px,px+pw],[Yv(0.9)]*2,color=INK,lw=0.5,ls=(0,(1.5,1)),zorder=6))
T(tx+tw-1.2,ty+th-4.9,"--- 90% target",fontsize=4.3,color=SUB,ha="right",va="center",zorder=3)
# ================= f: key finding chart
x0,y0=X[2],Y[1]
gx,gy,gw,gh=x0+9,y0+6.5,33,27
def YP(v): return gy+(v-0.5)/0.5*gh
ax.add_patch(Rectangle((gx,YP(0.5)),gw,YP(0.6)-YP(0.5),fc="#F7DCDC",ec="none",zorder=1))
T(gx+1,YP(0.55),"near chance",fontsize=4.6,color=RED,va="center",zorder=2)
ax.add_line(Line2D([gx,gx],[gy,gy+gh],color="#333333",lw=0.5)); ax.add_line(Line2D([gx,gx+gw],[gy,gy],color="#333333",lw=0.5))
for v in (0.5,0.75,1.0):
    ax.add_line(Line2D([gx-0.8,gx],[YP(v)]*2,color="#333333",lw=0.5)); T(gx-1.3,YP(v),f"{v:g}",fontsize=4.8,ha="right",va="center")
T(gx-6.2,gy+gh/2,"AUROC",fontsize=5,rotation=90,va="center",ha="center")
xl=["Pair","Gene","Family","Pathway"]; xp=[gx+3+i*(gw-6)/3 for i in range(4)]
for x,l in zip(xp,xl): T(x,gy-2.2,l,fontsize=4.8,ha="center",va="center")
for lab,c in [("phosphorylation",PHO),("activation",ACT),("inhibition",INH)]:
    v=NUM["perlabel"][lab]; ys=[YP(t) for t in v]
    ax.add_line(Line2D(xp,ys,color=c,lw=1.1,zorder=4))
    for x,y in zip(xp,ys): ax.add_patch(Circle((x,y),0.75,fc=c,ec="white",lw=0.4,zorder=5))
    T(xp[-1]+1.6,ys[-1],{"phosphorylation":"phos.","activation":"act.","inhibition":"inh."}[lab]+f" {v[-1]:.2f}",fontsize=5,color=c,va="center",fontweight="bold")
for ext in ("pdf","png"): fig.savefig(f"fig/fig_workflow.{ext}",dpi=400,bbox_inches="tight",pad_inches=0.03)
