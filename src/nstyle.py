import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
def apply():
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","DejaVu Sans"],
        "font.size":7,"axes.labelsize":7,"axes.titlesize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.2,
        "axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.2,"ytick.major.size":2.2,
        "xtick.major.pad":1.8,"ytick.major.pad":1.8,"axes.labelpad":2.5,
        "xtick.direction":"out","ytick.direction":"out","axes.spines.top":False,"axes.spines.right":False,
        "axes.edgecolor":"#1A1A1A","axes.labelcolor":"#1A1A1A","xtick.color":"#1A1A1A","ytick.color":"#1A1A1A","text.color":"#1A1A1A",
        "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":600,"legend.frameon":False,"legend.handlelength":1.2,
        "legend.handletextpad":0.4,"legend.borderaxespad":0.2,"legend.labelspacing":0.3,"axes.grid":False,
        "lines.linewidth":1.0,"lines.markersize":3.2,"patch.linewidth":0,"axes.titleweight":"normal",
        "axes.titlelocation":"left","axes.titlepad":3})
def letter(ax,s,x=-0.14,y=1.04):
    """Bold lowercase panel letter (Nature style)."""
    ax.text(x,y,s,transform=ax.transAxes,fontsize=8,fontweight="bold",va="bottom",ha="left")
def title(ax,t):
    """Short sentence-case panel heading in plain weight."""
    ax.set_title(t,fontsize=7,loc="left",color="#1A1A1A",pad=4)
# double-column and single-column widths (inches)
W2=7.16; W1=3.45
SPLITS=["pair-disjoint","gene-disjoint","family-disjoint","pathway-held-out"]; SPL=["Pair","Gene","Family","Pathway"]
SPS=["Pair","Gene","Fam.","Path."]
def splitaxis(ax,label=True,short=False):
    import numpy as np
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(SPS if short else SPL); ax.set_xlim(-0.45,3.45)
    if label: ax.set_xlabel("Held-out level")
def chance(ax,y=0.5,text=True,x=-0.4):
    ax.axhline(y,color="#9A9A9A",lw=0.5,ls=(0,(2.5,2)),zorder=0)
    if text: ax.text(x,y+0.008,"chance",fontsize=5.6,color="#8A8A8A",va="bottom")
def save(fig,name):
    for e in ("pdf","png"): fig.savefig(f"fig/{name}.{e}",dpi=600 if e=="png" else None,bbox_inches="tight",pad_inches=0.02)
# Colour-blind-safe palette (Okabe-Ito / Wong, Nat. Methods 2011)
REL={"activation":"#009E73","inhibition":"#D55E00","phosphorylation":"#5D4AA0"}
MOD={"Lookup":"#A3A3A3","LR-Lex":"#0072B2","LR-Graph":"#56B4E9","LR-Lex+Graph":"#7B8FA1","GBDT-Graph":"#E69F00","CharCNN":"#CC79A7","GNN":"#8C2D04",
     "Ensemble":"#1A1A1A","BioBERT":"#B2182B","BioBERT-FT":"#67001F","BioBERT v1.0":"#D6604D","SciBERT":"#F4A582","SPECTER":"#92C5DE"}
GREY="#A3A3A3"; ACC="#B2182B"; LIGHT="#E8E8E8"
# KEGG pathway names (figures show names only, never map identifiers)
NAMES={"01522":"Endocrine resistance","01524":"Platinum drug resistance","04010":"MAPK","04012":"ErbB","04015":"Rap1","04022":"cGMP–PKG","04024":"cAMP",
 "04060":"Cytokine–receptor interaction","04061":"Viral–cytokine interaction","04114":"Oocyte meiosis","04115":"p53","0414":"Ras","04014":"Ras","04140":"Autophagy",
 "04141":"ER protein processing","04144":"Endocytosis","04150":"mTOR","04152":"AMPK","04210":"Apoptosis","04211":"Longevity","04215":"Apoptosis (multiple species)",
 "04227":"Necroptosis","04261":"Adrenergic signaling (cardiac)","0521":"EGFR-TKI resistance","01521":"EGFR-TKI resistance"}
def pname(p): return NAMES.get(str(p),str(p))
