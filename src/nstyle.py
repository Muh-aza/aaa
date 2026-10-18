import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
def apply():
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","DejaVu Sans"],
        "font.size":7,"axes.labelsize":7,"axes.titlesize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.5,
        "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.major.size":2.5,"ytick.major.size":2.5,
        "xtick.direction":"out","ytick.direction":"out","axes.spines.top":False,"axes.spines.right":False,
        "axes.edgecolor":"#222222","axes.labelcolor":"#222222","xtick.color":"#222222","ytick.color":"#222222","text.color":"#222222",
        "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":600,"legend.frameon":False,"axes.grid":False,
        "lines.linewidth":0.9,"patch.linewidth":0.4})
def letter(ax,s,x=-0.14,y=1.04):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=9,fontweight="bold",va="bottom",ha="left")
# palette matched to Figure 1
REL={"activation":"#2E8B74","inhibition":"#C4692B","phosphorylation":"#7A5BA6"}
MOD={"Lookup":"#9A9A9A","LR-Lex":"#2E8B74","LR-Graph":"#8FB3D9","LR-Lex+Graph":"#7A5BA6","GBDT-Graph":"#C4692B","CharCNN":"#2B7BB9","GNN":"#B5446E"}
# KEGG pathway names (figures show names only, never map identifiers)
NAMES={"01522":"Endocrine resistance","01524":"Platinum drug resistance","04010":"MAPK","04012":"ErbB","04015":"Rap1","04022":"cGMP–PKG","04024":"cAMP",
 "04060":"Cytokine–receptor interaction","04061":"Viral–cytokine interaction","04114":"Oocyte meiosis","04115":"p53","0414":"Ras","04014":"Ras","04140":"Autophagy",
 "04141":"ER protein processing","04144":"Endocytosis","04150":"mTOR","04152":"AMPK","04210":"Apoptosis","04211":"Longevity","04215":"Apoptosis (multiple species)",
 "04227":"Necroptosis","04261":"Adrenergic signalling (cardiac)","0521":"EGFR-TKI resistance","01521":"EGFR-TKI resistance"}
def pname(p): return NAMES.get(str(p),str(p))
