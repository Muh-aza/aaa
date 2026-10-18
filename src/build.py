import pandas as pd, glob, os, sys
base=sys.argv[1]; rows=[]
for p in sorted(glob.glob(base+"/0*/*.csv")):
    pid=os.path.basename(os.path.dirname(p))
    df=pd.read_excel(p,engine="openpyxl")
    extra=[c for c in df.columns[3:] if df[c].notna().any()]
    df=df.iloc[:,:3]; df.columns=["starter","receiver","relation"]
    df=df.dropna()
    for c in df: df[c]=df[c].astype(str).str.strip()
    df["pathway"]=pid
    print(pid, len(df), dict(df.relation.value_counts()), "extra-nonnull:",extra)
    rows.append(df)
A=pd.concat(rows)
import nstyle; A["pathway"]=A.pathway.map(nstyle.pname)  # store pathway names, not KEGG map identifiers
 print(A.shape); print(A.relation.value_counts())
A.to_csv(sys.argv[2],index=False)
print("unique triples",A[["starter","receiver","relation"]].drop_duplicates().shape, "unique pairs",A[["starter","receiver"]].drop_duplicates().shape)
print("pairs w/ >1 label", (A.drop_duplicates(["starter","receiver","relation"]).groupby(["starter","receiver"]).size()>1).sum())
print("genes", len(set(A.starter)|set(A.receiver)))
