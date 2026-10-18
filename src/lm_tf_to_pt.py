import sys, torch, numpy as np, shutil
from transformers import BertConfig, BertModel
src, npz, out = sys.argv[1:4]
cfg = BertConfig.from_json_file(f"{src}/bert_config.json")
m = BertModel(cfg); sd = m.state_dict(); z = np.load(npz); new = {}
for n in z.files:
    k = n.replace("__", ".").replace("layer_", "layer.").replace("gamma","weight").replace("beta","bias")
    v = z[n]
    if k.endswith("kernel"): k = k[:-6]+"weight"; v = v.T
    if k.endswith("_embeddings"): k += ".weight"
    new[k.replace("bert.","",1)] = torch.tensor(np.ascontiguousarray(v))
print("missing", [k for k in sd if k not in new]); print("extra", [k for k in new if k not in sd])
print(m.load_state_dict(new, strict=False)); m.save_pretrained(out); shutil.copy(f"{src}/vocab.txt", out)
