import torch, shutil, sys
from transformers import BertModel, BertConfig
m=BertModel.from_pretrained(sys.argv[1]); m.save_pretrained(sys.argv[2]); shutil.copy(sys.argv[1]+"/vocab.txt",sys.argv[2])
sd=torch.load(sys.argv[3],map_location="cpu"); p="text_field_embedder.token_embedder_bert.bert_model."
new={k[len(p):]:v for k,v in sd.items() if k.startswith(p)}
s=BertModel(m.config); print(s.load_state_dict(new,strict=False)); s.save_pretrained(sys.argv[4]); shutil.copy(sys.argv[1]+"/vocab.txt",sys.argv[4])
