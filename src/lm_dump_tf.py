import sys, numpy as np, tensorflow as tf
r = tf.train.load_checkpoint(sys.argv[1])
d = {n.replace("/","__"): r.get_tensor(n) for n in r.get_variable_to_shape_map() if "adam" not in n and "global_step" not in n}
np.savez(sys.argv[2], **d); print(len(d))
