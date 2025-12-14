import pickle
import torch
import sys


from lattica_query.lattica_query_client import QueryClient
import lattica_query.query_toolkit as toolkit_interface
from lattica_query.serialization.hom_op_pb2 import (
    QueryClientSequentialHomOp as ProtoQueryClientSequentialHomOp,
)
from lattica_query import worker_api

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

# Read inputs from file
examples = []
with open(local_file_paths.PT_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        pix = [float(p) for p in line.split(" ")]
        if len(pix) != 28 * 28:
            raise ValueError(f"Expected 784 pixels, got {len(pix)}")
        examples.append(pix)


# Read data from local filesystem required for encoding and encrypting
context = pickle.load(open(local_file_paths.PATH_CONTEXT, "rb"))
hom_seq = pickle.load(open(local_file_paths.PATH_HOM_SEQ, "rb"))
sk =      pickle.load(open(local_file_paths.PATH_SK, "rb"))
homsec_proto = ProtoQueryClientSequentialHomOp()
homsec_proto.ParseFromString(hom_seq)
block_proto = homsec_proto.client_blocks[0]
pt_axis_external = block_proto.pt_axis_external if block_proto.HasField("pt_axis_external") else None

# For each input in the batch create a ciphertext
for i, pt in enumerate(examples):
    pt = torch.tensor(pt).reshape(28, 28)
    pt = worker_api.dumps_proto_tensor(pt)
    pt = toolkit_interface.apply_client_block(
        block_proto.SerializeToString(),
        context,
        pt
    )
    ct = toolkit_interface.enc(
        context,
        sk,
        pt,
        pack_for_transmission=True,
        n_axis_external=pt_axis_external
    )
    pickle.dump(ct, open(local_file_paths.get_ct_upload_path(i), "wb"))

