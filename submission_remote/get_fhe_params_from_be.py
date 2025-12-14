from lattica_query.auth import get_demo_token
from lattica_query.serialization.hom_op_pb2 import ClientData as ProtoClientData

import sys
import pickle

import submission_utils

local_file_paths = submission_utils.init(int(sys.argv[1]))

# Get access_token for public model
access_token = get_demo_token("sketchToNumber")
pickle.dump(access_token, open(local_file_paths.PATH_ACCESS_TOKEN, "wb"))

# Get encryption params and model metadata from BE
client = submission_utils.get_lattica_client(local_file_paths)
client_data = client.worker_api.get_user_init_data()

# Parse
client_data_proto = ProtoClientData()
client_data_proto.ParseFromString(client_data)
context = client_data_proto.serialized_context
hom_seq = client_data_proto.serialized_client_sequential_hom_op

# Save data to local file system
pickle.dump(context, open(local_file_paths.PATH_CONTEXT, "wb"))
pickle.dump(hom_seq, open(local_file_paths.PATH_HOM_SEQ, "wb"))
