import pickle
import sys
from lattica_query import worker_api
import lattica_query.query_toolkit as toolkit_interface

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

context = pickle.load(open(local_file_paths.PATH_CONTEXT, "rb"))
sk = pickle.load(open(local_file_paths.PATH_SK, "rb"))

results = []
for i in range(local_file_paths.BATCH_SIZE):
    ct_res = pickle.load(open(local_file_paths.get_ct_download_path(i), "rb"))
    res = toolkit_interface.dec(context, sk, ct_res, False)
    res = worker_api.load_proto_tensor(res)
    res = res.argmax()
    results.append(res)

with open(local_file_paths.PREDICTIONS_PATH, "w") as f:
    f.write("\n".join(str(int(r)) for r in results) + "\n")
