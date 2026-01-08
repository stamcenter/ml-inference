import pickle
import sys

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]), mute_logs=False)

client = submission_utils.get_lattica_client(local_file_paths)

ct = pickle.load(open(local_file_paths.get_ct_upload_path("batch"), "rb"))
ct_res = client.worker_api.apply_hom_pipeline(ct, block_index=1)
pickle.dump(ct_res, open(local_file_paths.get_ct_download_path("batch"), "wb"))
