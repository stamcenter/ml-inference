from pathlib import Path
import os
import pickle
import sys

from lattica_query.lattica_query_client import QueryClient

instance_name = ["single", "small", "medium", "large"]
batch_size = [1, 15, 1000, 10000]

def init(size):
    return LocalFilePaths(size)

class LocalFilePaths:
    def __init__(self, size):
        # Mute logs
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)

        self.BATCH_SIZE = batch_size[size]

        PARENT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.PT_PATH = PARENT_DIR / "datasets" / instance_name[size] / "intermediate" / "test_pixels.txt"
        self.IO_DIR =  PARENT_DIR / "io"       / instance_name[size]

        self.PREDICTIONS_PATH =  self.IO_DIR / "encrypted_model_predictions.txt"

        self.CLIENT_DATA_DIR = self.IO_DIR / "client_data"
        self.CLIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PATH_CONTEXT = self.CLIENT_DATA_DIR / "context.pkl"
        self.PATH_HOM_SEQ = self.CLIENT_DATA_DIR / "client_sequential_hom_op.pkl"
        self.PATH_ACCESS_TOKEN = self.CLIENT_DATA_DIR / "access_token.pkl"

        self.SK_DIR = self.IO_DIR / "secret_key"
        self.SK_DIR.mkdir(parents=True, exist_ok=True)
        self.PATH_SK = self.SK_DIR / "sk.pkl"

        self.PK_DIR = self.IO_DIR / "public_keys"
        self.PK_DIR.mkdir(parents=True, exist_ok=True)
        self.PATH_EK = self.PK_DIR / "ek.pkl"

        self.CT_UPLOAD_DIR = self.IO_DIR / "ciphertexts_upload"
        self.CT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        self.CT_DOWNLOAD_DIR = self.IO_DIR / "ciphertexts_download"
        self.CT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def get_ct_upload_path(self, i):
        return self.CT_UPLOAD_DIR / f"cipher_input_{i}.pkl"

    def get_ct_download_path(self, i):
        return self.CT_DOWNLOAD_DIR / f"cipher_result_{i}.pkl"

def get_lattica_client(local_file_paths):
    access_token = pickle.load(open(local_file_paths.PATH_ACCESS_TOKEN, "rb"))
    return QueryClient(access_token)
