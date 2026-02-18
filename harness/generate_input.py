#!/usr/bin/env python3
"""
Generate a new input for each run.
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
from pathlib import Path
import utils
from utils import parse_submission_arguments

# FIXME: Make Flags global to remove lazy import and simply the code.

def main():
    """
    Generate random value representing the query in the workload.
    """
    size, params, seed, num_runs, clrtxt, remote_be, model_name, dataset_name = utils.parse_submission_arguments('Generate input for FHE benchmark.')
    PIXELS_PATH = params.get_test_input_file()
    LABELS_PATH = params.get_ground_truth_labels_file()
    

    PIXELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    num_samples = params.get_batch_size()

    # check the dataset name and call the corresponding function to generate the test pixels and labels. 
    # The dataset modules will be imported lazily inside this function to avoid flag registration conflicts.
    if dataset_name == "mnist":
        from mnist import mnist as mnist_mod
        mnist_mod.export_test_pixels_labels(
                data_dir = params.datadir(), 
                pixels_file=PIXELS_PATH, 
                labels_file=LABELS_PATH, 
                num_samples=num_samples, 
                seed=seed)
    elif dataset_name == "cifar10":
        from cifar10 import cifar10 as cifar10_mod
        cifar10_mod.export_test_pixels_labels(
                data_dir = params.datadir(), 
                pixels_file=PIXELS_PATH, 
                labels_file=LABELS_PATH, 
                num_samples=num_samples, 
                seed=seed)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

if __name__ == "__main__":
    main()
