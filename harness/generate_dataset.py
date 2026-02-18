#!/usr/bin/env python3
"""
If the datasets are too large to include, generate them here or pull them 
from a storage source.
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
from pathlib import Path
import warnings
import numpy as np

# NOTE: Do not import dataset modules at top-level.  Dataset modules
# register absl flags when imported which can cause DuplicateFlagError if
# more than one dataset module is imported in the same process. Import the
# requested dataset lazily inside `main()` so dataset files can keep their
# flag definitions.

# FIXME: Make Flags global to remove lazy import and simply the code.

def main():
    """
    Usage:  python3 generate_dataset.py  <output_file>
    """

    if len(sys.argv) != 3:
        sys.exit("Usage: generate_dataset.py <output_file> [dataset_name]")

    DATASET_PATH = Path(sys.argv[1])
    DATASET_NAME = sys.argv[2]
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATASET_NAME == "mnist":
        from mnist import mnist as mnist_mod
        mnist_mod.export_test_data(output_file=DATASET_PATH, num_samples=10000, seed=None)
    elif DATASET_NAME == "cifar10":
        from cifar10 import cifar10 as cifar10_mod
        cifar10_mod.export_test_data(output_file=DATASET_PATH, num_samples=10000, seed=None)
    else:
        raise ValueError(f"Unsupported dataset name: {DATASET_NAME}")


if __name__ == "__main__":
    main()
