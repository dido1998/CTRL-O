import webdataset as wds
from tqdm import tqdm

# Replace with your shard path or URL
shard_path = "/home/mila/a/aniket.didolkar/scratch/language_conditioned_oclf/scripts/datasets/data/coco2017/val/shard-{000000..000014}.tar"

# Create a dataset from the shards
dataset = wds.WebDataset(shard_path)

# Iterate through the dataset and print the keys for each sample


import gzip
import io
import json

import numpy as np
import webdataset as wds

# Define the dataset path

# Define the keys to read
keys = [
    "__key__",
    "__url__",
    "all_bbox_centroids.npy",
    "bbox_centroids.npy",
    "caption.json",
    "contrastive_loss_mask.npy",
    "image.npy.gz",
    "instance_area.npy",
    "instance_bbox.npy",
    "instance_category.npy",
    "instance_iscrowd.npy",
    "instance_mask.npy.gz",
    "name.json",
    "name_embedding.npy.gz",
    "selected_indices.npy",
]

# Function to decode numpy arrays
def decode_numpy(np_str):
    return np.load(io.BytesIO(np_str))


# Function to decode gzip numpy arrays
def decode_numpy_gzip(np_str):
    with gzip.GzipFile(fileobj=io.BytesIO(np_str)) as f:
        return np.load(f)


# Function to decode json
def decode_json(json_str):
    return json.loads(json_str)


# Create a WebDataset pipeline


# Iterate over the dataset and read the specified keys
for sample in tqdm(dataset):
    data = {}

    instance_mask = decode_numpy_gzip(sample["instance_mask.npy.gz"])
    selected_indices = decode_numpy(sample["selected_indices.npy"])

    # Do something with the data
    # Example: print the shape of one of the numpy arrays
    if max(selected_indices) > instance_mask.shape[0]:
        print("Error")
