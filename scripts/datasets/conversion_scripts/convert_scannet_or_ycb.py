import argparse
import gzip
import io
import os

import cv2
import numpy as np
import webdataset
from tqdm import tqdm
from utils import ContextList, FakeIndices, make_subdirs_and_patterns

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--data_path", type=str, default="outputs/scannet")
parser.add_argument("--output_path", type=str, default="data/scannet")
args = parser.parse_args()


def get_numpy_compressed_bytes(np_array):
    """Convert into a serialized numpy array."""
    with io.BytesIO() as stream:
        np.save(stream, np_array)
        return gzip.compress(stream.getvalue(), compresslevel=5)


dir = f"{args.data_path}/{args.split}"

image_dir = os.path.join(dir, "image")
mask_dir = os.path.join(dir, "mask")

image_list = os.listdir(image_dir)
mask_list = os.listdir(mask_dir)

image_list.sort()
mask_list.sort()

split_names = [args.split]
split_indices = {split_name: FakeIndices() for split_name in split_names}
output_path = args.output_path
patterns, _ = make_subdirs_and_patterns(output_path, split_indices)

shard_writer_params = {
    "maxsize": 100 * 1024 * 1024,  # 100 MB
    "maxcount": 50,
    "keep_meta": True,
}

uniq_val = None
# Create shards of the data.
valid_count = 0
images = []
masks = []
max_number_of_objects = 0
for j in tqdm(range(len(mask_list))):
    image = cv2.imread(os.path.join(image_dir, image_list[j]))
    mask = cv2.imread(os.path.join(mask_dir, mask_list[j]))
    images.append(image)
    masks.append(mask)
    if mask.max() > max_number_of_objects:
        max_number_of_objects = mask.max()
with ContextList(webdataset.ShardWriter(p, **shard_writer_params) for p in patterns) as writers:

    writer = writers[0]

    for index, (img, mask) in tqdm(enumerate(zip(images, masks))):
        one_hot_masks = np.zeros((max_number_of_objects + 1, img.shape[0], img.shape[1]))
        for i in range(0, max_number_of_objects + 1):
            # assert (mask[:, :, 0] == mask[:, :, 1]).all()
            one_hot_masks[i] = (mask[:, :, 0] == i).astype(int)

        one_hot_masks = np.expand_dims(one_hot_masks, axis=-1)
        # print (one_hot_masks.shape)
        output = {
            "image.npy.gz": get_numpy_compressed_bytes(img),
            "mask.npy.gz": get_numpy_compressed_bytes(one_hot_masks),
        }
        output["__key__"] = str(index)
        writer.write(output)
