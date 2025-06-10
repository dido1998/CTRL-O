"""Convert a multi-object records dataset into a webdataset."""
import abc
import argparse
import gzip
import io
import json
import logging
import os
import pathlib
import random
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm
import webdataset
from PIL import Image
from pycocotools.coco import COCO
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)


class Handler(abc.ABC):
    @abc.abstractmethod
    def is_responsible(self, instance: Any) -> bool:
        pass

    @abc.abstractmethod
    def __call__(self, name: str, objs: Any) -> Tuple[str, bytes]:
        pass


class NumpyHandler(Handler):
    def is_responsible(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def __call__(self, name: str, obj: np.ndarray) -> bool:
        with io.BytesIO() as stream:
            np.save(stream, obj)
            return f"{name}.npy", stream.getvalue()


class JsonHandler(Handler):
    def is_responsible(self, obj: Any) -> bool:
        return isinstance(obj, (list, dict))

    def __call__(self, name: str, obj: np.ndarray) -> bool:
        return f"{name}.json", json.dumps(obj).encode("utf-8")


class GzipHandler(Handler):
    def __init__(self, compresslevel=5):
        self.compresslevel = compresslevel

    def is_responsible(self, obj: Any) -> bool:
        # Only compress if file is larger than a block, otherwise compression
        # is useless because the file will anyway use 512 bytes of space.
        return isinstance(obj, bytes) and len(obj) > tarfile.BLOCKSIZE

    def __call__(self, name: str, obj: bytes) -> bool:
        return f"{name}.gz", gzip.compress(obj, compresslevel=self.compresslevel)


DEFAULT_HANDLERS = [NumpyHandler(), JsonHandler(), GzipHandler()]


class AnnotationAggregator:
    """Class to aggregate COCO annotations from multiple COCO tasks."""

    def __init__(
        self,
        instance_annotation: str,
        stuff_annotation: str,
        caption_annotation: str,
        panoptic_annotation: str,
    ):
        self.instance_annotation = COCO(instance_annotation) if instance_annotation else None
        self.stuff_annotation = COCO(stuff_annotation) if stuff_annotation else None
        self.caption_annotation = COCO(caption_annotation) if caption_annotation else None

        if panoptic_annotation:
            self.panoptic_segmentations_folder = panoptic_annotation.rsplit(".", 1)[0]
            with open(panoptic_annotation, "r") as f:
                self.panoptic_annotation = json.load(f)
            self.panoptic_filenames_by_id = {
                ann["id"]: ann["file_name"] for ann in self.panoptic_annotation["images"]
            }
            self.panoptic_annotations_by_id = {
                ann["image_id"]: ann for ann in self.panoptic_annotation["annotations"]
            }
            self.categories_by_id = {
                cat["id"]: cat for cat in self.panoptic_annotation["categories"]
            }
        else:
            self.panoptic_segmentations_folder = None
            self.panoptic_annotation = None
            self.panoptic_filenames_by_id = None
            self.panoptic_annotations_by_id = None
            self.categories_by_id = None

        if self.caption_annotation:
            self.image_ids = sorted(list(self.caption_annotation.imgs.keys()))
        if self.instance_annotation:
            self.image_ids = sorted(list(self.instance_annotation.imgs.keys()))
        if self.stuff_annotation:
            self.image_ids = sorted(list(self.stuff_annotation.imgs.keys()))
        if self.panoptic_annotation:
            self.image_ids = sorted(list(self.panoptic_filenames_by_id))

    def _get_filename(self, image_id):
        if self.caption_annotation:
            return self.caption_annotation.loadImgs(image_id)[0]["file_name"]
        if self.instance_annotation:
            return self.instance_annotation.loadImgs(image_id)[0]["file_name"]
        if self.stuff_annotation:
            return self.stuff_annotation.loadImgs(image_id)[0]["file_name"]
        if self.panoptic_annotation:
            return self.panoptic_filenames_by_id[image_id]
        raise RuntimeError()

    def _get_segmentation_annotations(self, coco_object, prefix, image_id, include_crowd=False):
        ann_ids = coco_object.getAnnIds(image_id)
        annotations = coco_object.loadAnns(ann_ids)
        masks = [coco_object.annToMask(annotation) for annotation in annotations]
        output = {}
        if len(masks) > 0:
            # Stack to single array and add final dimension for compatibility with clevr dataset.
            output[f"{prefix}mask"] = np.stack(masks, axis=0)[..., None]
            output[f"{prefix}bbox"] = np.array(
                [annotation["bbox"] for annotation in annotations], dtype=np.float32
            )
            output[f"{prefix}category"] = np.array(
                [annotation["category_id"] for annotation in annotations],
                dtype=np.uint8,
            )
            output[f"{prefix}area"] = np.array(
                [annotation["area"] for annotation in annotations], dtype=np.float32
            )
            if include_crowd:
                output[f"{prefix}iscrowd"] = np.array(
                    [annotation["iscrowd"] for annotation in annotations], dtype=np.float32
                )
        return output

    @staticmethod
    def _panoptic_rgb2id(color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def _get_panoptic_annotations(self, image_id):
        annotations = self.panoptic_annotations_by_id[image_id]
        file_name = "{}.png".format(annotations["file_name"].rsplit(".")[0])
        try:
            pan_format = np.array(
                Image.open(os.path.join(self.panoptic_segmentations_folder, file_name)),
                dtype=np.uint32,
            )
        except IOError:
            raise KeyError("No prediction png file for id: {}".format(annotations["image_id"]))

        pan = self._panoptic_rgb2id(pan_format)

        masks = []
        for segm_info in annotations["segments_info"]:
            mask = (pan == segm_info["id"]).astype(np.uint8)
            masks.append(mask)

        output = {}
        output["panoptic_mask"] = np.stack(masks, axis=0)[..., None]
        output["panoptic_bbox"] = np.array(
            [segm_info["bbox"] for segm_info in annotations["segments_info"]], dtype=np.float32
        )
        output["panoptic_category"] = np.array(
            [segm_info["category_id"] for segm_info in annotations["segments_info"]], dtype=np.uint8
        )
        output["panoptic_area"] = np.array(
            [segm_info["area"] for segm_info in annotations["segments_info"]], dtype=np.float32
        )
        output["panoptic_iscrowd"] = np.array(
            [segm_info["iscrowd"] for segm_info in annotations["segments_info"]], dtype=np.uint8
        )
        output["panoptic_isthing"] = np.array(
            [
                self.categories_by_id[segm_info["category_id"]]["isthing"]
                for segm_info in annotations["segments_info"]
            ],
            dtype=np.uint8,
        )
        return output

    def _get_caption_annotations(self, image_id):
        ann_ids = self.caption_annotation.getAnnIds(image_id)
        annotations = self.caption_annotation.loadAnns(ann_ids)
        # Stack to single array and add final dimension for compatibility with clevr dataset.
        return {"caption": [annotation["caption"] for annotation in annotations]}

    def __getitem__(self, image_id) -> Tuple[str, Dict]:
        """Get file name and aggregated annotations for image id."""
        filename = self._get_filename(image_id)
        annotations = {}
        if self.caption_annotation:
            annotations.update(self._get_caption_annotations(image_id))
        if self.instance_annotation:
            annotations.update(
                self._get_segmentation_annotations(
                    self.instance_annotation, "instance_", image_id, include_crowd=True
                )
            )
        if self.stuff_annotation:
            annotations.update(
                self._get_segmentation_annotations(self.stuff_annotation, "stuff_", image_id)
            )
        if self.panoptic_annotation:
            annotations.update(self._get_panoptic_annotations(image_id))

        return filename, annotations


class TestAnnotations:
    def __init__(self, test_annotation: str):
        self.test_annotation = COCO(test_annotation)
        self.image_ids = sorted(list(self.test_annotation.imgs.keys()))

    def _get_filename(self, image_id):
        return self.test_annotation.loadImgs(image_id)[0]["file_name"]

    def __getitem__(self, image_id) -> Tuple[str, Dict]:
        """Get file name and aggregated annotations for image id."""
        filename = self._get_filename(image_id)
        return filename, {}


def convert_to_bytes(name, obj, handlers: List[Handler] = DEFAULT_HANDLERS):
    for handler in handlers:
        if handler.is_responsible(obj):
            name, obj = handler(name, obj)
    return name, obj


def main(
    dataset_path: str,
    output_path: str,
    instance_annotation: str,
    stuff_annotation: str,
    panoptic_annotation: str,
    caption_annotation: str,
    test_annotation: str,
    subset_list: Optional[str] = None,
    seed: Optional[int] = None,
    only_images: bool = False,
):
    if instance_annotation or stuff_annotation or caption_annotation or panoptic_annotation:
        annotator = AnnotationAggregator(
            instance_annotation, stuff_annotation, caption_annotation, panoptic_annotation
        )
    elif test_annotation:
        annotator = TestAnnotations(test_annotation)
    else:
        raise RuntimeError(
            "Either instance, stuff, panoptic, and caption annotations or test annotations must be "
            "provided."
        )

    # Create output directory
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    if subset_list:
        with open(subset_list, "r") as f:
            subset_list = set(line.strip() for line in f.readlines())

    # Setup parameters for shard writers.
    shard_writer_params = {
        "maxsize": 50 * 1024 * 1024,  # 50 MB
        "maxcount": 1000,
        "keep_meta": True,
        "encoder": False,
    }

    image_ids = list(annotator.image_ids)  # Make copy.
    print("Number of instances:", len(image_ids))
    random.seed(seed)
    random.shuffle(image_ids)  # Ensure instances are shuffled.

    instance_count = 0
    with webdataset.ShardWriter(get_shard_pattern(output_path), **shard_writer_params) as writer:
        for image_id in tqdm.tqdm(image_ids):
            filename, annotations = annotator[image_id]
            if subset_list:
                if filename in subset_list:
                    subset_list.remove(filename)
                else:
                    # Skip samples not in subset list
                    continue

            if only_images:
                output = {}
            else:
                output = dict([convert_to_bytes(name, obj) for name, obj in annotations.items()])
            output["__key__"] = str(image_id)
            _, ext = os.path.splitext(filename)
            image_path = os.path.join(dataset_path, filename)
            with open(image_path, "rb") as f:
                output[f"image{ext}"] = f.read()
            # print(output.keys())
            writer.write(output)
            instance_count += 1
    if subset_list is not None and len(subset_list) != 0:
        # We did not process all images in the list.
        logging.error(f"{len(subset_list)} samples in the subset list where not processed")

    logging.info(f"Wrote {instance_count} instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--stuff", type=str, default=None)
    parser.add_argument("--panoptic", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--subset_list", type=str, default=None)
    parser.add_argument("--seed", type=int, default=23894734)
    parser.add_argument("--only-images", action="store_true", default=None)

    args = parser.parse_args()

    main(
        args.dataset_path,
        args.output_path,
        args.instance,
        args.stuff,
        args.panoptic,
        args.caption,
        args.test,
        args.subset_list,
        args.seed,
        args.only_images,
    )
