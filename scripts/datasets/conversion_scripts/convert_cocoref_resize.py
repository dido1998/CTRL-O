"""Convert a multi-object records dataset into a webdataset."""
import argparse
import json
import logging
import os
import pathlib
import pickle
import random
from typing import Dict, Optional, Tuple

import cv2
import torch
import handlers
import numpy as np
import tqdm
import webdataset
from PIL import Image
from pycocotools.coco import COCO
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)


def resize_bboxes(bboxes, original_size, new_size):
    """
    Resize bounding boxes according to the new image size.

    :param bboxes: Array of bounding boxes in the format [x_min, y_min, x_max, y_max]
    :param original_size: Tuple of original image size (width, height)
    :param new_size: Tuple of new image size (width, height)
    :return: Array of resized bounding boxes
    """
    ow, oh = original_size
    nw, nh = new_size
    scale_x = nw / ow
    scale_y = nh / oh

    resized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * scale_y)
        y_min = int(y_min * scale_x)
        x_max = int(x_max * scale_y)
        y_max = int(y_max * scale_x)
        resized_bboxes.append([x_min, y_min, x_max, y_max])

    return np.array(resized_bboxes)


def resize_masks(masks, new_size):
    """
    Resize instance masks to the new image size.

    :param masks: List of instance masks as numpy arrays
    :param new_size: Tuple of new image size (width, height)
    :return: List of resized instance masks
    """
    resized_masks = []
    for mask in masks:
        resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        resized_masks.append(resized_mask)

    return resized_masks


class AnnotationAggregator:
    """Class to aggregate COCO annotations from multiple COCO tasks."""

    def __init__(
        self,
        instance_annotation: str,
        stuff_annotation: str,
        caption_annotation: str,
        panoptic_annotation: str,
        category_embedding_path: Optional[str] = None,
        split: Optional[str] = None,
        is_refcoco: bool = False,
        dataset: Optional[str] = None,
        download_dir: Optional[str] = None,
    ):
        self.split = split
        self.instance_annotation = COCO(instance_annotation) if instance_annotation else None
        self.stuff_annotation = COCO(stuff_annotation) if stuff_annotation else None
        self.caption_annotation = COCO(caption_annotation) if caption_annotation else None
        self.is_refcoco = is_refcoco
        if is_refcoco:
            assert dataset is not None
            assert download_dir is not None
        if category_embedding_path is not None:
            with open(category_embedding_path, "rb") as f:
                self.category_name_to_embedding = pickle.load(f)

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
        if self.is_refcoco:
            assert self.instance_annotation
            self.dataset_split_map = {"refcocog": "google", "refcoco": "unc", "refcoco+": "unc"}
            self.dataset = dataset
            self.download_dir = download_dir
            if dataset in self.dataset_split_map:
                from refer import REFER
                refer = REFER(download_dir, dataset=dataset, splitBy=self.dataset_split_map[dataset])
                val_refs = [refer.Refs[ref_id] for ref_id in refer.Refs if refer.Refs[ref_id]["split"] == "val"]
                train_refs = [refer.Refs[ref_id] for ref_id in refer.Refs if refer.Refs[ref_id]["split"] == "train"]
                val_image_ids = sorted(list(set([ref["image_id"] for ref in val_refs])))
                train_image_ids = [ids for ids in self.image_ids if ids not in val_image_ids]
                train_with_ref_image_ids = sorted(list(set([ref["image_id"] for ref in train_refs])))
                if split == "val":
                    self.image_ids = val_image_ids
                    self.refs = val_refs
                elif split == "train":
                    for image_id in train_with_ref_image_ids:
                        assert image_id in self.image_ids, f"Image {image_id} not in image_ids"
                    self.image_ids = train_with_ref_image_ids
                    self.refs = train_refs
                else:
                    raise ValueError(f"Unknown split {split}")
                from llm2vec import LLM2Vec
                self.l2v = LLM2Vec.from_pretrained(
                    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.bfloat16,
                )
                set_of_ann_ids = set([ref["ann_id"] for ref in self.refs])
                self.ann_ids_to_refs = {ann_id: [] for ann_id in set_of_ann_ids}
                for ref in self.refs:
                    self.ann_ids_to_refs[ref["ann_id"]].append(ref)
            else:
                raise ValueError(f"Unknown dataset {dataset}")

    def get_additional_queries_emb(self, additional_queries):
        assert self.is_refcoco
        additional_queries_reps = self.l2v.encode(additional_queries, batch_size=1)
        category_name_to_embedding = dict(zip(additional_queries, additional_queries_reps))
        for key in category_name_to_embedding.keys():
            category_name_to_embedding[key] = category_name_to_embedding[key].cpu().numpy()
        return category_name_to_embedding

    @staticmethod
    def sentence_to_raw_sentences(sentences):
        return [sentence["sent"] for sentence in sentences]
        
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

            # sample points to make image (224, 224)
            num_masks = 0
            count = 0
            _masks = []
            bboxes = []
            bbox_centroids = []
            names = []
            categories = []
            area = []
            is_crowd = []
            if self.is_refcoco:
                references = []
                embeddings = []
            for annotation in annotations:
                if self.is_refcoco:
                    ann_id = annotation["id"]
                    if ann_id in self.ann_ids_to_refs:
                        print(f"Adding ref: {ann_id}")
                        ref = self.ann_ids_to_refs[ann_id][0]
                        #pick random ref from the list of available refs
                        #should be done in a better way
                        sent = [random.choice(ref["sentences"])]
                        sent = self.sentence_to_raw_sentences(sent)
                        _masks.append(resize_masks([coco_object.annToMask(annotation)], (224, 224))[0])
                        image_size = (masks[0].shape[0], masks[0].shape[1])
                        bbox = annotation["bbox"]
                        bbox = resize_bboxes([bbox], image_size, (224, 224))[0]
                        bboxes.append(bbox)
                        bbox_centroids.append([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
                        names.append(coco_object.loadCats(annotation["category_id"])[0]["name"])
                        categories.append(annotation["category_id"])
                        area.append(annotation["area"])
                        embedding = self.get_additional_queries_emb(sent)
                        embedding = np.stack(list(embedding.values()))
                        embeddings.append(embedding)
                        references.append(sent)
                        if "iscrowd" in annotation:
                            is_crowd.append(annotation["iscrowd"])
                    else:
                        pass

            assert len(_masks) > 0, "All the images should have at least one reference"
            selected_names = []
            selected_indices = []
            selected_bbox_centroids = []
            contrastive_mask = []
            if self.is_refcoco:
                selected_references = []
                selected_embeddings = []
            if len(names) > 7:
                already_selected = set()
                for k in range(7):
                    select_idx = random.randint(0, len(names) - 1)
                    while select_idx in already_selected:
                        select_idx = random.randint(0, len(names) - 1)
                    already_selected.add(select_idx)
                    selected_names.append(names[select_idx])
                    selected_indices.append(select_idx)
                    selected_bbox_centroids.append(bbox_centroids[select_idx])
                    contrastive_mask.append(1)
                    if self.is_refcoco:
                        selected_references.append(references[select_idx])
                        selected_embeddings.append(embeddings[select_idx])
            else:
                for k in range(len(names)):
                    selected_names.append(names[k])
                    selected_indices.append(k)
                    selected_bbox_centroids.append(bbox_centroids[k])
                    contrastive_mask.append(1)
                    if self.is_refcoco:
                        selected_references.append(references[k])
                        selected_embeddings.append(embeddings[k])
                for k in range(7 - len(names)):
                    selected_names.append("other")
                    selected_indices.append(-1)
                    selected_bbox_centroids.append([-1, -1])
                    contrastive_mask.append(0)
                    if self.is_refcoco:
                        selected_references.append(["other"])
                        other_embedding = self.category_name_to_embedding["other"][None, :]
                        selected_embeddings.append(other_embedding)
            output["selected_indices"] = np.array(selected_indices)
            output["contrastive_loss_mask"] = np.array(contrastive_mask)

            name_to_embedding = np.stack(
                [self.category_name_to_embedding[name] for name in selected_names], axis=0
            )
            # Stack to single array and add final dimension for compatibility with clevr dataset.
            output[f"name"] = list(selected_names)
            output[f"name_embedding"] = name_to_embedding
            if self.is_refcoco:
                output[f"references"] = selected_references
                output[f"references_embedding"] = np.concatenate(selected_embeddings, axis=0, dtype=np.float32)

            output[f"bbox_centroids"] = np.array(selected_bbox_centroids, dtype=np.float32)

            output[f"all_bbox_centroids"] = np.array(bbox_centroids, dtype=np.float32)
            output[f"{prefix}mask"] = np.stack(_masks, axis=0)[..., None]
            output[f"{prefix}bbox"] = np.array([bbox for bbox in bboxes], dtype=np.float32)
            output[f"{prefix}category"] = np.array(
                [category for category in categories], dtype=np.uint8
            )
            output[f"{prefix}area"] = np.array([a for a in area], dtype=np.float32)
            if include_crowd:
                output[f"{prefix}iscrowd"] = np.array([ic for ic in is_crowd], dtype=np.float32)
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
    category_embedding_path: Optional[str] = None,
    split: Optional[str] = None,
    is_refcoco: bool = False,
    dataset: Optional[str] = None,
    download_dir: Optional[str] = None,
):
    if instance_annotation or stuff_annotation or caption_annotation or panoptic_annotation:
        annotator = AnnotationAggregator(
            instance_annotation,
            stuff_annotation,
            caption_annotation,
            panoptic_annotation,
            category_embedding_path,
            split,
            is_refcoco,
            dataset,
            download_dir,
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
            if "name_embedding" not in annotations:
                continue
            if split == "val":
                if (
                    "instance_mask" in annotations
                    and max(annotations["selected_indices"])
                    > annotations["instance_mask"].shape[0] - 1
                ):
                    print(annotations["selected_indices"], len(annotations["instance_mask"]))
                    continue

            _, ext = os.path.splitext(filename)
            image_path = os.path.join(dataset_path, filename)
            assert os.path.exists(image_path), image_path
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))

            annotations["image"] = img

            if only_images:
                output = {}
            else:
                output = dict(
                    [handlers.convert_to_bytes(name, obj) for name, obj in annotations.items()]
                )
            output["__key__"] = str(image_id)
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
    parser.add_argument("--category_embedding_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--is_refcoco", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--download_dir", type=str, default=None)

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
        args.category_embedding_path,
        args.split,
        args.is_refcoco,
        args.dataset,
        args.download_dir,
    )
