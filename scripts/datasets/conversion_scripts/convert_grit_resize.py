import argparse
import json
import logging
import os
import pathlib
import random
import tarfile
from typing import Any, Dict, Iterator, Optional

import cv2
import handlers
import numpy as np
import torch
import webdataset
from llm2vec import LLM2Vec
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)


l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token


def visualize_masks_and_labels(formatted_data: Dict) -> None:
    """
    Visualize the instance masks and labels directly on the image using OpenCV.

    :param formatted_data: The formatted data containing instance masks and labels.
    """
    # Extract relevant data
    image = formatted_data["image"]  # The resized image
    instance_mask = formatted_data["instance_mask_v2"]  # The instance masks (H, W, num_objects)
    names = formatted_data["name"]

    # Create a colored overlay from the instance masks
    colored_mask = np.zeros_like(image)

    for idx in range(instance_mask.shape[-1]):
        mask = instance_mask[:, :, idx]
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

        # Apply the mask to the colored overlay
        colored_mask[mask == 1] = color

        # Get the center of the bounding box for the label
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            label = names[idx]

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate the text size to center the text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2

            # Put the label at the center of the bounding box
            cv2.putText(
                colored_mask,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Blend the colored mask with the original image
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0, image)

    # Save the image to a file
    image_path = f"test/{formatted_data['__key__']}.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, image)
    print(f"Image saved as {image_path}")


def check_dict_for_nan(data_dict: Dict[str, Any]):
    """
    Check all numpy arrays in a dictionary for NaN values and raise an error if any are found.

    :param data_dict: A dictionary containing various elements, potentially including numpy arrays.
    """
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):  # Check if the value is a numpy array
            if np.isnan(value).any():
                logging.warning(f"NaN values found in formatted data for {key}")
                return True
            if value.size == 0:  # Check if the array is empty
                logging.warning(f"Empty array found in formatted data for {key}")
                return True

    return False


def format_item(data: Dict, image: np.ndarray) -> Dict:
    """
    Convert the input data dictionary from COYO-700M to the desired format for training.

    :param data: A dictionary containing raw data from COYO-700M.
    :param image: The corresponding image as a numpy array.
    :return: A dictionary formatted for training.
    """

    # Resize the image first
    resized_image = cv2.resize(image, (224, 224))
    height, width, _ = resized_image.shape  # Use resized image dimensions

    # Extract the necessary fields
    caption = data.get("caption", "").strip()
    noun_chunks = data.get("noun_chunks", [])

    # Initialize lists
    bbox_centroids = []
    selected_indices = []
    contrastive_loss_mask = []
    names = []
    bounding_boxes = []

    # Process noun chunks to extract required information
    for i, chunk in enumerate(noun_chunks):
        if i >= 7:  # Limit to 7
            break
        start_idx, end_idx, x_min, y_min, x_max, y_max, confidence = chunk
        text_chunk = caption[int(start_idx) : int(end_idx)]
        names.append(text_chunk)
        bounding_boxes.append((x_min, y_min, x_max, y_max))
        centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2]
        bbox_centroids.append(centroid)
        selected_indices.append(i)
        contrastive_loss_mask.append(1)

    # Ensure bbox_centroids and selected_indices have exactly 7 elements
    while len(bbox_centroids) < 7:
        names.append("")  # Fill with empty strings
        bbox_centroids.append([-1, -1])  # Fill with placeholder coordinates
        selected_indices.append(-1)
        contrastive_loss_mask.append(0)

    # Create instance masks directly in this function
    num_objects = len(bounding_boxes)
    instance_mask = np.zeros((height, width, num_objects), dtype=np.uint8)

    for idx, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
        # Rescale the bounding box coordinates to match the resized image
        x_min = x_min * width
        y_min = y_min * height
        x_max = x_max * width
        y_max = y_max * height

        mask_channel = instance_mask[:, :, idx]  # Select the 2D slice
        mask_channel = np.ascontiguousarray(mask_channel)  # Ensure the slice is contiguous
        cv2.rectangle(
            mask_channel, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=1, thickness=-1
        )
        instance_mask[:, :, idx] = mask_channel  # Store the updated mask channel

    instance_mask = instance_mask.transpose(2, 0, 1)  # Transpose to (num_objects, H, W)
    q_reps = l2v.encode(names, show_progress_bar=False)
    q_reps = q_reps.cpu().numpy()
    q_reps = np.stack(q_reps, axis=0)  # stack the embeddings

    names_as_str = " ".join([f"{name.strip()}." for name in names])
    target_ids = tokenizer.encode(
        names_as_str,
        add_special_tokens=False,
        padding="max_length",  # Pad to the max sequence length in the batch
        truncation=True,  # Truncate sequences that are too long
        max_length=20,  # Set the max length to 5
        return_tensors="np",  # Return as numpy arrays
    )[0]
    # Construct the formatted data dictionary
    formatted_data = {
        "__key__": str(data.get("id", "")),  # Using 'id' as the key
        "all_bbox_centroids": np.array(bbox_centroids),  # Updated centroids after resizing
        "bbox_centroids": np.array(bbox_centroids),  # Use the first 7 centroids or fill
        "contrastive_loss_mask": np.array(contrastive_loss_mask),
        "image": resized_image,
        "instance_mask": instance_mask,  # The created instance masks based on the resized image
        "name": names,
        "name_embedding": q_reps,
        "target_ids": target_ids,
        "selected_indices": np.array(selected_indices),
    }

    return formatted_data


def load_annotations(dataset_path: str, visualize: bool = False) -> Iterator[Dict]:
    tar_files = [f for f in os.listdir(dataset_path) if f.endswith(".tar")]
    for filename in tar_files:
        tar_file = os.path.join(dataset_path, filename)
        with tarfile.open(tar_file, "r") as tar:
            members = tar.getmembers()

            # Create a lookup dictionary for images
            image_lookup = {}
            for member in members:
                if member.isfile() and member.name.endswith(".jpg"):
                    base_name = os.path.splitext(os.path.basename(member.name))[0]
                    image_lookup[base_name] = member

            for member in tqdm(members, desc=filename):
                if member.isfile() and member.name.endswith(".json"):
                    json_content = tar.extractfile(member).read()
                    data = json.loads(json_content)

                    base_name = os.path.splitext(os.path.basename(member.name))[0]
                    image_member = image_lookup.get(base_name)

                    if image_member:
                        try:
                            image_content = tar.extractfile(image_member).read()
                            if image_content is None:
                                raise ValueError(f"Failed to read image content from tar file: {image_member.name}")

                            image_array = np.frombuffer(image_content, np.uint8)
                            if image_array.size == 0:
                                raise ValueError(f"Image array is empty for image: {image_member.name}")

                            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                            if image is None:
                                raise ValueError(f"Failed to decode image: {image_member.name}")

                        except Exception as e:
                            logging.error(f"Error processing image {image_member.name}: {e}")
                            continue

                        formatted_data = format_item(data, image)
                        if check_dict_for_nan(formatted_data) or len(formatted_data["name"]) < 1:
                            continue 
                        
                        if visualize:
                            visualize_masks_and_labels(formatted_data)

                        yield formatted_data


def main(
    dataset_path: str,
    output_path: str,
    seed: int,
    category_embedding_path: Optional[str],
    test_split: float = 0.2,  # Default to 20% test split
):
    # Prepare output directories
    train_output_path = os.path.join(output_path, "train")
    test_output_path = os.path.join(output_path, "test")
    pathlib.Path(train_output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_output_path).mkdir(parents=True, exist_ok=True)

    # Setup parameters for shard writers
    shard_writer_params = {
        "maxsize": 50 * 1024 * 1024,  # 50 MB
        "maxcount": 1000,
        "keep_meta": True,
        "encoder": False,
    }

    random.seed(seed)

    # Write to train and test shards
    train_writer = webdataset.ShardWriter(
        get_shard_pattern(train_output_path), **shard_writer_params
    )
    test_writer = webdataset.ShardWriter(get_shard_pattern(test_output_path), **shard_writer_params)

    instance_count_train, instance_count_test = 0, 0
    for formatted_data in load_annotations(dataset_path, visualize=False):
        output = dict([handlers.convert_to_bytes(name, obj) for name, obj in formatted_data.items()])
        output["__key__"] = formatted_data["__key__"]

        if random.random() > test_split:
            train_writer.write(output)
            instance_count_train += 1
        else:
            test_writer.write(output)
            instance_count_test += 1

    train_writer.close()
    test_writer.close()

    logging.info(
        f"Wrote {instance_count_train} training instances and {instance_count_test} test instances."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, required=True)
    parser.add_argument("output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=23894734)
    parser.add_argument("--category_embedding_path", type=str, default=None)
    parser.add_argument(
        "--test_split", type=float, default=0.2, help="Percentage of data to use for testing"
    )

    args = parser.parse_args()

    main(
        args.dataset_path,
        args.output_path,
        args.seed,
        args.category_embedding_path,
        args.test_split,
    )
