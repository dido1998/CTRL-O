import os
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import dataclasses
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import ocl
from ocl.cli import eval_utils
from huggingface_hub import hf_hub_download
from typing import Optional
import argparse
import torch
from llm2vec import LLM2Vec
import pickle

os.environ['HF_TOKEN'] = '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'
# os.environ['HF_HOME'] = '/ssdstore/azadaia/.cache/huggingface'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


    
def get_additional_queries_emb(additional_queries):
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        )
    additional_queries_reps = l2v.encode(additional_queries, batch_size=1)
    category_name_to_embedding = dict(zip(additional_queries, additional_queries_reps))
    for key in category_name_to_embedding.keys():
        category_name_to_embedding[key] = category_name_to_embedding[key].cpu().numpy()
    
    with open("./scripts/app/additional_queries_emb.pkl", "rb+") as f:
        additional_queries_emb = pickle.load(f)
        additional_queries_emb = {**additional_queries_emb, **category_name_to_embedding}
        pickle.dump(additional_queries_emb, f)
    return category_name_to_embedding
    

@dataclasses.dataclass
class InferenceConfig:
    """Configuration for single image inference."""
    # repo_name: Optional[str] = "oclmodels/contrastive_loss_dinosaur_small_patch14_dinov2"
    repo_name: Optional[str] = "oclmodels/contrastive_loss_dinosaur_refcocog"
    # repo_name: Optional[str] = "oclmodels/dinosaur_only_language"
    checkpoint_path: Optional[str] = None
    train_config_path: Optional[str] = None
    image_path: str = "../notebooks/test.jpg"

# Initialize the config store
cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceConfig)

# Load the configuration
config = OmegaConf.structured(InferenceConfig)

def load_model(config: InferenceConfig, n_slots: int = 7):
    assert config.repo_name is not None or config.checkpoint_path is not None
    
    if config.repo_name is not None:
        repo_dir = os.path.join("./checkpoints" , config.repo_name)
        os.makedirs(repo_dir, exist_ok=True)
        config.train_config_path = hf_hub_download(
            repo_id=config.repo_name,
            filename="config.yaml",
            local_dir=repo_dir
        )
        config.checkpoint_path = hf_hub_download(
            repo_id=config.repo_name,
            filename="model.ckpt",
            local_dir=repo_dir
        )
        print(f"{config.train_config_path}")
    else:
        config.train_config_path = hydra.utils.to_absolute_path(config.train_config_path)
        config.checkpoint_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    
    config_dir, config_name = os.path.split(config.train_config_path)
    config_dir = os.path.abspath(config_dir)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir):
        train_config = hydra.compose(os.path.splitext(config_name)[0])
    train_config.models.conditioning.n_slots = n_slots
    datamodule, model = eval_utils.build_from_train_config(
        train_config,
        config.checkpoint_path,
        datamodule_kwargs=dict(eval_batch_size=1),
        checkpoint_hook=None,
    )
    return datamodule, model

def preprocess_image(image_path: str, device: torch.device):
    # Adjust these transforms based on your model's requirements
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def visualize_masks(output, n):
    pred_masks_matched = ocl.visualizations.Segmentation(
        denormalization=ocl.preprocessing.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    image_vis = ocl.visualizations.Image(
        denormalization=ocl.preprocessing.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    vis = pred_masks_matched(
        image=output['input']["image"],
        mask=output['object_decoder'].masks_as_image
    )

    image = image_vis(image=output['input']["image"])
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image.img_tensor.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(vis.img_tensor.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Predicted Masks Matched')
    os.makedirs(f"./notebooks/vis_{n}", exist_ok=True)
    plt.savefig(f"./notebooks/vis_{n}/masks.png")

def visualize_centroids(output, n):
    image_vis = ocl.visualizations.Image(
        denormalization=ocl.preprocessing.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    image = image_vis(image=output['input']["image"])
    centroids = output['input']['bbox_centroids'].cpu().numpy()[0]

    plt.figure(figsize=(8, 8))
    img = image.img_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    
    height, width = img.shape[:2]
    for i, centroid in enumerate(centroids):
        x = centroid[0] * width
        y = centroid[1] * height
        plt.plot(x, y, 'ro', markersize=10)
        plt.text(x, y, f"{i}", fontsize=12, color='red')
    
    plt.axis('off')
    plt.title('Image with Bounding Box Centroids')

    os.makedirs(f"../notebooks/vis_{n}", exist_ok=True)
    plt.savefig(f"../notebooks/vis_{n}/centroids.png")


def inference(config: InferenceConfig, n_examples: int = 100, n_slots: int = 7):
    datamodule, model = load_model(config, n_slots=n_slots)
    outputs = []
    for i, batch in enumerate(datamodule.val_dataloader()):
        if i < n_examples:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)   
            with torch.no_grad():
                output = model(batch)
            outputs.append(output)
    return outputs

def transform_batch(batch, change_queries):
    with open("./scripts/app/additional_queries_emb.pkl", "rb") as f:
        additional_queries_emb = pickle.load(f)

    print(batch["references"][0])
    for i in range(len(batch['name_embedding'][0])):
        if i in change_queries.keys():
            print(i)
            if change_queries[i]["name"] in additional_queries_emb.keys():
                print("here")
                batch['name_embedding'][0][i] = torch.Tensor(additional_queries_emb[change_queries[i]["name"]])
                batch['references_embedding'][0][i] = torch.Tensor(additional_queries_emb[change_queries[i]["name"]])
            else:
                batch['name_embedding'][0][i] = torch.Tensor(get_additional_queries_emb([change_queries[i]["name"]])[change_queries[i]["name"]])
                batch['references_embedding'][0][i] = torch.Tensor(get_additional_queries_emb([change_queries[i]["name"]])[change_queries[i]["name"]])
            batch['bbox_centroids'][0][i] = torch.Tensor(change_queries[i]["coors"])
            
            batch["name"][i] = change_queries[i]["name"]
    return batch

def visualize_masks_binary(output, n, n_slots):
    image_vis = ocl.visualizations.Image(
        denormalization=ocl.preprocessing.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    image = image_vis(image=output['input']["image"])
    masks = output['object_decoder'].masks_as_image[0]  # Shape: [7, 224, 224]

    plt.figure(figsize=(20, 15))

    # Display original image
    plt.subplot(3, 3, 1)
    plt.imshow(image.img_tensor.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Input Image')

    # Display individual masks and corresponding image parts
    for i in range(n_slots):
        # Display image part
        plt.subplot(3, 3, i+2)
        masked_image = image.img_tensor.cuda() * masks[i].unsqueeze(0)
        plt.imshow(masked_image.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(f'Image Part {i+1}')

    plt.tight_layout()
    os.makedirs(f"./notebooks/vis_{n}", exist_ok=True)
    plt.savefig(f"./notebooks/vis_{n}/masks_binary.png")
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Visualization App")
    parser.add_argument('--n', type=int, default=60, help='Batch number to process')
    args = parser.parse_args()
    n = args.n
    change_queries = {
        0: {"name": "head", "coors": [ 0.648,  0.23]}, 
        #               5: {"name": "person", "coors": [ 0.355,  0.714]},
        #               2: {"name": "plate", "coors": [ 0.5255,  0.884]}
                    }
    import pickle
    with open("./notebooks/additional_queries_emb.pkl", "rb") as f:
        additional_queries_emb = pickle.load(f)


    datamodule, model = load_model(config)
    for i, batch in enumerate(datamodule.val_dataloader()):
        if i == n:
            batch = transform_batch(batch, change_queries, additional_queries_emb)
            print(batch["name"])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)   
            with torch.no_grad():
                    output = model(batch)
            break
        # Visualize the centroids
    visualize_centroids(output, n)  # Replace 'n' with the desired value or variable
    visualize_masks(output, n)
    visualize_masks_binary(output, n)
    
#create a dash app to interactivelly visualize the output of the model given points from the user click
# points are given in the format of [x, y] and in addition user can select the name of the object
# the app should display the image, the masks and the centroids
# the app should be able to handle multiple images, multiple points and multiple names
