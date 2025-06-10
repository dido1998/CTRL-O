import pickle

import torch
from llm2vec import LLM2Vec
from pycocotools.coco import COCO

if __name__ == "__main__":
    # Initialize the COCO API for instance annotations
    coco = COCO(
        "/home/mila/a/aniket.didolkar/scratch/language_conditioned_oclf/scripts/datasets/outputs/coco/annotations/instances_train2017.json"
    )

    # Get all category IDs
    category_ids = coco.getCatIds()

    # Get all category names
    categories = coco.loadCats(category_ids)
    category_names = [category["name"] for category in categories]

    coco = COCO(
        "/home/mila/a/aniket.didolkar/scratch/language_conditioned_oclf/scripts/datasets/outputs/coco/annotations/instances_val2017.json"
    )
    category_ids = coco.getCatIds()

    # Get all category names
    categories = coco.loadCats(category_ids)

    for category in categories:
        if category["name"] not in category_names:
            category_names.append(category["name"])

    coco = COCO(
        "/home/mila/a/aniket.didolkar/scratch/language_conditioned_oclf/scripts/datasets/outputs/coco/annotations/stuff_val2017.json"
    )
    category_ids = coco.getCatIds()

    # Get all category names
    categories = coco.loadCats(category_ids)

    for category in categories:
        if category["name"] not in category_names:
            category_names.append(category["name"])

    coco = COCO(
        "/home/mila/a/aniket.didolkar/scratch/language_conditioned_oclf/scripts/datasets/outputs/coco/annotations/stuff_train2017.json"
    )
    category_ids = coco.getCatIds()

    # Get all category names
    categories = coco.loadCats(category_ids)

    for category in categories:
        if category["name"] not in category_names:
            category_names.append(category["name"])

    print("COCO Categories:")
    print(category_names)

    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    queries = category_names
    q_reps = l2v.encode(queries)

    category_name_to_embedding = dict(zip(queries, q_reps))
    for key in category_name_to_embedding.keys():
        category_name_to_embedding[key] = category_name_to_embedding[key].cpu().numpy()
    print(category_name_to_embedding)

    with open(
        "/home/mila/a/aniket.didolkar/scratch/language_conditioned_oclf/scripts/datasets/outputs/coco/category_name_to_llam3_emb.pkl",
        "wb",
    ) as f:
        pickle.dump(category_name_to_embedding, f)
