import os
import pickle
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HOME'] = '/ssdstore/azadaia/.cache/huggingface'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from llm2vec import LLM2Vec
l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

additional_queries = ["head", 
                      "neck", 
                      "torso", 
                      "tail", 
                      "wheel", 
                      "hand", 
                      "person", 
                      "plate"]
additional_queries_reps = l2v.encode(additional_queries, batch_size=1)
category_name_to_embedding = dict(zip(additional_queries, additional_queries_reps))
for key in category_name_to_embedding.keys():
    category_name_to_embedding[key] = category_name_to_embedding[key].cpu().numpy()
print(category_name_to_embedding)

with open("/home/azadaia/projects/language_conditioned_oclf/notebooks/additional_queries_emb.pkl", "wb") as f:
    pickle.dump(category_name_to_embedding, f)