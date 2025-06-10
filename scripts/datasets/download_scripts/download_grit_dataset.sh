git lfs install

DATA_PATH=${DATA_PATH:=./outputs}
git clone https://huggingface.co/datasets/zzliang/GRIT $DATA_PATH/grit_hf_dataset

pip install img2dataset

mkdir -p $DATA_PATH/grit
img2dataset --url_list $DATA_PATH/grit_hf_dataset --input_format "parquet"\
    --url_col "url" --caption_col "caption" --output_format webdataset \
    --output_folder $DATA_PATH/grit --processes_count 4 --thread_count 64 --image_size 224 \
    --resize_only_if_bigger=True --resize_mode="no" --skip_reencode=True \
    --save_additional_columns '["id","noun_chunks", "width", "height" "ref_exps","clip_similarity_vitb32"]' \
    --enable_wandb False