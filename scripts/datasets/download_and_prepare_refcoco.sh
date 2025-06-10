python ./download_scripts/download_refcoco.py

DATA_PATH=${DATA_PATH:=./outputs}
OUTPUT_PATH=${OUTPUT_PATH:=./data}
COCO_SEED=23894734


mkdir -p $OUTPUT_PATH/refcoco/train


poetry run python conversion_scripts/convert_cocoref_resize.py $DATA_PATH/refcoco_raw/images/train2014 $OUTPUT_PATH/refcocog/val --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --seed 23894734 --split val --instance $DATA_PATH/coco/annotations/instances_train2014.json --caption $DATA_PATH/coco/annotations/captions_train2014.json --is_refcoco --dataset refcocog --download_dir $DATA_PATH/refcoco_raw


poetry run python conversion_scripts/convert_cocoref_resize.py $DATA_PATH/refcoco_raw/images/train2014 $OUTPUT_PATH/refcocog/train --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --seed 23894734 --split train --instance $DATA_PATH/coco/annotations/instances_train2014.json --caption $DATA_PATH/coco/annotations/captions_train2014.json --is_refcoco --dataset refcocog --download_dir $DATA_PATH/refcoco_raw
