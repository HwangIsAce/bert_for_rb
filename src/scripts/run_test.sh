WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=1 \
python run_mlm.py \
    --config_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./test_output \
    --overwrite_output_dir \

    # --model_name_or_path roberta-base \
# CUDA_VISIBLE_DEVICES=1,2,3 \


