WANDB_PROJECT=ingt-v1 \
WANDB_MODE=online \
CUDA_VISIBLE_DEVICES=0,1 \
python /home/donghee/projects/mlm2/src/run_mlm.py \
    --run_name v1-ingr-only \
    --model_type bert \
    --tokenizer_name bert-base-uncased \
    --max_seq_length 10 \
    --train_file /disk1/data/ing_mlm_data/processed/v1_ing_only/train.txt \
    --validation_file /disk1/data/ing_mlm_data/processed/v1_ing_only/val.txt \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --do_train \
    --do_eval \
    --output_dir ./checkpoints/v1-ingr-only \
    --overwrite_output_dir \
    --logging_steps 100 \
    --log_level info \
    --evaluation_strategy steps \
    --num_train_epochs 100 \
    --save_steps 5000 \
    --line_by_line true \
    --pad_to_max_length true \
    --config_overrides num_attention_heads=3,num_hidden_layers=3 \
    
    # --max_steps=200 \
    # --learning_rate



