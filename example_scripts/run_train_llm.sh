DATASET="opinionqa_individual_val0p05_test0p60_evalpartial_0p40_seed42_topk_3"
MASTER_PORT=29501

torchrun --nnodes=1 \
    --nproc-per-node=2 \
    --master_port=${MASTER_PORT} \
    scripts/llm/run_finetuning.py \
    --enable_fsdp \
    --low_cpu_fsdp \
    --fsdp_config.pure_bf16 \
    --use_peft=true \
    --use_fast_kernels \
    --checkpoint_type StateDictType.FULL_STATE_DICT \
    --peft_method='lora' \
    --use_fp16 \
    --mixed_precision \
    --batch_size_training 32 \
    --val_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --dist_checkpoint_root_folder None \
    --dist_checkpoint_folder None \
    --batching_strategy='padding' \
    --dataset_path ${DATASET} \
    --output_dir outputs/llm_finetuning/${DATASET}/ \
    --name ${DATASET} \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --model_nickname "llama2_7b" \
    --is_chat='true' \
    --lr 2e-4 \
    --seed 42 \
    --num_epochs 3 \
    --weight_decay 0 \
    --loss_function_type ce \
    --which_scheduler cosine \
    --warmup_ratio 0.1 \
    --gamma 0.95 \
    --lora_config.r 8 \
    --lora_config.lora_alpha 32 \
    --wandb_config.project YOUR_PROJECT_NAME \
    --wandb_config.entity YOUR_ENTITY_NAME