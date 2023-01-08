#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/udp_all.out
##SBATCH --error=../tr_output/udp_all.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.80gb:1              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully


TASK_NAME=$1
eval_step=${2:-'evaluate_pos1'}
pred_lang=${3:-'en_ewt'}

# to see ID and state of GPUs assigned
nvidia-smi



if [[ "$eval_step" = "evaluate_pos1" ]]; then
    pwd
    source vnv/vnv-adp-l/bin/activate

    model_name="bert"
    result_file="experiments/pos/${model_name}-result-${TASK_NAME}.txt"
    rm ${result_file}

    python scripts/run_pos.py \
        --model_name_or_path experiments/pos/${model_name}-$TASK_NAME \
        --do_predict \
        --all_predict true \
        --task_name $TASK_NAME \
        --label_column_name upos \
        --dataset_name universal_dependencies \
        --ds_script_name scripts/universal_dependencies.py \
        --result_file ${result_file} \
        --per_device_train_batch_size 12 \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_seq_length 256 \
        --output_dir experiments/pos/${model_name}-$TASK_NAME \
        --overwrite_output_dir \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/pos/$TASK_NAME

    model_name="mbert"
    result_file="experiments/pos/${model_name}-result-${TASK_NAME}.txt"
    rm ${result_file}
    
    python scripts/run_pos.py \
        --model_name_or_path experiments/pos/${model_name}-$TASK_NAME \
        --do_predict \
        --all_predict true \
        --task_name $TASK_NAME \
        --label_column_name upos \
        --dataset_name universal_dependencies \
        --ds_script_name scripts/universal_dependencies.py \
        --result_file ${result_file} \
        --per_device_train_batch_size 12 \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_seq_length 256 \
        --output_dir experiments/pos/${model_name}-$TASK_NAME \
        --overwrite_output_dir \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/pos/$TASK_NAME

    deactivate
fi

if [[ "$eval_step" = "evaluate_pos2" ]]; then
    pwd
    source vnv/vnv-adp-l/bin/activate

    model_name="bert"
    result_file="experiments/pos/finetuned-${model_name}-result-${TASK_NAME}-${pred_lang}.txt"
    rm ${result_file}

    python scripts/run_pos.py \
        --model_name_or_path experiments/pos/${model_name}-$TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --all_predict false \
        --dataset_name universal_dependencies \
        --dataset_config_name ${pred_lang} \
        --ds_script_name scripts/universal_dependencies.py \
        --result_file ${result_file} \
        --task_name $TASK_NAME \
        --per_device_train_batch_size 12 \
        --label_column_name upos \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_steps 10 \
        --max_seq_length 256 \
        --output_dir experiments/pos/finetuned/${model_name}-${pred_lang} \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/pos/${pred_lang} \
        --overwrite_output_dir

    model_name="mbert"
    result_file="experiments/pos/finetuned-${model_name}-result-${TASK_NAME}-${pred_lang}.txt"
    rm ${result_file}

    python scripts/run_pos.py \
        --model_name_or_path experiments/pos/${model_name}-$TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --all_predict false \
        --dataset_name universal_dependencies \
        --dataset_config_name ${pred_lang} \
        --ds_script_name scripts/universal_dependencies.py \
        --result_file ${result_file} \
        --task_name $TASK_NAME \
        --per_device_train_batch_size 12 \
        --label_column_name upos \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_steps 10 \
        --max_seq_length 256 \
        --output_dir experiments/pos/finetuned/${model_name}-${pred_lang} \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/pos/${pred_lang} \
        --overwrite_output_dir

    deactivate
fi

