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

# to see ID and state of GPUs assigned
nvidia-smi

pwd
source vnv/vnv-adp-l/bin/activate


rm experiments/pos/bert-result-${TASK_NAME}.txt
python scripts/run_pos.py \
    --model_name_or_path experiments/pos/bert-$TASK_NAME \
    --do_predict \
    --all_predict true \
    --task_name $TASK_NAME \
    --label_column_name upos \
    --dataset_name universal_dependencies \
    --ds_script_name scripts/universal_dependencies.py \
    --result_file experiments/pos/bert-result-${TASK_NAME}.txt \
    --per_device_train_batch_size 12 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --output_dir experiments/pos/bert-$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/pos/$TASK_NAME

rm experiments/pos/mbert-result-${TASK_NAME}.txt
python scripts/run_pos.py \
    --model_name_or_path experiments/pos/mbert-$TASK_NAME \
    --do_predict \
    --all_predict true \
    --task_name $TASK_NAME \
    --label_column_name upos \
    --dataset_name universal_dependencies \
    --ds_script_name scripts/universal_dependencies.py \
    --result_file experiments/pos/mbert-result-${TASK_NAME}.txt \
    --per_device_train_batch_size 12 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --output_dir experiments/pos/mbert-$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/pos/$TASK_NAME

deactivate

