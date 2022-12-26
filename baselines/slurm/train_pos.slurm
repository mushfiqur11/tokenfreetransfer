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
source vnv/vnv-trns/bin/activate

python scripts/run_pos.py \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name universal_dependencies \
    --task_name $TASK_NAME \
    --label_column_name upos \
    --per_device_train_batch_size 12 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --max_steps 10 \
    --max_seq_length 256 \
    --output_dir experiments/pos/bert-$TASK_NAME \
    --overwrite_output_dir


python scripts/run_pos.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name universal_dependencies \
    --task_name $TASK_NAME \
    --label_column_name upos \
    --per_device_train_batch_size 12 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --max_steps 10 \
    --max_seq_length 256 \
    --output_dir experiments/pos/mbert-$TASK_NAME \
    --overwrite_output_dir

deactivate