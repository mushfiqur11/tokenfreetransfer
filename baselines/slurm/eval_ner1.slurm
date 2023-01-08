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
eval_step=$2
pred_lang=${3:-'en'}
# to see ID and state of GPUs assigned
nvidia-smi


if [[ "$eval_step" = "evaluate_ner1" ]]; then
    pwd
    source vnv/vnv-adp-l/bin/activate

    model_name="bert"
    rm experiments/ner/${model_name}-result-${TASK_NAME}.txt

    python scripts/run_ner.py \
        --model_name_or_path experiments/ner/${model_name}-$TASK_NAME \
        --do_predict \
        --all_predict true \
        --dataset_name wikiann \
        --ds_script_name wikiann \
        --result_file experiments/ner/${model_name}-result-${TASK_NAME}.txt \
        --lang_config ../metadata/ner_metadata.json \
        --task_name $TASK_NAME \
        --per_device_train_batch_size 12 \
        --label_column_name ner_tags \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_seq_length 256 \
        --output_dir experiments/ner/${model_name}-$TASK_NAME \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/ner/$TASK_NAME

    model_name="mbert"
    rm experiments/ner/${model_name}-result-${TASK_NAME}.txt
    
    python scripts/run_ner.py \
        --model_name_or_path experiments/ner/${model_name}-$TASK_NAME \
        --do_predict \
        --all_predict true \
        --dataset_name wikiann \
        --ds_script_name wikiann \
        --result_file experiments/ner/${model_name}-result-${TASK_NAME}.txt \
        --lang_config ../metadata/ner_metadata.json \
        --task_name $TASK_NAME \
        --per_device_train_batch_size 12 \
        --label_column_name ner_tags \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_seq_length 256 \
        --output_dir experiments/ner/${model_name}-$TASK_NAME \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/ner/$TASK_NAME
    deactivate
fi

if [[ "$eval_step" = "evaluate_ner2" ]]; then
    pwd
    source vnv/vnv-adp-l/bin/activate

    model_name="bert"
    rm experiments/ner/finetuned-${model_name}-result-${TASK_NAME}-${pred_lang}.txt

    python scripts/run_ner.py \
        --model_name_or_path experiments/ner/${model_name}-$TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --all_predict false \
        --dataset_name wikiann \
        --dataset_config_name ${pred_lang} \
        --ds_script_name wikiann \
        --result_file experiments/ner/finetuned-${model_name}-result-${TASK_NAME}-${pred_lang}.txt \
        --lang_config ../metadata/ner_metadata.json \
        --task_name $TASK_NAME \
        --per_device_train_batch_size 12 \
        --label_column_name ner_tags \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_steps 10 \
        --max_seq_length 256 \
        --output_dir experiments/ner/finetuned/${model_name}-${pred_lang} \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/ner/${pred_lang} \
        --overwrite_output_dir


    model_name="mbert"
    rm experiments/ner/finetuned-${model_name}-result-${TASK_NAME}-${pred_lang}.txt

    python scripts/run_ner.py \
        --model_name_or_path experiments/ner/${model_name}-$TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --all_predict false \
        --dataset_name wikiann \
        --dataset_config_name ${pred_lang} \
        --ds_script_name wikiann \
        --result_file experiments/ner/finetuned-${model_name}-result-${TASK_NAME}-${pred_lang}.txt \
        --lang_config ../metadata/ner_metadata.json \
        --task_name $TASK_NAME \
        --per_device_train_batch_size 12 \
        --label_column_name ner_tags \
        --learning_rate 1e-4 \
        --num_train_epochs 10 \
        --max_steps 10 \
        --max_seq_length 256 \
        --output_dir experiments/ner/finetuned/${model_name}-${pred_lang} \
        --cache_dir /scratch/ffaisal/hug_cache/tokens/datasets/ner/${pred_lang} \
        --overwrite_output_dir
    deactivate
fi

