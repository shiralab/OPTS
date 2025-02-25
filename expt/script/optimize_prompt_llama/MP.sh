#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

PROMPT_DESIGNING_LLM=gpt-4o-mini
TASK_SOLVING_LLM=llama3-8B-Instruct
METHOD=MP
DATASET_NAME=BBH

for TASK_NAME in boolean_expressions causal_judgement date_understanding disambiguation_qa dyck_languages formal_fallacies geometric_shapes hyperbaton logical_deduction_five_objects logical_deduction_seven_objects logical_deduction_three_objects movie_recommendation multistep_arithmetic_two navigate object_counting penguins_in_a_table reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks sports_understanding temporal_sequences tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects web_of_lies word_sorting
do
    python ../main_test.py \
        --config_file ./setting/${PROMPT_DESIGNING_LLM}_${TASK_SOLVING_LLM}/${DATASET_NAME}/${METHOD}.json \
        --test_file ../dataset/${DATASET_NAME}/${TASK_NAME}/test.json \
        --final_prompt_file ../prompt/${DATASET_NAME}/original_prompts/${TASK_NAME}.txt \
        --prompt_template_file ../prompt_template/${DATASET_NAME}/template_cot/${TASK_NAME}.txt \
        --output_folder ./outputs/${METHOD}/${PROMPT_DESIGNING_LLM}_${TASK_SOLVING_LLM}/${DATASET_NAME}/${TASK_NAME}
done