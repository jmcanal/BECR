#! /usr/bin/env bash

# Set this to where you extracted the downloaded file
export DATA_PATH=./bin/data/test_tweet/

export VOCAB_SOURCE=${DATA_PATH}/train/vocab.sources.txt
export VOCAB_TARGET=${DATA_PATH}/train/vocab.targets.txt
export TRAIN_SOURCES=${DATA_PATH}/train/sources.txt
export TRAIN_TARGETS=${DATA_PATH}/train/targets.txt
export DEV_SOURCES=${DATA_PATH}/dev/sources.txt
export DEV_TARGETS=${DATA_PATH}/dev/targets.txt

export DEV_TARGETS_REF=${DATA_PATH}/dev/targets.txt
export TRAIN_STEPS=1000

export MODEL_DIR=./model
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

tensorboard --logdir $MODEL_DIR

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt

./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt > ${PRED_DIR}/bleu.out