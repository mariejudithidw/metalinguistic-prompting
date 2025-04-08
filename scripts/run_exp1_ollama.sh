#!/bin/bash

CORPUS=$1       # e.g., "p18" or "news"
MODEL=$2        # e.g., "llama2"
SAFEMODEL=$3    # e.g., "llama2" for file naming

RESULTDIR="results/exp1_word-prediction"
DATAFILE="datasets/exp1/${CORPUS}/corpus.csv"

mkdir -p $RESULTDIR

for EVAL_TYPE in "direct" "metaQuestionSimple" "metaInstruct" "metaQuestionComplex"; do
    OUTFILE="${RESULTDIR}/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}.json"

    # Uncomment to save full vocab distributions
    # DISTFOLDER="${RESULTDIR}/dists/${CORPUS}_${SAFEMODEL}_${EVAL_TYPE}"
    # mkdir -p $DISTFOLDER

    echo "Running Experiment 1 (word prediction): model = ${MODEL}; eval_type = ${EVAL_TYPE}"

    python run_exp1_word-prediction.py \
        --model $MODEL \
        --model_type "ollama" \
        --eval_type ${EVAL_TYPE} \
        --data_file $DATAFILE \
        --out_file ${OUTFILE}
        # --dist_folder $DISTFOLDER
done
