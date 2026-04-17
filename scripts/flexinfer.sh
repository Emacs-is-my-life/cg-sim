#!/bin/bash

INPUT_CFG="examples/llama3-flexinfer/input.yaml"
mkdir -p tmp/results

START_MB=2048
END_MB=10240
STEP_MB=128

for ((mb=START_MB; mb<=END_MB; mb+=STEP_MB)); do
  kb=$((mb * 1024))
  out="tmp/results/flexinfer_mem_${mb}MB.json"

  echo "Running FlexInfer with Memory: ${mb} MB"
  python main.py \
    -i "$INPUT_CFG" \
	logger.args.output_path="${out}" \
	hardware.memory.0.args.memory_size_KB="${kb}"
done
