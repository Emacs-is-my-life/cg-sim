#!/bin/bash
#
# FlexInfer memory sweep. Follows the AGENTS.md output convention:
#   output/<experiment-setup>/sim_results/<run-id>.json
#
# For Python-driven sweeps with summary.csv + analysis tree, prefer:
#   python3 scripts/experiments/sweep_memory.py \
#       examples/run/llamacpp_llama-3-8B_flexinfer.yaml \
#       cpu ram 0 2,3,4,5,6,7,8,9,10 flexinfer-mem-sweep
#
# This shell variant exists for a fine-grained MB-stepping sweep
# without analyses; it still drops JSONs in the conventional path.

INPUT_CFG="examples/run/llamacpp_llama-3-8B_flexinfer.yaml"
EXPERIMENT="flexinfer-mem-sweep-fine"
RESULTS_DIR="output/${EXPERIMENT}/sim_results"
mkdir -p "${RESULTS_DIR}"

START_MB=2048
END_MB=10240
STEP_MB=128

for ((mb=START_MB; mb<=END_MB; mb+=STEP_MB)); do
  kb=$((mb * 1024))
  result="${RESULTS_DIR}/${mb}MB.json"

  echo "Running FlexInfer with Memory: ${mb} MB → ${result}"
  python3 main.py \
    -i "$INPUT_CFG" \
    logger.args.result_path="${result}" \
    hardware.memory.0.args.memory_size_KB="${kb}"
done
