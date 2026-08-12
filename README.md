# HateScope

<p align="center">
  <img src="assets/hatescope-framework.png" alt="HateScope framework" width="920">
</p>

HateScope benchmarks long-form Chinese hate-speech understanding. It evaluates four capabilities jointly: hatefulness detection, discrimination-type classification, target recognition, and attribution-rationale generation.

## 📦 Repository

```text
data/                         Refined evaluation data and schema
evaluation/evaluate_hatescope.py
                              vLLM inference, parsing, metrics, and judging
scripts/run_vllm.sh           Single-node launcher
scripts/run_slurm.sbatch      Generic 8-GPU Slurm launcher
```

The released `data/hatescope_refined.jsonl` is the 1,931-record deduplicated refinement pool found with the experiment code: 884 hateful and 1,047 non-hateful examples. The paper reports a 1,330-example benchmark after quality review and non-hate resampling (884 hateful and 446 non-hateful). The final 1,330-record selection manifest was not present in the archived experiment directory, so this repository does not claim that the pool is the exact paper split.

## 🚀 Evaluation

Install vLLM in a CUDA environment, then set local model paths:

```bash
export MODEL_PATH=/path/to/candidate-model
export JUDGE_MODEL_PATH=/path/to/DeepSeek-V4-Flash
bash scripts/run_vllm.sh --run-name my-model
```

Candidate generation uses vLLM with deterministic decoding by default (`temperature=0`, `top_p=1`). The script writes predictions, metrics, the redacted run configuration, and the fixed judge prompts under `outputs/`; generated results and logs are ignored by Git.

## ⚖️ LLM as Judge

Following the paper, DeepSeek-V4-Flash should be used as the fixed judge for target recognition and attribution. Hatefulness and discrimination type use macro-F1. Target and attribution use a fixed 0-100 rubric embedded in the evaluation script. If hatefulness is wrong, all downstream scores are zero; the overall score is the arithmetic mean of the four task scores.

The judge can run locally with vLLM through `--judge-model-path`, or through an OpenAI-compatible service:

```bash
export OPENAI_API_KEY=your_runtime_secret
python evaluation/evaluate_hatescope.py \
  --model-path "$MODEL_PATH" \
  --judge-api-base "$JUDGE_API_BASE" \
  --judge-api-model "$JUDGE_API_MODEL"
```

Keep credentials in the runtime environment or a secret manager. Do not place keys or private endpoints in repository files or Slurm scripts.

## 🧾 Output schema

Each candidate must return one JSON object:

```json
{"target": "", "argument": "", "group": "无", "hateful": 0}
```

Allowed groups are `性别歧视`, `种族歧视`, `地域歧视`, `性少数歧视`, `其他歧视类型`, and `无`.

## Ethics

The dataset contains harmful language for safety research. It should be handled as a diagnostic benchmark, not as a moderation policy or a source of deployable rules.
