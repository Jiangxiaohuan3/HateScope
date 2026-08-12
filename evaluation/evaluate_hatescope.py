#!/usr/bin/env python3
"""End-to-end evaluation for HateScope with vLLM.

Candidate generation:
  - Loads the evaluated model with vLLM.
  - Prompts it to output a strict JSON object:
    {target, argument, group, hateful}

Judge evaluation:
  - Uses the explicitly configured DeepSeek-V4-Flash judge, either through
    vLLM or an OpenAI-compatible endpoint.
  - Uses LLM-as-judge for target identification and argument generation.
  - End-to-end rule: if hateful is wrong, group/target/argument scores are 0.
  - All task metrics are reported on a 0-100 scale.
  - Overall is the unweighted average of four tasks:
    Hate Macro-F1, Group Macro-F1, Target Judge Score, Argument Judge Score.

Example:
  python3 evaluation/evaluate_hatescope.py \
    --model-path /path/to/candidate-model \
    --judge-model-path /path/to/DeepSeek-V4-Flash \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 512

Outputs:
  outputs/<run_name>/
    predictions.jsonl
    metrics.json
    run_config.json
    judge_prompts.md
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "hatescope_1330.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

VALID_GROUPS = [
    "性别歧视",
    "种族歧视",
    "地域歧视",
    "性少数歧视",
    "其他歧视类型",
    "无",
]

CANDIDATE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "argument": {"type": "string"},
        "group": {"type": "string", "enum": VALID_GROUPS},
        "hateful": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["target", "argument", "group", "hateful"],
    "additionalProperties": False,
}

GROUP_ALIASES = {
    "": "无",
    "none": "无",
    "null": "无",
    "non-hate": "无",
    "non_hate": "无",
    "无": "无",
    "没有": "无",
    "其他歧視類型": "其他歧视类型",
    "其他歧视类型": "其他歧视类型",
    "others": "其他歧视类型",
    "other": "其他歧视类型",
    "性別歧視": "性别歧视",
    "性别歧视": "性别歧视",
    "sexism": "性别歧视",
    "種族歧視": "种族歧视",
    "种族歧视": "种族歧视",
    "racism": "种族歧视",
    "地域歧視": "地域歧视",
    "地域歧视": "地域歧视",
    "region": "地域歧视",
    "性少數歧視": "性少数歧视",
    "性少数歧视": "性少数歧视",
    "lgbtq": "性少数歧视",
    "lgbt": "性少数歧视",
}


ARGUMENT_JUDGE_SYSTEM_PROMPT = """你是中文仇恨表达评测专家。你的任务是评价候选模型生成的 argument/explanation 是否正确解释了输入文本为什么构成或不构成歧视/仇恨表达。

你必须严格基于给定 input、gold target、gold group、gold hateful 和 gold argument 评估候选 argument。不要因为候选文本写得流畅就给高分；不要奖励候选中没有被原文支持的臆测；不要因为措辞不同就扣分，只要语义正确即可。

请只输出 JSON，不要输出 Markdown 或额外解释。JSON 格式必须为：
{
  "score": 0,
  "reason": "一句简短中文理由"
}

score 必须是 0 到 100 之间的数字，表示百分制得分。"""


ARGUMENT_JUDGE_USER_TEMPLATE = """请评价候选 argument 的质量。

【输入文本】
{input_text}

【Gold 标注】
- hateful: {gold_hateful}
- target: {gold_target}
- group: {gold_group}
- gold argument: {gold_argument}

【候选模型输出】
- predicted hateful: {pred_hateful}
- predicted target: {pred_target}
- predicted group: {pred_group}
- candidate argument: {pred_argument}

【评分目标】
你只需要评价 candidate argument 本身是否是一个正确、完整、忠实的解释。注意：
1. 如果 gold hateful=1，candidate argument 应该说明：
   - 文本攻击了谁；
   - 使用了什么负面归因、刻板印象、污名化、贬损、嘲讽、敌意或泛化逻辑；
   - 为什么这与 gold group 对应的群体歧视/仇恨相关；
   - 解释必须忠实于原文，不得添加原文不存在的严重指控。
2. 如果 gold hateful=0，candidate argument 应该说明：
   - 为什么文本不构成针对受保护群体的仇恨或歧视；
   - 如果文本只是个体批评、事实讨论、引用/反驳、模糊负面评价、普通争议表达或没有群体攻击，应明确指出；
   - 不应凭空构造攻击对象、群体或歧视理由。
3. 如果 predicted hateful 与 gold hateful 不一致，通常 candidate argument 应给 0 分；只有在候选 argument 明确自我纠正并给出了与 gold 一致的解释时，才可给非常低的部分分。

【评分细则，0-100 分】
- 100：完全正确。忠实于原文，覆盖 target、argument mechanism、group/hateful 判断，解释完整清晰。
- 80：基本正确。核心判断和主要理由正确，但略有遗漏或表述不够具体。
- 60：部分正确。抓住一部分关键原因，但遗漏重要机制、target/group 关联不清，或解释较泛。
- 40：较弱。与 gold 有少量相关性，但核心原因不完整，存在明显泛化、含混或轻微不忠实。
- 20：大多错误。只提到很表面的内容，或 target/group/hateful 解释大多不对，但并非完全无关。
- 0：完全错误、空泛、与原文矛盾、把正例解释成负例/把负例解释成正例、臆造原文没有的信息，或没有可评估内容。

请输出 JSON："""


TARGET_JUDGE_SYSTEM_PROMPT = """你是中文仇恨表达评测专家。你的任务是评价候选模型识别的 target 是否与 gold target 在语义上匹配。

请只输出 JSON，不要输出 Markdown 或额外解释。JSON 格式必须为：
{
  "score": 0,
  "reason": "一句简短中文理由"
}

score 必须是 0 到 100 之间的数字，表示百分制得分。"""


TARGET_JUDGE_USER_TEMPLATE = """请评价候选 target 是否正确。

【输入文本】
{input_text}

【Gold 标注】
- hateful: {gold_hateful}
- gold target: {gold_target}
- gold group: {gold_group}

【候选模型输出】
- predicted hateful: {pred_hateful}
- predicted target: {pred_target}
- predicted group: {pred_group}

【评分细则，0-100 分】
- 100：target 与 gold target 语义等价，或在原文语境中准确指向同一被攻击对象。
- 80：基本正确，但范围略宽或略窄，不影响主要判断。
- 60：部分正确，识别到了相关对象，但遗漏关键限定或包含明显多余对象。
- 40：较弱，只与 gold target 有间接关系，或者范围过宽/过窄导致判断受影响。
- 20：大多错误，仅有很弱相关性。
- 0：完全错误、为空但 gold 有 target、凭空构造 target，或 gold hateful=0 时却构造了明确攻击对象。

如果 gold hateful=0 且 gold target 为空，候选 target 为空、无、没有、未明确攻击对象等，都应给 100 分；如果候选构造了具体群体攻击对象，应给低分。

请输出 JSON："""


@dataclass
class ParsedOutput:
    valid_json: bool
    schema_valid: bool
    label_valid: bool
    target: str
    argument: str
    group: str
    hateful: int | None
    parse_error: str | None = None


def load_vllm():
    try:
        from vllm import LLM, SamplingParams

        return LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "vLLM is required. Install it in the active environment before running this script."
        ) from exc


def load_data(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_group(value: Any) -> str:
    text = normalize_text(value)
    key = text.lower()
    if text in GROUP_ALIASES:
        return GROUP_ALIASES[text]
    if key in GROUP_ALIASES:
        return GROUP_ALIASES[key]
    return text


def normalize_hateful(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        return int(value) if value in (0.0, 1.0) else None
    text = normalize_text(value).lower()
    if text in {"1", "true", "yes", "hate", "hateful", "有", "是", "仇恨", "歧视"}:
        return 1
    if text in {"0", "false", "no", "non-hate", "non_hate", "无", "否", "非仇恨", "不歧视"}:
        return 0
    return None


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = extract_answer_region(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
        except Exception as exc:
            last_error = str(exc)
    json_objects = extract_json_objects(raw)
    if json_objects:
        required_keys = {"target", "argument", "group", "hateful"}
        for obj in reversed(json_objects):
            lower_keys = {str(k).lower() for k in obj}
            if required_keys <= lower_keys:
                group = normalize_group(obj.get("group", obj.get("Group", "")))
                hateful = normalize_hateful(obj.get("hateful", obj.get("Hateful", None)))
                if group in VALID_GROUPS and hateful in (0, 1):
                    return obj, None
        for obj in reversed(json_objects):
            lower_keys = {str(k).lower() for k in obj}
            if required_keys <= lower_keys:
                return obj, None
        return json_objects[-1], None
    return None, last_error if "last_error" in locals() else "no JSON object found"


def extract_answer_region(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.rsplit(marker, 1)[1]
    stripped = text.lstrip()
    if stripped.startswith("<think>") or stripped.startswith("Thinking Process:"):
        return ""
    return text


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    stack = 0
    start: int | None = None
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"' and stack > 0:
            in_string = True
            continue
        if char == "{":
            if stack == 0:
                start = idx
            stack += 1
            continue
        if char == "}" and stack:
            stack -= 1
            if stack == 0 and start is not None:
                candidate = text[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    start = None
                    continue
                if isinstance(obj, dict):
                    objects.append(obj)
                start = None
    return objects


def parse_candidate_output(text: str) -> ParsedOutput:
    obj, error = extract_json_object(text)
    if obj is None:
        return ParsedOutput(False, False, False, "", "", "", None, error)

    target = normalize_text(obj.get("target", obj.get("Target", "")))
    argument = normalize_text(obj.get("argument", obj.get("Argument", "")))
    group = normalize_group(obj.get("group", obj.get("Group", "")))
    hateful = normalize_hateful(obj.get("hateful", obj.get("Hateful", None)))

    required_keys = {"target", "argument", "group", "hateful"}
    lower_keys = {str(k).lower() for k in obj}
    schema_valid = required_keys <= lower_keys
    label_valid = group in VALID_GROUPS and hateful in (0, 1)

    return ParsedOutput(True, schema_valid, label_valid, target, argument, group, hateful)


def parse_judge_score(text: str) -> tuple[float, str, bool]:
    obj, _ = extract_json_object(text)
    if obj is None:
        return 0.0, "judge output is not valid JSON", False
    try:
        score = float(obj.get("score", 0.0))
    except Exception:
        score = 0.0
    if math.isnan(score) or math.isinf(score):
        score = 0.0
    # Judges are instructed to return 0-100. If a judge accidentally returns
    # a 0-1 value, keep backward-compatible behavior by scaling it.
    if 0.0 <= score <= 1.0:
        score *= 100.0
    score = max(0.0, min(100.0, score))
    reason = normalize_text(obj.get("reason", ""))
    if reason.lower().startswith("judge api failed"):
        return score, reason, False
    return score, reason, True


def load_raw_outputs_from_predictions(path: Path, records: list[dict[str, Any]]) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(records):
        raise RuntimeError(
            f"Prediction count mismatch: loaded {len(rows)} rows from {path}, "
            f"but data has {len(records)} records."
        )

    by_id = {
        row.get("id"): row.get("raw_output", "")
        for row in rows
        if row.get("id") is not None
    }
    if all(record.get("id") in by_id for record in records):
        return [str(by_id[record.get("id")] or "") for record in records]
    return [str(row.get("raw_output", "") or "") for row in rows]


def build_candidate_prompt(record: dict[str, Any]) -> str:
    return f"""你是中文仇恨表达分析助手。请阅读输入文本，并输出一个严格 JSON 对象。

任务：
1. 判断文本是否构成针对受保护群体的歧视或仇恨表达。
2. 如果构成，识别攻击对象 target，说明攻击理由 argument，判断群体类别 group。
3. 如果不构成，必须输出 hateful=0、group=\"无\"，并且 target 和 argument 都输出空字符串 \"\"。

Argument 应该包含什么：
- Argument 不是简单摘抄一个词，也不是泛泛地说“有歧视”。
- 如果 hateful=1，argument 需要完整说明文本中用于攻击 target 的负面归因、刻板印象、污名化、贬损、嘲讽、敌意或泛化逻辑，并解释为什么这构成对应 group 的歧视/仇恨。
- Argument 必须忠实于输入文本，不要添加原文没有的事实或指控。
- 如果 hateful=0，不需要 argument，直接输出空字符串。

Target 应该是什么：
- 如果 hateful=1，target 是文本中被攻击、被污名化或被泛化贬损的对象。
- 如果 hateful=0，不需要 target，直接输出空字符串。

允许的 group 只能是：
性别歧视、种族歧视、地域歧视、性少数歧视、其他歧视类型、无

输出格式必须严格为 JSON，不要输出 Markdown，不要输出解释性前后缀：
{{
  "target": "",
  "argument": "",
  "group": "无",
  "hateful": 0
}}

【输入文本】
{record["input"]}

【输出样例，仅用于示范格式和 argument 写法，不是对上面输入文本的答案】
{{
  "target": "香港人",
  "argument": "该文本针对“香港人”进行负面归因或攻击，核心依据是“香港人歧视大陆人,并且对外国人态度谄媚”。这种表述将具体对象与地域歧视相关的负面特征、刻板印象或贬损判断联系起来，并通过泛化、污名化、嘲讽或敌意表达强化对该群体的负面评价，因此构成歧视或仇恨表达。",
  "group": "地域歧视",
  "hateful": 1
}}
"""


def format_chat_prompt(
    llm: Any,
    user_prompt: str,
    system_prompt: str | None = None,
    disable_thinking: bool = False,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        tokenizer = llm.get_tokenizer()
        if getattr(tokenizer, "chat_template", None):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if disable_thinking:
                kwargs["enable_thinking"] = False
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        pass

    if system_prompt:
        return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"
    return f"<user>\n{user_prompt}\n</user>\n<assistant>\n"


def run_generation(
    llm: Any,
    sampling_params: Any,
    prompts: list[str],
    batch_size: int,
) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        generated = llm.generate(batch, sampling_params)
        for item in generated:
            outputs.append(item.outputs[0].text)
    return outputs


def load_generation_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("id") is not None:
                cache[str(item["id"])] = str(item.get("raw_output", "") or "")
    return cache


def append_generation_cache(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_generation_with_cache(
    llm: Any,
    sampling_params: Any,
    records: list[dict[str, Any]],
    prompts: list[str],
    batch_size: int,
    cache_path: Path,
) -> list[str]:
    cache = load_generation_cache(cache_path)
    pending = [
        (idx, record, prompt)
        for idx, (record, prompt) in enumerate(zip(records, prompts))
        if str(record.get("id", idx)) not in cache
    ]
    print(f"Generation cache: {len(cache)}/{len(records)} existing, {len(pending)} pending.", flush=True)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        generated = llm.generate([item[2] for item in batch], sampling_params)
        cache_items = []
        for (idx, record, _), output in zip(batch, generated):
            raw = output.outputs[0].text
            record_id = str(record.get("id", idx))
            cache[record_id] = raw
            cache_items.append(
                {
                    "idx": idx,
                    "id": record.get("id"),
                    "input_hash": record.get("input_hash"),
                    "raw_output": raw,
                }
            )
        append_generation_cache(cache_path, cache_items)
        print(
            f"Generated and cached {min(start + len(batch), len(pending))}/{len(pending)} pending examples.",
            flush=True,
        )

    outputs = []
    missing = []
    for idx, record in enumerate(records):
        record_id = str(record.get("id", idx))
        if record_id not in cache:
            missing.append(record_id)
            outputs.append("")
        else:
            outputs.append(cache[record_id])
    if missing:
        raise RuntimeError(f"Generation cache is missing {len(missing)} records, e.g. {missing[:5]}")
    return outputs


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: float,
        retries: int,
    ) -> None:
        base = api_base.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def generate_one(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error = ""
        for attempt in range(self.retries + 1):
            req = urllib_request.Request(self.endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                return normalize_text(obj["choices"][0]["message"]["content"])
            except urllib_error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body[:500]}"
            except Exception as exc:
                last_error = str(exc)
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8))
        return json.dumps({"score": 0, "reason": f"judge API failed: {last_error}"}, ensure_ascii=False)


def run_openai_judge_generation(
    judge: OpenAICompatibleClient,
    prompts: list[tuple[str, str]],
    workers: int,
) -> list[str]:
    if not prompts:
        return []
    outputs = ["" for _ in prompts]
    max_workers = max(1, min(workers, len(prompts)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(judge.generate_one, system_prompt, user_prompt): idx
            for idx, (system_prompt, user_prompt) in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            outputs[future_to_idx[future]] = future.result()
    return outputs


def load_judge_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("key")
            if key:
                cache[str(key)] = str(item.get("raw_output", "") or "")
    return cache


def append_judge_cache(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_openai_judge_generation_with_cache(
    judge: OpenAICompatibleClient,
    prompts: list[tuple[str, str]],
    keys: list[str],
    workers: int,
    cache_path: Path,
) -> list[str]:
    if not prompts:
        return []
    cache = load_judge_cache(cache_path)
    outputs = [cache.get(key, "") for key in keys]
    pending = [
        (idx, key, prompt)
        for idx, (key, prompt) in enumerate(zip(keys, prompts))
        if key not in cache
    ]
    print(f"Judge cache {cache_path.name}: {len(cache)}/{len(keys)} existing, {len(pending)} pending.", flush=True)
    if not pending:
        return outputs

    max_workers = max(1, min(workers, len(pending)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(judge.generate_one, prompt[0], prompt[1]): (idx, key)
            for idx, key, prompt in pending
        }
        completed = 0
        for future in as_completed(future_to_item):
            idx, key = future_to_item[future]
            raw = future.result()
            outputs[idx] = raw
            append_judge_cache(cache_path, [{"key": key, "raw_output": raw}])
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(f"Judge cached {completed}/{len(pending)} pending outputs for {cache_path.name}.", flush=True)
    return outputs


def binary_macro_f1(y_true: list[int], y_pred: list[int | None]) -> float:
    scores = []
    for label in [0, 1]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def multiclass_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> float:
    return round(value * 100.0, 2)


def pct_mean(values: list[float]) -> float:
    return round(mean(values), 2)


def build_target_judge_prompt(record: dict[str, Any], pred: ParsedOutput) -> str:
    return TARGET_JUDGE_USER_TEMPLATE.format(
        input_text=record["input"],
        gold_hateful=record["hateful"],
        gold_target=record.get("target", ""),
        gold_group=record.get("group", "无"),
        pred_hateful=pred.hateful,
        pred_target=pred.target,
        pred_group=pred.group,
    )


def build_argument_judge_prompt(record: dict[str, Any], pred: ParsedOutput) -> str:
    return ARGUMENT_JUDGE_USER_TEMPLATE.format(
        input_text=record["input"],
        gold_hateful=record["hateful"],
        gold_target=record.get("target", ""),
        gold_group=record.get("group", "无"),
        gold_argument=record.get(
            "argument",
            record.get("argument_refined", record.get("argument_original", "")),
        ),
        pred_hateful=pred.hateful,
        pred_target=pred.target,
        pred_group=pred.group,
        pred_argument=pred.argument,
    )


def safe_model_name(path: str) -> str:
    name = Path(path.rstrip("/")).name or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def judge_cache_key(task: str, record: dict[str, Any], pred: ParsedOutput) -> str:
    payload = {
        "task": task,
        "id": record.get("id"),
        "input_hash": record.get("input_hash"),
        "pred": {
            "target": pred.target,
            "argument": pred.argument,
            "group": pred.group,
            "hateful": pred.hateful,
        },
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{task}:{record.get('id', '')}:{digest}"


def build_candidate_sampling_params(SamplingParams: Any, args: argparse.Namespace) -> Any:
    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.guided_json:
        import inspect

        sampling_signature = inspect.signature(SamplingParams)
        if "structured_outputs" in sampling_signature.parameters:
            try:
                from vllm.sampling_params import StructuredOutputsParams
            except Exception as exc:
                raise RuntimeError("This vLLM installation does not expose StructuredOutputsParams.") from exc
            kwargs["structured_outputs"] = StructuredOutputsParams(
                json=CANDIDATE_JSON_SCHEMA,
                disable_additional_properties=True,
            )
        elif "guided_decoding" in sampling_signature.parameters:
            try:
                from vllm.sampling_params import GuidedDecodingParams
            except Exception as exc:
                raise RuntimeError("This vLLM installation does not expose GuidedDecodingParams.") from exc
            kwargs["guided_decoding"] = GuidedDecodingParams(json=CANDIDATE_JSON_SCHEMA)
        else:
            raise RuntimeError("This vLLM installation does not support JSON guided decoding in SamplingParams.")
    return SamplingParams(**kwargs)


def write_run_config(out_dir: Path, args: argparse.Namespace) -> None:
    run_config = vars(args).copy()
    for key in ("model_api_key", "judge_api_key"):
        if run_config.get(key):
            run_config[key] = "***redacted***"
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_sample_only_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    raw_outputs: list[str],
    parsed_outputs: list[ParsedOutput],
) -> None:
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for record, raw, pred in zip(records, raw_outputs, parsed_outputs):
            item = {
                "id": record.get("id"),
                "input_hash": record.get("input_hash"),
                "input": record["input"],
                "gold": {
                    "target": record.get("target", ""),
                    "argument": record.get("argument", record.get("argument_refined", "")),
                    "group": record.get("group", "无"),
                    "hateful": record.get("hateful"),
                },
                "raw_output": raw,
                "parsed": {
                    "valid_json": pred.valid_json,
                    "schema_valid": pred.schema_valid,
                    "label_valid": pred.label_valid,
                    "parse_error": pred.parse_error,
                    "target": pred.target,
                    "argument": pred.argument,
                    "group": pred.group,
                    "hateful": pred.hateful,
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = {
        "mode": "sample_only",
        "num_examples": len(records),
        "candidate_model": args.model_api_model or args.model_path,
        "format_score": pct(
            (
                mean([float(pred.valid_json) for pred in parsed_outputs])
                + mean([float(pred.schema_valid) for pred in parsed_outputs])
                + mean([float(pred.label_valid) for pred in parsed_outputs])
            )
            / 3.0
        ),
        "valid_json_rate": pct(mean([float(pred.valid_json) for pred in parsed_outputs])),
        "schema_compliance_rate": pct(mean([float(pred.schema_valid) for pred in parsed_outputs])),
        "label_validity_rate": pct(mean([float(pred.label_valid) for pred in parsed_outputs])),
        "pred_hateful_distribution": dict(Counter(str(pred.hateful) for pred in parsed_outputs)),
        "pred_group_distribution": dict(Counter(str(pred.group) for pred in parsed_outputs)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_run_config(out_dir, args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nWrote sample-only outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HateScope with local vLLM models or OpenAI-compatible APIs.")
    parser.add_argument("--model-path", default=None, help="Local candidate model path or HF id.")
    parser.add_argument("--model-api-base", default=None, help="OpenAI-compatible candidate API base URL.")
    parser.add_argument("--model-api-key", default=None, help="Candidate API key. Defaults to MODEL_API_KEY or OPENAI_API_KEY.")
    parser.add_argument("--model-api-model", default=None, help="Candidate model name exposed by the API.")
    parser.add_argument("--model-api-workers", type=int, default=8)
    parser.add_argument("--model-api-timeout", type=float, default=120.0)
    parser.add_argument("--model-api-retries", type=int, default=3)
    parser.add_argument("--judge-model-path", default=None, help="Local DeepSeek-V4-Flash judge path or model id.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--predictions-input", type=Path, default=None, help="Reuse raw_output from an existing predictions.jsonl and skip candidate generation.")
    parser.add_argument("--sample-only", action="store_true", help="Run candidate generation only and write predictions.jsonl without judge evaluation.")
    parser.add_argument("--no-resume", action="store_true", help="Disable generation and judge cache resume files.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=False to chat templates that support it.")
    parser.add_argument("--guided-json", action="store_true", help="Use vLLM guided decoding to constrain candidate outputs to the target JSON schema.")

    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--judge-api-base", default=None, help="OpenAI-compatible judge API base URL.")
    parser.add_argument("--judge-api-key", default=None, help="Judge API key. Defaults to OPENAI_API_KEY or JUDGE_API_KEY.")
    parser.add_argument("--judge-api-model", default=None, help="OpenAI-compatible judge model name.")
    parser.add_argument("--judge-api-workers", type=int, default=8)
    parser.add_argument("--judge-api-timeout", type=float, default=120.0)
    parser.add_argument("--judge-api-retries", type=int, default=3)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--judge-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--judge-dtype", default=None)
    parser.add_argument("--kv-cache-dtype", default=None, help="vLLM kv_cache_dtype, e.g. fp8 for DeepSeek V4.")
    parser.add_argument("--judge-kv-cache-dtype", default=None, help="Judge vLLM kv_cache_dtype. Defaults to --kv-cache-dtype.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--judge-max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--judge-trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--judge-enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--judge-language-model-only", action="store_true")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--judge-enable-expert-parallel", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--judge-max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--judge-max-num-batched-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.sample_only and args.predictions_input is not None:
        raise RuntimeError("--sample-only cannot be combined with --predictions-input.")

    model_api_enabled = bool(args.model_api_base or args.model_api_model)
    if model_api_enabled and not (args.model_api_base and args.model_api_model):
        raise RuntimeError("--model-api-base and --model-api-model must be provided together.")
    if model_api_enabled and args.model_path:
        raise RuntimeError("Choose either --model-path or the candidate API options, not both.")
    if not model_api_enabled and not args.model_path and args.predictions_input is None:
        raise RuntimeError("Set --model-path, or provide both --model-api-base and --model-api-model.")
    model_api_key = args.model_api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if model_api_enabled and not model_api_key:
        raise RuntimeError("Candidate API key is required. Set MODEL_API_KEY/OPENAI_API_KEY or pass --model-api-key.")

    judge_api_enabled = bool(args.judge_api_base or args.judge_api_model)
    if judge_api_enabled and not (args.judge_api_base and args.judge_api_model):
        raise RuntimeError("--judge-api-base and --judge-api-model must be provided together.")
    if not args.sample_only and not judge_api_enabled and not args.judge_model_path:
        raise RuntimeError(
            "Set --judge-model-path for a local DeepSeek-V4-Flash judge, or provide "
            "both --judge-api-base and --judge-api-model."
        )
    judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if judge_api_enabled and not judge_api_key:
        raise RuntimeError("Judge API key is required. Set OPENAI_API_KEY/JUDGE_API_KEY or pass --judge-api-key.")

    judge_model_path = args.judge_model_path or args.model_path
    same_judge = not model_api_enabled and judge_model_path == args.model_path
    judge_model_label = args.judge_api_model if judge_api_enabled else judge_model_path
    candidate_model_label = args.model_api_model or args.model_path or "predictions"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{safe_model_name(candidate_model_label)}_{timestamp}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_data(args.data, args.limit)
    if not records:
        raise RuntimeError(f"No records loaded from {args.data}")

    needs_vllm = (
        args.predictions_input is None and not model_api_enabled
    ) or (
        not args.sample_only and not judge_api_enabled
    )
    LLM = SamplingParams = None
    if needs_vllm:
        LLM, SamplingParams = load_vllm()

    candidate_llm = None
    if args.predictions_input:
        raw_outputs = load_raw_outputs_from_predictions(args.predictions_input, records)
    elif model_api_enabled:
        candidate_api_client = OpenAICompatibleClient(
            api_base=args.model_api_base,
            api_key=model_api_key,
            model=args.model_api_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.model_api_timeout,
            retries=args.model_api_retries,
        )
        candidate_api_prompts = [("", build_candidate_prompt(record)) for record in records]
        if args.no_resume:
            raw_outputs = run_openai_judge_generation(
                candidate_api_client, candidate_api_prompts, args.model_api_workers
            )
        else:
            raw_outputs = run_openai_judge_generation_with_cache(
                candidate_api_client,
                candidate_api_prompts,
                [str(record.get("id", idx)) for idx, record in enumerate(records)],
                args.model_api_workers,
                out_dir / "generation_raw.jsonl",
            )
    else:
        llm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
        }
        if args.max_model_len:
            llm_kwargs["max_model_len"] = args.max_model_len
        if args.kv_cache_dtype:
            llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        if args.trust_remote_code:
            llm_kwargs["trust_remote_code"] = True
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if args.language_model_only:
            llm_kwargs["language_model_only"] = True
        if args.enable_expert_parallel:
            llm_kwargs["enable_expert_parallel"] = True
        if args.max_num_seqs:
            llm_kwargs["max_num_seqs"] = args.max_num_seqs
        if args.max_num_batched_tokens:
            llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

        candidate_llm = LLM(**llm_kwargs)
        candidate_sampling = build_candidate_sampling_params(SamplingParams, args)

        candidate_prompts = [
            format_chat_prompt(candidate_llm, build_candidate_prompt(record), disable_thinking=args.disable_thinking)
            for record in records
        ]
        if args.no_resume:
            raw_outputs = run_generation(candidate_llm, candidate_sampling, candidate_prompts, args.batch_size)
        else:
            raw_outputs = run_generation_with_cache(
                candidate_llm,
                candidate_sampling,
                records,
                candidate_prompts,
                args.batch_size,
                out_dir / "generation_raw.jsonl",
            )
    parsed_outputs = [parse_candidate_output(text) for text in raw_outputs]
    if args.sample_only:
        write_sample_only_outputs(out_dir, args, records, raw_outputs, parsed_outputs)
        return

    if judge_api_enabled:
        judge_llm = None
    elif same_judge:
        if candidate_llm is None:
            raise RuntimeError("Local same-model judge requires candidate generation in this run.")
        judge_llm = candidate_llm
    else:
        if candidate_llm is not None:
            del candidate_llm
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        judge_kwargs = {
            "model": judge_model_path,
            "tensor_parallel_size": args.judge_tensor_parallel_size or args.tensor_parallel_size,
            "dtype": args.judge_dtype or args.dtype,
            "gpu_memory_utilization": args.judge_gpu_memory_utilization or args.gpu_memory_utilization,
            "seed": args.seed,
        }
        if args.judge_max_model_len:
            judge_kwargs["max_model_len"] = args.judge_max_model_len
        judge_kv_cache_dtype = args.judge_kv_cache_dtype or args.kv_cache_dtype
        if judge_kv_cache_dtype:
            judge_kwargs["kv_cache_dtype"] = judge_kv_cache_dtype
        if args.judge_trust_remote_code or args.trust_remote_code:
            judge_kwargs["trust_remote_code"] = True
        if args.judge_enforce_eager or args.enforce_eager:
            judge_kwargs["enforce_eager"] = True
        if args.judge_language_model_only or args.language_model_only:
            judge_kwargs["language_model_only"] = True
        if args.judge_enable_expert_parallel or args.enable_expert_parallel:
            judge_kwargs["enable_expert_parallel"] = True
        judge_max_num_seqs = args.judge_max_num_seqs or args.max_num_seqs
        if judge_max_num_seqs:
            judge_kwargs["max_num_seqs"] = judge_max_num_seqs
        judge_max_num_batched_tokens = args.judge_max_num_batched_tokens or args.max_num_batched_tokens
        if judge_max_num_batched_tokens:
            judge_kwargs["max_num_batched_tokens"] = judge_max_num_batched_tokens
        judge_llm = LLM(**judge_kwargs)

    judge_sampling = None
    judge_api_client = None
    if judge_api_enabled:
        judge_api_client = OpenAICompatibleClient(
            api_base=args.judge_api_base,
            api_key=judge_api_key,
            model=args.judge_api_model,
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            max_tokens=args.judge_max_tokens,
            timeout=args.judge_api_timeout,
            retries=args.judge_api_retries,
        )
    else:
        judge_sampling = SamplingParams(
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            max_tokens=args.judge_max_tokens,
            seed=args.seed,
        )

    target_prompts = []
    argument_prompts = []
    target_judge_indices = []
    argument_judge_indices = []
    target_judge_keys = []
    argument_judge_keys = []
    argument_auto_correct_non_hate_count = 0
    for idx, (record, pred) in enumerate(zip(records, parsed_outputs)):
        if pred.hateful == record["hateful"]:
            target_judge_indices.append(idx)
            target_judge_keys.append(judge_cache_key("target", record, pred))
            if judge_api_enabled:
                target_prompts.append(
                    (TARGET_JUDGE_SYSTEM_PROMPT, build_target_judge_prompt(record, pred))
                )
            else:
                target_prompts.append(
                    format_chat_prompt(
                        judge_llm,
                        build_target_judge_prompt(record, pred),
                        TARGET_JUDGE_SYSTEM_PROMPT,
                        disable_thinking=args.disable_thinking,
                    )
                )
            if int(record["hateful"]) == 1:
                argument_judge_indices.append(idx)
                argument_judge_keys.append(judge_cache_key("argument", record, pred))
                if judge_api_enabled:
                    argument_prompts.append(
                        (ARGUMENT_JUDGE_SYSTEM_PROMPT, build_argument_judge_prompt(record, pred))
                    )
                else:
                    argument_prompts.append(
                        format_chat_prompt(
                        judge_llm,
                        build_argument_judge_prompt(record, pred),
                        ARGUMENT_JUDGE_SYSTEM_PROMPT,
                        disable_thinking=args.disable_thinking,
                    )
                )
            else:
                argument_auto_correct_non_hate_count += 1

    if judge_api_enabled:
        if args.no_resume:
            target_judge_raw = run_openai_judge_generation(judge_api_client, target_prompts, args.judge_api_workers)
            argument_judge_raw = run_openai_judge_generation(judge_api_client, argument_prompts, args.judge_api_workers)
        else:
            target_judge_raw = run_openai_judge_generation_with_cache(
                judge_api_client,
                target_prompts,
                target_judge_keys,
                args.judge_api_workers,
                out_dir / "judge_target_raw.jsonl",
            )
            argument_judge_raw = run_openai_judge_generation_with_cache(
                judge_api_client,
                argument_prompts,
                argument_judge_keys,
                args.judge_api_workers,
                out_dir / "judge_argument_raw.jsonl",
            )
    else:
        target_judge_raw = run_generation(judge_llm, judge_sampling, target_prompts, args.batch_size) if target_prompts else []
        argument_judge_raw = run_generation(judge_llm, judge_sampling, argument_prompts, args.batch_size) if argument_prompts else []

    target_scores = [0.0 for _ in records]
    target_reasons = ["" for _ in records]
    target_judge_valid = [False for _ in records]
    argument_scores = [0.0 for _ in records]
    argument_reasons = ["" for _ in records]
    argument_judge_valid = [False for _ in records]

    for local_idx, record_idx in enumerate(target_judge_indices):
        score, reason, valid = parse_judge_score(target_judge_raw[local_idx])
        target_scores[record_idx] = score
        target_reasons[record_idx] = reason
        target_judge_valid[record_idx] = valid

    for record_idx, (record, pred) in enumerate(zip(records, parsed_outputs)):
        if pred.hateful == record["hateful"] and int(record["hateful"]) == 0:
            argument_scores[record_idx] = 100.0
            argument_reasons[record_idx] = "gold_hateful=0 且模型正确预测为 0，按规则 argument 默认正确。"
            argument_judge_valid[record_idx] = True

    for local_idx, record_idx in enumerate(argument_judge_indices):
        score, reason, valid = parse_judge_score(argument_judge_raw[local_idx])
        argument_scores[record_idx] = score
        argument_reasons[record_idx] = reason
        argument_judge_valid[record_idx] = valid

    y_true_hateful = [int(record["hateful"]) for record in records]
    y_pred_hateful = [pred.hateful if pred.hateful in (0, 1) else None for pred in parsed_outputs]

    hate_correct = [
        int(pred_hateful == gold_hateful)
        for pred_hateful, gold_hateful in zip(y_pred_hateful, y_true_hateful)
    ]

    gold_groups = [record["group"] for record in records]
    pred_groups_e2e = []
    group_correct = []
    for record, pred, h_ok in zip(records, parsed_outputs, hate_correct):
        if not h_ok:
            pred_groups_e2e.append("__wrong__")
            group_correct.append(0)
        else:
            pred_group = pred.group if pred.group in VALID_GROUPS else "__invalid__"
            pred_groups_e2e.append(pred_group)
            group_correct.append(int(pred_group == record["group"]))

    valid_json_rate = mean([float(pred.valid_json) for pred in parsed_outputs])
    schema_rate = mean([float(pred.schema_valid) for pred in parsed_outputs])
    label_validity_rate = mean([float(pred.label_valid) for pred in parsed_outputs])
    format_score = (valid_json_rate + schema_rate + label_validity_rate) / 3.0

    hate_accuracy = mean([float(v) for v in hate_correct])
    hate_macro_f1 = binary_macro_f1(y_true_hateful, y_pred_hateful)
    group_accuracy_e2e = mean([float(v) for v in group_correct])
    group_macro_f1_e2e = multiclass_macro_f1(gold_groups, pred_groups_e2e, VALID_GROUPS)
    target_score_e2e_pct = pct_mean(target_scores)
    argument_score_e2e_pct = pct_mean(argument_scores)

    full_match_threshold = 80.0
    full_match = mean(
        [
            float(
                hate_correct[i]
                and group_correct[i]
                and target_scores[i] >= full_match_threshold
                and argument_scores[i] >= full_match_threshold
            )
            for i in range(len(records))
        ]
    )

    hate_macro_f1_pct = pct(hate_macro_f1)
    group_macro_f1_e2e_pct = pct(group_macro_f1_e2e)
    overall = round(
        (
            hate_macro_f1_pct
            + group_macro_f1_e2e_pct
            + target_score_e2e_pct
            + argument_score_e2e_pct
        )
        / 4.0,
        2,
    )

    metrics = {
        "num_examples": len(records),
        "candidate_model": candidate_model_label,
        "judge_model": judge_model_label,
        "judge_backend": "openai_compatible_api" if judge_api_enabled else "vllm",
        "metric_scale": "0-100 percentage scale",
        "overall": overall,
        "overall_definition": "mean(hate_macro_f1, group_macro_f1_e2e, target_score_e2e, argument_score_e2e)",
        "format_score": pct(format_score),
        "valid_json_rate": pct(valid_json_rate),
        "schema_compliance_rate": pct(schema_rate),
        "label_validity_rate": pct(label_validity_rate),
        "hate_accuracy": pct(hate_accuracy),
        "hate_macro_f1": hate_macro_f1_pct,
        "group_accuracy_e2e": pct(group_accuracy_e2e),
        "group_macro_f1_e2e": group_macro_f1_e2e_pct,
        "target_score_e2e": target_score_e2e_pct,
        "argument_score_e2e": argument_score_e2e_pct,
        "full_match_at_80": pct(full_match),
        "target_judge_coverage_rate": pct(len(target_judge_indices) / len(records)),
        "argument_judge_coverage_rate": pct(len(argument_judge_indices) / len(records)),
        "argument_auto_correct_non_hate_count": argument_auto_correct_non_hate_count,
        "target_judge_valid_rate_on_judged": pct(mean([float(target_judge_valid[i]) for i in target_judge_indices])) if target_judge_indices else 0.0,
        "argument_judge_valid_rate_on_judged": pct(mean([float(argument_judge_valid[i]) for i in argument_judge_indices])) if argument_judge_indices else 0.0,
        "gold_hateful_distribution": dict(Counter(y_true_hateful)),
        "pred_hateful_distribution": dict(Counter(str(v) for v in y_pred_hateful)),
        "gold_group_distribution": dict(Counter(gold_groups)),
        "pred_group_e2e_distribution": dict(Counter(pred_groups_e2e)),
    }

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for idx, (record, raw, pred) in enumerate(zip(records, raw_outputs, parsed_outputs)):
            item = {
                "id": record.get("id"),
                "input_hash": record.get("input_hash"),
                "input": record["input"],
                "gold": {
                    "target": record.get("target", ""),
                    "argument": record.get("argument", record.get("argument_refined", "")),
                    "group": record.get("group", "无"),
                    "hateful": record.get("hateful"),
                },
                "raw_output": raw,
                "parsed": {
                    "valid_json": pred.valid_json,
                    "schema_valid": pred.schema_valid,
                    "label_valid": pred.label_valid,
                    "parse_error": pred.parse_error,
                    "target": pred.target,
                    "argument": pred.argument,
                    "group": pred.group,
                    "hateful": pred.hateful,
                },
                "scores": {
                    "hateful_correct": hate_correct[idx],
                    "hateful_correct_score": 100.0 if hate_correct[idx] else 0.0,
                    "group_correct_e2e": group_correct[idx],
                    "group_correct_e2e_score": 100.0 if group_correct[idx] else 0.0,
                    "target_score_e2e": round(target_scores[idx], 2),
                    "argument_score_e2e": round(argument_scores[idx], 2),
                    "full_match_at_80": int(
                        hate_correct[idx]
                        and group_correct[idx]
                        and target_scores[idx] >= full_match_threshold
                        and argument_scores[idx] >= full_match_threshold
                    ),
                },
                "judge": {
                    "target_reason": target_reasons[idx],
                    "target_valid_json": target_judge_valid[idx],
                    "argument_reason": argument_reasons[idx],
                    "argument_valid_json": argument_judge_valid[idx],
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_run_config(out_dir, args)
    (out_dir / "judge_prompts.md").write_text(
        "# Target Judge System Prompt\n\n"
        + TARGET_JUDGE_SYSTEM_PROMPT
        + "\n\n# Target Judge User Template\n\n"
        + TARGET_JUDGE_USER_TEMPLATE
        + "\n\n# Argument Judge System Prompt\n\n"
        + ARGUMENT_JUDGE_SYSTEM_PROMPT
        + "\n\n# Argument Judge User Template\n\n"
        + ARGUMENT_JUDGE_USER_TEMPLATE
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nWrote evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
