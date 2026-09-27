import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from tqdm import tqdm


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def flatten_group_records(records):
    flat = []
    for item in records:
        pairs = item.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            flat.append(item)
            continue

        for pair_position, pair in enumerate(pairs):
            row = dict(pair)
            row["group"] = item.get("group")
            row["group_score"] = item.get("group_score")
            row["group_pair_position"] = pair_position
            row["domain"] = row.get("domain", item.get("domain"))
            row["prompt_index"] = row.get("prompt_index", item.get("prompt_index"))
            row["prompt"] = row.get("prompt", item.get("prompt", ""))
            row["pair_count"] = item.get("pair_count")
            row["selection_score_max"] = item.get("selection_score_max")
            row["selection_score_mean"] = item.get("selection_score_mean")
            row["mahalanobis_ood_score_max"] = item.get("mahalanobis_ood_score_max")
            row["mahalanobis_ood_score_mean"] = item.get("mahalanobis_ood_score_mean")
            row["margin_risk_max"] = item.get("margin_risk_max")
            row["margin_risk_mean"] = item.get("margin_risk_mean")
            flat.append(row)
    return flat


def write_jsonl(path, records):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_done(path):
    if not path or not os.path.exists(path):
        return {}, []
    done = {}
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = make_key(item)
            done[key] = item
            rows.append(item)
    return done, rows


def make_key(item):
    return f"{item.get('prompt_index')}::{item.get('pair_index')}"


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_winner(value):
    value = str(value).strip().lower()
    if value in {"1", "assistant_1", "assistant 1", "a", "assistant_a", "assistant a"}:
        return "assistant_1"
    if value in {"2", "assistant_2", "assistant 2", "b", "assistant_b", "assistant b"}:
        return "assistant_2"
    if value in {"0", "tie", "same", "equal", "both", "unknown", "cannot_decide"}:
        return "tie"
    return value


def is_false_value(value):
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    return str(value).strip().lower() in {"false", "0", "no"}


def build_prompt(item):
    return f"""You are an impartial preference judge. Compare two assistant responses to the same user prompt.

Judge primarily by correctness and direct usefulness. Also consider reasoning quality, coherence, appropriate detail, and safety. Do not prefer an answer only because it is longer. Do not let response order affect your judgment.

You must choose one of the two responses. Do not output tie, equal, unknown, or cannot_decide.

Return only valid JSON with this schema:
{{
  "winner": "assistant_1" | "assistant_2",
  "reason": "A concise explanation of the decisive differences.",
  "confidence": 0.0 to 1.0
}}

[User Prompt]
{item.get("prompt", "")}

[Assistant 1]
{item.get("assistant_1", "")}

[Assistant 2]
{item.get("assistant_2", "")}
"""


def extract_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def judge_one(item, args):
    client_args = {
        "api_key": os.environ.get(args.api_key_env, args.api_key),
        "http_client": httpx.Client(trust_env=args.trust_env_proxy),
    }
    base_url = os.environ.get(args.base_url_env, args.base_url)
    if base_url:
        client_args["base_url"] = base_url
    client = OpenAI(**client_args)
    prompt = build_prompt(item)
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            request = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": "You are a careful impartial evaluator."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": args.temperature,
            }
            normalized_model = args.model.lower().replace("-", "").replace("_", "")
            if normalized_model.startswith("gpt5"):
                request["max_completion_tokens"] = args.max_tokens
            else:
                request["max_tokens"] = args.max_tokens
            response = client.chat.completions.create(
                **request,
            )
            content = response.choices[0].message.content
            parsed = extract_json(content)
            if parsed is None:
                parsed = {
                    "winner": "parse_error",
                    "reason": content,
                    "confidence": None,
                }
            winner = normalize_winner(parsed.get("winner", "parse_error"))
            if winner not in {"assistant_1", "assistant_2"}:
                winner = "parse_error"
            # Preserve acquisition metadata and the exact pair seen by the judge.
            result = dict(item)
            result.update({
                "router_winner": winner,
                "router_reason": parsed.get("reason", ""),
                "router_confidence": parsed.get("confidence"),
                "router_raw_response": content,
                "router_usage": extract_usage(response),
                "judge_model": args.model,
            })
            return result
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    result = dict(item)
    result["router_winner"] = "error"
    result["router_reason"] = str(last_error)
    result["judge_model"] = args.model
    return result


def main():
    parser = argparse.ArgumentParser(description="Judge selected response pairs with an OpenAI-compatible API.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--api_key_env", default="JUDGE_API_KEY")
    parser.add_argument("--api_key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--base_url_env", default="JUDGE_BASE_URL")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--trust_env_proxy", action="store_true", default=False)
    parser.add_argument("--only_incorrect", action="store_true", default=False)
    parser.add_argument("--only_router_winners", default="")
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    if not args.model:
        raise ValueError("Set JUDGE_MODEL or pass --model.")

    records = flatten_group_records(read_jsonl(args.input_path))
    if args.only_incorrect:
        records = [item for item in records if is_false_value(item.get("is_correct"))]
    if args.only_router_winners:
        wanted_winners = {
            normalize_winner(winner)
            for winner in args.only_router_winners.split(",")
            if winner.strip()
        }
        records = [
            item
            for item in records
            if normalize_winner(item.get("router_winner", "")) in wanted_winners
        ]
    if args.limit > 0:
        records = records[: args.limit]

    done = {}
    results = []
    if args.resume:
        done, results = load_done(args.output_path)
        records = [item for item in records if make_key(item) not in done]

    if not args.api_key:
        raise EnvironmentError(f"Missing API key. Set {args.api_key_env} or pass --api_key.")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(judge_one, item, args) for item in records]
        with open(args.output_path, "a" if args.resume else "w", encoding="utf-8") as handle:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Router judging selected samples"):
                result = future.result()
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

    total = len(results)
    counts = {}
    for item in results:
        winner = item.get("router_winner", "unknown")
        counts[winner] = counts.get(winner, 0) + 1
    print(f"Saved {total} router judgements to {args.output_path}")
    print("Winner counts:")
    for winner, count in sorted(counts.items()):
        print(f"  {winner}: {count}")


if __name__ == "__main__":
    main()
