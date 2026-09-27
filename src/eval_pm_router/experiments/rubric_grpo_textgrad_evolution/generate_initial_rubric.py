#!/usr/bin/env python3
import argparse
import json
import os
import random
import tempfile


DOMAINS = ["chat", "code", "math", "safety-refuse", "safety-response"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def build_examples_by_domain(rows, per_domain, seed):
    by_domain = {domain: [] for domain in DOMAINS}
    for row in rows:
        winner = row.get("router_winner")
        domain = row.get("domain")
        if winner not in {"assistant_1", "assistant_2"}:
            continue
        if domain not in by_domain:
            continue
        by_domain[domain].append(row)
    missing = [domain for domain, items in by_domain.items() if not items]
    if missing:
        raise ValueError(f"No valid judge-labelled examples for domains: {', '.join(missing)}")

    rng = random.Random(seed)
    examples_by_domain = {}
    for domain in DOMAINS:
        items = list(by_domain[domain])
        rng.shuffle(items)
        examples_by_domain[domain] = []
        for row in items[:per_domain]:
            examples_by_domain[domain].append({
                "domain": domain,
                "prompt": truncate(row.get("prompt"), 700),
                "assistant_1": truncate(row.get("assistant_1"), 900),
                "assistant_2": truncate(row.get("assistant_2"), 900),
                "judge_winner": row["router_winner"],
                "judge_reason": truncate(row.get("router_reason"), 500),
            })
    return examples_by_domain


def flatten_examples(examples_by_domain):
    examples = []
    for domain in DOMAINS:
        examples.extend(examples_by_domain[domain])
    return examples


def make_batches(examples_by_domain, batch_per_domain):
    max_count = max(len(examples_by_domain[domain]) for domain in DOMAINS)
    batches = []
    for start in range(0, max_count, batch_per_domain):
        batch = []
        for domain in DOMAINS:
            batch.extend(examples_by_domain[domain][start:start + batch_per_domain])
        if batch:
            batches.append(batch)
    return batches


def split_batch(examples):
    middle = max(1, len(examples) // 2)
    return examples[:middle], examples[middle:]


def build_prompt(examples, previous_general="", previous_domain=""):
    payload = json.dumps(examples, ensure_ascii=False, indent=2)
    domains = "\n".join(
        f"## {domain}\n"
        f"- Prefer the response that [{domain}-specific pairwise scoring criterion].\n"
        f"- Penalize the response that [{domain}-specific pairwise scoring criterion].\n"
        f"- Discount [{domain}-specific misleading or non-decisive feature].\n"
        f"- Break ties by [{domain}-specific tie-break criterion]."
        for domain in DOMAINS
    )
    previous = ""
    if previous_general or previous_domain:
        previous = f"""[Current Rubric Draft]
# General Updated Rubric
{previous_general}
# Domain-Specific Rubric
{previous_domain}

Use the current draft as the starting point. Analyze the new strong-judge reasons carefully, identify what pairwise judgment principle they reveal, and make an incremental improvement only where the new batch gives useful evidence. Revisit the general rubric first: keep it as a real cross-domain preference hierarchy rather than a short generic summary. Preserve useful rules, merge duplicates, delete rules that ask the model to explain itself or generate an answer, and keep the rubric concise but sufficiently detailed.
"""
    return f"""Write an initial rubric to be prepended to a downstream preference reward model that compares two completed assistant responses.

You are given selected response pairs and a strong judge's binary judgements. Use only the supplied winner and reason fields as supervision. Carefully analyze why the judge preferred one response over the other, then distill reusable scoring principles for future preference comparisons. Do not mention the judge, hidden benchmark answers, ground-truth labels, or difficulty labels in the rubric.

The rubric will be part of the reward model input, not a chat instruction for producing an explanation. Therefore:
- Write pairwise scoring criteria only, as concise as possible while still detailed enough to compare two existing responses.
- Use preference-rubric language, not answer-generation language.
- Do not ask the model to output a winner, JSON, confidence, or reason.
- Do not include meta-instructions such as "explain your decision".
- Use concise bullet points.
- Generalize beyond these examples.
- Prefer principles that are repeatedly supported by judge reasons over one-off details.
- When updating an existing draft, make incremental changes rather than rewriting from scratch.
- Avoid answer-order bias.
- Do not reward verbosity, polish, or formatting by itself.
- Put only domain-agnostic principles in General Updated Rubric, but make it a complete cross-domain decision hierarchy.
- Put domain-specific principles only in the matching domain subsection.
- Do not place code, math, chat, or safety-specific rules in the general section.
- General Updated Rubric should cover decisive correctness errors, instruction following, evidence and verifiability, completeness, safety, clarity, bias controls, and tie-breaks.
- Every bullet must begin with one of these preference verbs: Prefer, Penalize, Discount, Treat, Break ties, Prioritize, Deprioritize, Rank, Compare, Reward.
- Do not begin bullets with answer-generation verbs such as Provide, Use, Ensure, Include, Offer, Respond, Address, Maintain, Demonstrate, Show, Write, Present, Verify, Follow, Redirect, Refrain, Encourage, Document, Handle, Adhere, or Focus.

{previous}
[Selected Examples]
{payload}

Return only the rubric in this exact markdown structure. Keep the full rubric under 900 words. Do not add any extra heading, preface, code fence, process note, or case-specific note.

# General Updated Rubric
- Prefer the response that [domain-agnostic pairwise scoring criterion].
- Penalize the response that [domain-agnostic pairwise scoring criterion].
- Discount [domain-agnostic non-decisive feature or bias].
- Treat [domain-agnostic tradeoff or secondary criterion].
- Break ties by [domain-agnostic tie-break criterion].
- Prioritize [domain-agnostic primary comparison criterion].
# Domain-Specific Rubric
{domains}
"""


def parse_rubric(markdown):
    from generate_candidates import parse_rubric as parse_candidate_rubric
    from generate_candidates import validate_preference_wording
    general, domain = parse_candidate_rubric(markdown)
    validate_preference_wording(general, domain, DOMAINS)
    return general, domain


def save_failed_attempt(output_dir, batch_index, attempt, raw_text, reason):
    failed_dir = os.path.join(output_dir, "failed_batches")
    os.makedirs(failed_dir, exist_ok=True)
    attempt_dir = tempfile.mkdtemp(prefix=f"batch_{batch_index:02d}_attempt_{attempt}_", dir=failed_dir)
    with open(os.path.join(attempt_dir, "raw_response.md"), "w", encoding="utf-8") as handle:
        handle.write(raw_text.rstrip() + "\n")
    with open(os.path.join(attempt_dir, "reason.txt"), "w", encoding="utf-8") as handle:
        handle.write(reason.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--examples_per_domain", type=int, default=6)
    parser.add_argument(
        "--batch_examples_per_domain",
        type=int,
        default=0,
        help="If positive, update the rubric in batches with this many examples per domain.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_tokens", type=int, default=24000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    if args.examples_per_domain < 1:
        parser.error("examples_per_domain must be positive")
    if args.batch_examples_per_domain < 0:
        parser.error("batch_examples_per_domain must be non-negative")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    rows = read_jsonl(args.input)
    examples_by_domain = build_examples_by_domain(rows, args.examples_per_domain, args.seed)
    all_examples = flatten_examples(examples_by_domain)
    batch_size = args.batch_examples_per_domain or args.examples_per_domain
    batches = make_batches(examples_by_domain, batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    general = ""
    domain = ""
    text = ""
    index = 0
    while index < len(batches):
        examples = batches[index]
        batch_number = index + 1
        retry_note = ""
        temperature = args.temperature
        for attempt in range(1, 6):
            prompt = build_prompt(examples, general, domain)
            if retry_note:
                prompt += (
                    "\n\n[Format repair instruction]\n"
                    f"{retry_note}\n"
                    "Return the complete rubric with exactly these headings: "
                    "# General Updated Rubric, # Domain-Specific Rubric, "
                    "## chat, ## code, ## math, ## safety-refuse, ## safety-response."
                )
            messages = [
                {"role": "system", "content": "You write pairwise preference-rubric criteria for a reward model that compares two completed responses. Always return every required markdown heading."},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt", truncation=False).to(model.device)
            if inputs["input_ids"].shape[1] > args.max_input_tokens:
                if len(examples) <= 1:
                    raise ValueError(
                        "Initial-rubric prompt exceeds max_input_tokens even for a single example; "
                        "lower answer truncation limits in generate_initial_rubric.py"
                    )
                left, right = split_batch(examples)
                batches[index:index + 1] = [left, right]
                print(
                    f"Split initial rubric batch {batch_number}: "
                    f"{inputs['input_ids'].shape[1]} tokens > {args.max_input_tokens}; "
                    f"new batch sizes are {len(left)} and {len(right)}"
                )
                break

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=args.top_p,
                    top_k=0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            try:
                if generated.shape[0] >= args.max_new_tokens:
                    raise ValueError(f"output reached max_new_tokens={args.max_new_tokens}; may be truncated")
                general, domain = parse_rubric(text)
                break
            except ValueError as error:
                reason = str(error)
                save_failed_attempt(args.output_dir, batch_number, attempt, text, reason)
                retry_note = (
                    f"Previous attempt was rejected: {reason}. The next answer must include all required sections, "
                    "even if brief, and include no extra headings, preface, postscript, placeholders, or process notes. "
                    "Every bullet must start with Prefer, Penalize, Discount, Treat, Break ties, Prioritize, "
                    "Deprioritize, Rank, Compare, or Reward."
                )
                temperature = max(0.3, temperature - 0.15)
                print(f"Retrying initial rubric batch {batch_number}/{len(batches)} attempt {attempt}: {reason}")
        else:
            raise ValueError(
                f"Failed to generate a valid initial rubric for batch {batch_number} after 5 attempts; "
                f"inspect {os.path.join(args.output_dir, 'failed_batches')}"
            )
        if inputs["input_ids"].shape[1] > args.max_input_tokens:
            continue
        batch_dir = os.path.join(args.output_dir, f"batch_{batch_number:02d}")
        os.makedirs(batch_dir, exist_ok=True)
        for name, value in [
            ("full_rubric.md", text),
            ("general_rubric.md", general),
            ("domain_specific_rubric.md", domain),
            ("source_examples.json", json.dumps(examples, ensure_ascii=False, indent=2)),
        ]:
            with open(os.path.join(batch_dir, name), "w", encoding="utf-8") as handle:
                handle.write(value.rstrip() + "\n")
        print(f"Updated initial rubric with batch {batch_number}/{len(batches)}")
        index += 1

    for name, value in [
        ("full_rubric.md", text),
        ("general_rubric.md", general),
        ("domain_specific_rubric.md", domain),
        ("source_examples.json", json.dumps(all_examples, ensure_ascii=False, indent=2)),
    ]:
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as handle:
            handle.write(value.rstrip() + "\n")
    print(f"Saved initial rubric to {args.output_dir}")


if __name__ == "__main__":
    main()
