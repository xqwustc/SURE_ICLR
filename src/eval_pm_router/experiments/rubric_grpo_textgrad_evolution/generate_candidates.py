#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache


@lru_cache(maxsize=1)
def load_generator(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    return tokenizer, model


DOMAINS = ["chat", "code", "math", "safety-refuse", "safety-response"]
META_TEXT_PATTERN = re.compile(
    r"\b("
    r"rollout directive|textual gradient|previous group feedback|current-round cases|"
    r"selected examples|current rubric draft|current general rubric|current domain rubric|"
    r"parent rubric|strong judge|external judge|judge reasons|search strategy"
    r")\b",
    re.I,
)
PREFERENCE_BULLET_PATTERN = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+"
    r"(?:prefer|penalize|discount|treat|break ties|prioritize|deprioritize|rank|compare|reward)\b",
    re.I,
)
ANSWER_GENERATION_BULLET_PATTERN = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+"
    r"(?:provide|use|ensure|include|offer|respond|address|maintain|demonstrate|show|write|present|"
    r"verify|follow|redirect|refrain|encourage|document|handle|adhere|focus)\b",
    re.I,
)
SEARCH_DIRECTIONS = [
    "prioritize factual correctness and decisive error detection",
    "strengthen the cross-domain general decision hierarchy",
    "improve robustness to polished but subtly incorrect answers",
    "clarify domain-specific evidence and verification criteria",
    "reduce position, verbosity, and style bias",
    "improve treatment of incomplete reasoning and unsupported claims",
    "make safety judgments distinguish refusal quality from helpful safe response",
    "make math and code criteria emphasize executable or checkable correctness",
    "seek a balanced revision across all domains without adding verbosity",
]
MUTATIONS = [
    ("targeted", "Revise decisive pairwise preference criteria in each affected domain. Name observable evidence and a tie-break rule."),
    ("general_foundation", "Rewrite the general rubric into a stronger cross-domain pairwise decision hierarchy, then make only targeted domain edits supported by the cases."),
    ("prune", "Delete weak or repetitive rules while preserving a complete pairwise hierarchy and the most decisive domain checks."),
    ("priority", "Replace the flat checklist with an explicit preference hierarchy: disqualifying errors, primary criteria, secondary criteria, then tie-breakers. Change which criteria dominate."),
    ("verification", "Rebuild the rubric around concrete falsification checks for comparing claims, code, math, and safety. Replace vague quality adjectives with observable tests."),
    ("tradeoffs", "Replace absolute rules with conditional pairwise tradeoffs: when completeness, caution, brevity, or reasoning should determine the preferred response."),
    ("counterhypothesis", "Challenge an assumption in the current rubric or feedback. Remove or reverse a plausible harmful judging heuristic, while preserving correctness and safety."),
    ("restart_minimal", "Design from scratch without the parent's text. Use a compact but complete preference hierarchy with at least six general rules and at least four rules per domain."),
    ("restart_procedure", "Design from scratch without the parent's text. Use a short domain-specific comparison procedure rather than a generic quality checklist."),
]


def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def case_items(path):
    if not path:
        return []
    rows = []
    for row in read_jsonl(path):
        winner = row.get("router_winner") or row.get("judge_winner")
        if winner not in {"assistant_1", "assistant_2"}:
            raise ValueError(f"Invalid router_winner={winner!r} in case batch")
        item = {
            "domain": row.get("domain", ""),
            "prompt": truncate(row.get("prompt"), 500),
            "assistant_1": truncate(row.get("assistant_1"), 700),
            "assistant_2": truncate(row.get("assistant_2"), 700),
            "judge_winner": winner,
            "judge_reason": truncate(row.get("router_reason") or row.get("judge_reason"), 450),
        }
        if "parent_prediction" in row:
            item["parent_prediction"] = row.get("parent_prediction")
            item["parent_is_correct"] = bool(row.get("parent_is_correct"))
        rows.append(item)
    return rows


def case_context_from_items(rows, max_chars):
    if not rows:
        return "No current-round cases were provided."
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    kept = []
    size = 2
    for row in rows:
        item = json.dumps(row, ensure_ascii=False, indent=2)
        if size + len(item) + 3 > max_chars:
            break
        kept.append(row)
        size += len(item) + 3
    return json.dumps(kept, ensure_ascii=False, indent=2)


def case_context(path, max_chars):
    return case_context_from_items(case_items(path), max_chars)


def mutable_domains_from_cases(rows):
    if not rows:
        return list(DOMAINS)
    wrong = {row["domain"] for row in rows if row.get("parent_is_correct") is False and row.get("domain") in DOMAINS}
    source = wrong or {row["domain"] for row in rows if row.get("domain") in DOMAINS}
    return [domain for domain in DOMAINS if domain in source]


def domain_bodies(domain_rubric):
    sections = {}
    current = None
    for line in domain_rubric.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            heading = match.group(1).strip("* ").rstrip(":").strip()
            if heading not in DOMAINS:
                raise ValueError(f"unknown domain section: {heading}")
            if heading in sections:
                raise ValueError(f"duplicate section: {heading}")
            sections[heading] = []
            current = heading
        elif current:
            sections[current].append(line)
        elif line.strip():
            raise ValueError(f"text outside domain sections: {line.strip()[:80]}")
    missing = [domain for domain in DOMAINS if domain not in sections]
    if missing:
        raise ValueError(f"missing domain sections: {', '.join(missing)}")
    bodies = {domain: "\n".join(lines).strip() for domain, lines in sections.items()}
    empty = [domain for domain in DOMAINS if not bodies[domain]]
    if empty:
        raise ValueError(f"empty domain sections: {', '.join(empty)}")
    return bodies


def render_domain_bodies(bodies):
    return "\n\n".join(f"## {domain}\n{bodies[domain].strip()}" for domain in DOMAINS)


def rubric_bullets(text):
    for line in text.splitlines():
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
            yield line.strip()


def validate_preference_wording(general, domain, mutable_domains=None):
    bodies = domain_bodies(domain)
    sections = [("General Updated Rubric", general)]
    sections.extend((name, bodies[name]) for name in (mutable_domains or DOMAINS))
    errors = []
    for section, body in sections:
        bullets = list(rubric_bullets(body))
        if not bullets:
            errors.append(f"{section}: no bullet criteria")
            continue
        for bullet in bullets:
            if ANSWER_GENERATION_BULLET_PATTERN.match(bullet):
                errors.append(f"{section}: answer-generation wording: {bullet[:100]}")
            elif not PREFERENCE_BULLET_PATTERN.match(bullet):
                errors.append(f"{section}: bullet must start with a preference verb: {bullet[:100]}")
    if errors:
        raise ValueError("non-preference rubric wording; " + "; ".join(errors[:6]))


def freeze_unselected_domains(parent_domain, candidate_domain, mutable_domains):
    parent = domain_bodies(parent_domain)
    candidate = domain_bodies(candidate_domain)
    mutable = set(mutable_domains)
    for domain in DOMAINS:
        if domain not in mutable:
            candidate[domain] = parent[domain]
    return render_domain_bodies(candidate)


def parse_rubric(markdown):
    titles = ["General Updated Rubric", "Domain-Specific Rubric", *DOMAINS]
    aliases = {title.lower(): title for title in titles}
    aliases["general rubric"] = titles[0]
    sections = {}
    current = None
    for line in markdown.splitlines():
        if re.match(r"^\s*```(?:markdown|md)?\s*$", line, re.I):
            continue
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        heading = (match.group(1) if match else line.strip()).strip("* ").rstrip(":").strip()
        title = aliases.get(heading.lower())
        if title:
            if title in sections:
                raise ValueError(f"duplicate section: {title}")
            sections[title] = []
            current = title
        elif match:
            raise ValueError(f"unknown section: {heading}")
        elif current:
            sections[current].append(line)
        elif line.strip():
            raise ValueError(f"text outside rubric sections: {line.strip()[:80]}")
    missing = [title for title in titles if title not in sections]
    if missing:
        raise ValueError(f"missing sections: {', '.join(missing)}")
    bodies = {title: '\n'.join(lines).strip() for title, lines in sections.items()}
    empty = [title for title in [titles[0], *DOMAINS] if not bodies[title]]
    if empty:
        raise ValueError(f"empty sections: {', '.join(empty)}")
    if bodies[titles[1]]:
        raise ValueError("Domain-Specific Rubric must contain only domain subsections")
    placeholders = [title for title in [titles[0], *DOMAINS] if re.search(r"\[[^\]]+\]", bodies[title])]
    if placeholders:
        raise ValueError(f"unreplaced template placeholders: {', '.join(placeholders)}")
    meta_sections = [title for title in [titles[0], *DOMAINS] if META_TEXT_PATTERN.search(bodies[title])]
    if meta_sections:
        raise ValueError(f"meta/process text leaked into rubric: {', '.join(meta_sections)}")
    domain = render_domain_bodies(bodies)
    return bodies[titles[0]], domain


def row_summary(row):
    return {
        "domain": row.get("domain"),
        "prompt": truncate(row.get("prompt"), 220),
        "prediction": row.get("prediction"),
        "expected": row.get("judge_winner") or row.get("router_winner"),
        "reason": truncate(row.get("judge_reason") or row.get("router_reason"), 240),
    }


def format_case_group(title, rows):
    if not rows:
        return []
    lines = [title]
    for row in rows:
        lines.append(json.dumps(row_summary(row), ensure_ascii=False))
    return lines


def format_feedback_summary(summary, max_chars, mutable_domains):
    candidates = summary.get("candidates", {})
    best_name = summary.get("best_candidate")
    best = candidates.get(best_name, {})
    parent = candidates.get("parent", {})
    lines = [
        "Variable-specific feedback for the next rubric update.",
        "Treat general_rubric as the cross-domain variable. Treat each domain subsection as a separate domain variable.",
        "Update general_rubric only when a pattern appears across domains or changes the global decision hierarchy.",
        "Update a domain variable only when the feedback names that domain; keep other domain variables unchanged.",
        f"previous_parent_macro={parent.get('macro_accuracy')} previous_best_macro={best.get('macro_accuracy')} accepted={summary.get('accepted')}",
        "",
        "[general_rubric feedback]",
        "- Strengthen cross-domain rules that reduce repeated fix/regress patterns.",
        "- Prefer observable checks and tie-breaks over broad quality words.",
    ]
    domain_fix_regress = best.get("domain_fix_regress", {})
    errors = summary.get("representative_errors", {})
    fixes = summary.get("representative_fixes", {})
    regressions = summary.get("representative_regressions", {})
    for domain in DOMAINS:
        if mutable_domains and domain not in mutable_domains:
            continue
        stats = domain_fix_regress.get(domain, {})
        lines.extend([
            "",
            f"[{domain}_rubric feedback]",
            f"- fixed={stats.get('fixed', 0)} regressed={stats.get('regressed', 0)} net={stats.get('net', 0)} reward={stats.get('reward', 0.0)}",
        ])
        lines.extend(format_case_group("- Parent mistakes still missed by the best candidate:", errors.get(domain, [])))
        lines.extend(format_case_group("- Useful fixes to preserve:", fixes.get(domain, [])))
        lines.extend(format_case_group("- Regressions to avoid:", regressions.get(domain, [])))
    text = "\n".join(lines)
    return text[:max_chars]


def feedback_text(path, max_chars, mutable_domains=None):
    if not path or not os.path.isfile(path):
        return "No previous-generation feedback is available. Explore competing judging policies, including substantial rewrites."
    text = read_text(path)
    try:
        return format_feedback_summary(json.loads(text), max_chars, set(mutable_domains or []))
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return text[:max_chars]


def build_prompt(general, domain, feedback, mutation, direction, cases, mutable_domains):
    mode, directive = mutation
    parent = "The parent text is intentionally withheld for an independent restart."
    domain_map = domain_bodies(domain)
    if not mode.startswith("restart_"):
        parent = f"[Current General Rubric]\n{general}\n\n[Current Domain Rubric]\n{domain}"
    else:
        # The raw feedback embeds best/worst rubric text; hide it for true restarts.
        feedback = "Develop an independent judging policy for the five domains; no previous rubric or feedback is supplied."
    mutable = set(mutable_domains)
    mutable_text = "\n".join(
        f"- {domain}_rubric: {'UPDATE using this round cases and feedback' if domain in mutable else 'COPY VERBATIM from the current rubric'}"
        for domain in DOMAINS
    )
    domain_template = "\n".join(
        "## " + name + "\n" + (
            "- Prefer the response that [" + name + "-specific pairwise scoring criterion].\n"
            "- Penalize the response that [" + name + "-specific pairwise scoring criterion].\n"
            "- Discount [" + name + "-specific misleading or non-decisive feature].\n"
            "- Break ties by [" + name + "-specific tie-break criterion]."
            if name in mutable else domain_map[name]
        )
        for name in DOMAINS
    )
    return f"""Evolve a pairwise preference-judging rubric. The rubric will be prepended to a fixed preference reward model that compares two completed assistant responses. You are editing compact pairwise scoring criteria, not writing instructions for an assistant to produce an answer.

[Required Search Strategy: {mode}]
{directive}
Search emphasis: {direction}.
Change the actual decision policy, not just wording or formatting. You may delete, replace, or reorder existing rules. Do not copy the previous best candidate merely because it scored highest. Feedback is evidence to investigate, not a requirement to preserve all existing rules.

When advantage-guided feedback is available, directly compare candidates using their relative advantages: prioritize reusable added/removed rule changes marked reinforce, and investigate alternatives to changes marked avoid. Larger absolute advantages indicate larger deviations from the candidate group's mean reward, not statistical confidence. Apply domain-specific advantages only to the matching domain and general advantages to cross-domain priorities. A positive advantage alone does not establish improvement over the parent; check the reward and case evidence. Do not copy or undo rules mechanically, and do not write advantage values into the resulting rubric.

Carefully study the current-round cases and strong-judge reasons. Some cases include the parent rubric's prediction; focus especially on cases where parent_is_correct is false. Infer reusable pairwise judging principles from why one assistant response is preferred over the other. Do not copy case-specific answers into the rubric. Make one coherent candidate, generalize beyond examples, avoid answer-order bias, and do not mention benchmark difficulty labels or reveal expected winners. Do not reward verbosity or formatting by itself.

Write criteria that are as concise as possible while still detailed enough to guide scoring. First make General Updated Rubric a real cross-domain pairwise decision hierarchy: decisive correctness errors, instruction following, evidence and verifiability, completeness, safety, clarity, bias controls, and tie-breaks. Do not leave general as a short generic summary. Put only domain-agnostic rules in General Updated Rubric. Put chat/code/math/safety-refuse/safety-response-specific rules only inside the matching subsection; do not mix domain-specific rules into general, and do not copy one domain's rule into another unless it is genuinely reusable. Each bullet must describe how to compare two existing responses, not how to generate a response.

Every bullet must begin with one of these preference verbs: Prefer, Penalize, Discount, Treat, Break ties, Prioritize, Deprioritize, Rank, Compare, Reward. Do not begin bullets with answer-generation verbs such as Provide, Use, Ensure, Include, Offer, Respond, Address, Maintain, Demonstrate, Show, Write, Present, Verify, Follow, Redirect, Refrain, Encourage, Document, Handle, Adhere, or Focus. Keep the complete rubric under 1000 words so evaluation cases retain context space.

[Rubric Variables To Optimize]
- general_rubric: UPDATE every round as the cross-domain decision hierarchy.
{mutable_text}

{parent}

[Current-Round Cases For TextGrad Analysis]
{cases}

[Textual Gradient / Previous Group Feedback]
{feedback}

Fill this exact template, replacing each bracketed instruction with rubric text.
Copy every heading verbatim. Use short bullets inside sections, not additional markdown headings.
# General Updated Rubric
- Prefer the response that [domain-agnostic pairwise scoring criterion].
- Penalize the response that [domain-agnostic pairwise scoring criterion].
- Discount [domain-agnostic non-decisive feature or bias].
- Treat [domain-agnostic tradeoff or tie-break criterion].
- Break ties by [domain-agnostic tie-break criterion].
# Domain-Specific Rubric
{domain_template}
Every subsection must contain actionable criteria. Return only the rubric, with no preface, code fence, search-strategy commentary, extra sections, or case-specific notes.
"""


def fingerprint(general, domain):
    normalized = " ".join(f"{general}\n{domain}".split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def shingles(general, domain, mutable_domains=None):
    # Ignore fixed headings so format compliance does not count as similarity.
    parts = [general]
    domain_map = domain_bodies(domain)
    for name in (mutable_domains or DOMAINS):
        parts.append(domain_map[name])
    text = re.sub(r"(?m)^#+[^\n]*", "", "\n".join(parts))
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + 3]) for i in range(max(1, len(words) - 2))}


def similarity(left, right):
    return len(left & right) / max(1, len(left | right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parent_rubric", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feedback", default="")
    parser.add_argument("--case_batch", default="")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max_case_chars", type=int, default=20000)
    parser.add_argument("--max_feedback_chars", type=int, default=24000)
    parser.add_argument("--max_input_tokens", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_similarity", type=float, default=0.8)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--previous_generation", default="")
    parser.add_argument("--resume", action="store_true", help="Reuse and validate already saved candidates")
    args = parser.parse_args()
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or not 0 < args.max_similarity <= 1:
        parser.error("temperature must be positive; top_p and max_similarity must be in (0, 1]")
    if args.generation < 1 or args.rollouts < 1:
        parser.error("generation and rollouts must be positive")
    import torch
    from transformers import set_seed

    sampling_seed = args.seed + args.generation - 1
    set_seed(sampling_seed)

    general = read_text(os.path.join(args.parent_rubric, "general_rubric.md"))
    domain = read_text(os.path.join(args.parent_rubric, "domain_specific_rubric.md"))
    items = case_items(args.case_batch)
    mutable_domains = mutable_domains_from_cases(items)
    feedback = feedback_text(args.feedback, args.max_feedback_chars, mutable_domains)
    cases = case_context_from_items(items, args.max_case_chars)

    tokenizer, model = load_generator(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    seen = {fingerprint(general, domain)}
    prior_shingles = [shingles(general, domain, mutable_domains)]
    if args.previous_generation:
        for path in sorted(glob.glob(os.path.join(args.previous_generation, "candidate_*"))):
            if not os.path.isdir(path):
                continue
            old_general = read_text(os.path.join(path, "general_rubric.md"))
            old_domain = read_text(os.path.join(path, "domain_specific_rubric.md"))
            seen.add(fingerprint(old_general, old_domain))
            prior_shingles.append(shingles(old_general, old_domain, mutable_domains))
    manifest = []
    manifest_path = os.path.join(args.output_dir, "candidate_manifest.json")
    previous_manifest = {}
    if args.resume and os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            previous_manifest = {row["candidate"]: row for row in json.load(handle)}
    for index in range(1, args.rollouts + 1):
        direction = SEARCH_DIRECTIONS[(index + args.generation - 2) % len(SEARCH_DIRECTIONS)]
        mutation = MUTATIONS[(index - 1) % len(MUTATIONS)]
        candidate_name = f"candidate_{index:02d}"
        candidate_dir = os.path.join(args.output_dir, candidate_name)
        if args.resume and os.path.isdir(candidate_dir):
            candidate_general = read_text(os.path.join(candidate_dir, "general_rubric.md"))
            candidate_domain = read_text(os.path.join(candidate_dir, "domain_specific_rubric.md"))
            parse_rubric(f"# General Updated Rubric\n{candidate_general}\n# Domain-Specific Rubric\n{candidate_domain}")
            validate_preference_wording(candidate_general, candidate_domain, mutable_domains)
            digest = fingerprint(candidate_general, candidate_domain)
            candidate_shingles = shingles(candidate_general, candidate_domain, mutable_domains)
            max_similarity = max(similarity(candidate_shingles, previous) for previous in prior_shingles)
            saved_manifest = previous_manifest.get(candidate_name, {"candidate": candidate_name, "sha256": digest, "reused": True})
            is_fallback = bool(saved_manifest.get("fallback_used"))
            if not is_fallback and (digest in seen or max_similarity >= args.max_similarity):
                raise ValueError(f"Saved {candidate_name} violates current duplicate/similarity settings")
            seen.add(digest)
            prior_shingles.append(candidate_shingles)
            # Older failed runs have no manifest; do not invent their sampling settings.
            manifest.append(saved_manifest)
            print(f"Reusing saved rollout {index}/{args.rollouts}: {candidate_dir}")
            continue
        base_prompt = build_prompt(general, domain, feedback, mutation, direction, cases, mutable_domains)
        retry_feedback = ""
        temperature = args.temperature
        fallback_used = False
        for attempt in range(1, 6):
            prompt = (
                f"{base_prompt}\n\n[Rollout Directive]\n"
                f"This is candidate {index} of {args.rollouts}. Explore a distinct revision that {direction}. "
                f"Sampling attempt: {attempt}. Do not merely paraphrase the current rubric or another candidate. {retry_feedback}"
            )
            messages = [
                {"role": "system", "content": "You write pairwise preference-rubric criteria for a reward model that compares two completed responses. Follow the assigned mutation strategy and exact output format."},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                rendered, return_tensors="pt", truncation=False
            ).to(model.device)
            if inputs["input_ids"].shape[1] > args.max_input_tokens:
                raise ValueError("Evolution prompt exceeds max_input_tokens; reduce max_feedback_chars or shorten the parent rubric")
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
                candidate_general, candidate_domain = parse_rubric(text)
                validate_preference_wording(candidate_general, candidate_domain, mutable_domains)
                candidate_domain = freeze_unselected_domains(domain, candidate_domain, mutable_domains)
                text = f"# General Updated Rubric\n{candidate_general}\n\n# Domain-Specific Rubric\n{candidate_domain}"
                parse_rubric(text)
            except ValueError as error:
                reason = str(error)
                temperature = max(0.7, temperature - 0.15)
                retry_feedback = (
                    f"Previous attempt rejected: {reason}. Fill ALL seven template headings. "
                    "Use fewer words per section, finish safety-response, and include no extra headings, "
                    "preface, postscript, placeholders, or process notes. Every new bullet must start with "
                    "Prefer, Penalize, Discount, Treat, Break ties, Prioritize, Deprioritize, Rank, Compare, or Reward."
                )
            else:
                digest = fingerprint(candidate_general, candidate_domain)
                candidate_shingles = shingles(candidate_general, candidate_domain, mutable_domains)
                max_similarity = max(similarity(candidate_shingles, previous) for previous in prior_shingles)
                if digest not in seen and max_similarity < args.max_similarity:
                    break
                reason = f"duplicate or near-duplicate content (similarity={max_similarity:.3f})"
                retry_feedback = f"Previous attempt rejected: {reason}. Change the decision rules more substantially."
                temperature = args.temperature
            diagnostics = os.path.join(args.output_dir, "failed_attempts")
            os.makedirs(diagnostics, exist_ok=True)
            failure_dir = tempfile.mkdtemp(prefix=f"{candidate_name}_attempt_{attempt}_", dir=diagnostics)
            with open(os.path.join(failure_dir, "raw_response.md"), "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
            with open(os.path.join(failure_dir, "reason.txt"), "w", encoding="utf-8") as handle:
                handle.write(reason + "\n")
            print(f"Retrying rollout {index}: {reason} on attempt {attempt}")
        else:
            candidate_general = general
            candidate_domain = domain
            text = f"# General Updated Rubric\n{candidate_general}\n\n# Domain-Specific Rubric\n{candidate_domain}"
            digest = fingerprint(candidate_general, candidate_domain)
            candidate_shingles = shingles(candidate_general, candidate_domain, mutable_domains)
            max_similarity = 1.0
            attempt = 5
            fallback_used = True
            print(
                f"Rollout {index} failed after 5 attempts; using parent rubric as fallback. "
                f"Inspect {diagnostics} for rejected outputs."
            )
        seen.add(digest)
        prior_shingles.append(candidate_shingles)
        os.makedirs(candidate_dir)
        for name, value in [
            ("full_rubric.md", text),
            ("general_rubric.md", candidate_general),
            ("domain_specific_rubric.md", candidate_domain),
        ]:
            with open(os.path.join(candidate_dir, name), "w", encoding="utf-8") as handle:
                handle.write(value.rstrip() + "\n")
        manifest.append({"candidate": f"candidate_{index:02d}", "direction": direction, "mutation": mutation[0], "sha256": digest,
                         "max_similarity": max_similarity, "attempts": attempt, "generation": args.generation,
                         "seed": sampling_seed, "temperature": temperature, "top_p": args.top_p, "top_k": 0,
                         "mutable_domains": mutable_domains, "fallback_used": fallback_used})
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Saved rollout {index}/{args.rollouts} to {candidate_dir}")

    with open(os.path.join(args.output_dir, "candidate_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
