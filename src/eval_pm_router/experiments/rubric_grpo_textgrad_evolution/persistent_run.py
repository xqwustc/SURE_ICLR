"""Keep generation and distributed evaluation models alive across rounds."""
import contextlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def worker(module):
    with contextlib.redirect_stdout(sys.stderr):
        target = importlib.import_module(module)
    for line in sys.stdin:
        sys.argv = [module, *json.loads(line)]
        with contextlib.redirect_stdout(sys.stderr):
            target.main()
        print('DONE', flush=True)


def copy_rubric(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('general_rubric.md', 'domain_specific_rubric.md'):
        shutil.copyfile(source / name, destination / name)


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def generation_batches(pool_path, prompt_probe_path, eval_path, per_domain, seed, generation):
    if prompt_probe_path.is_file() and eval_path.is_file():
        return
    groups = {}
    for row in read_jsonl(pool_path):
        groups.setdefault(str(row.get('domain', 'unknown')), []).append(row)
    rng = random.Random(seed + generation * 1009)
    prompt_probe = []
    eval_batch = []
    counts = {}
    for domain, rows in sorted(groups.items()):
        rows = list(rows)
        rng.shuffle(rows)
        prompt_chosen = rows[:min(per_domain, len(rows))]
        eval_chosen = rows[len(prompt_chosen):len(prompt_chosen) + per_domain]
        if len(eval_chosen) < per_domain:
            eval_chosen.extend(rows[:per_domain - len(eval_chosen)])
        prompt_probe.extend(prompt_chosen)
        eval_batch.extend(eval_chosen)
        counts[domain] = {
            'prompt_probe': len(prompt_chosen),
            'eval': len(eval_chosen),
            'overlap': len({row.get('pair_index') for row in prompt_chosen} & {row.get('pair_index') for row in eval_chosen}),
        }
    rng.shuffle(prompt_probe)
    rng.shuffle(eval_batch)
    write_jsonl(prompt_probe_path, prompt_probe)
    write_jsonl(eval_path, eval_batch)
    with open(eval_path.with_suffix('.metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'generation': generation,
                'seed': seed + generation * 1009,
                'per_domain': per_domain,
                'counts': counts,
                'prompt_probe_total': len(prompt_probe),
                'eval_total': len(eval_batch),
            },
            handle,
            indent=2,
        )
        handle.write('\n')
    shutil.copyfile(eval_path.with_suffix('.metadata.json'), prompt_probe_path.with_suffix('.metadata.json'))


def parent_error_prompt_batch(predictions_path, output_path, per_domain, seed, generation):
    if output_path.is_file():
        return
    groups = {}
    for row in read_jsonl(predictions_path):
        row = dict(row)
        row['router_winner'] = row.get('judge_winner')
        row['router_reason'] = row.get('judge_reason', row.get('router_reason', ''))
        row['parent_prediction'] = row.get('prediction')
        row['parent_is_correct'] = bool(row.get('is_correct'))
        groups.setdefault(str(row.get('domain', 'unknown')), []).append(row)
    rng = random.Random(seed + generation * 2027)
    selected = []
    counts = {}
    for domain, rows in sorted(groups.items()):
        rows = list(rows)
        wrong = [row for row in rows if not row.get('parent_is_correct')]
        correct = [row for row in rows if row.get('parent_is_correct')]
        rng.shuffle(wrong)
        rng.shuffle(correct)
        chosen = (wrong + correct)[:min(per_domain, len(rows))]
        selected.extend(chosen)
        counts[domain] = {'wrong_first': len(wrong[:per_domain]), 'total': len(chosen)}
    # Keep parent mistakes near the front so case truncation preserves the useful signal.
    selected.sort(key=lambda row: (bool(row.get('parent_is_correct')), str(row.get('domain')), rng.random()))
    write_jsonl(output_path, selected)
    with open(output_path.with_suffix('.metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'generation': generation,
                'seed': seed + generation * 2027,
                'per_domain': per_domain,
                'selection': 'parent_errors_first',
                'counts': counts,
                'total': len(selected),
            },
            handle,
            indent=2,
        )
        handle.write('\n')


def main():
    env = os.environ
    root = Path(env['RUN_ROOT'])
    initial = root / 'trajectory/round_00'
    generations = int(env['GENERATIONS'])
    eval_gpus = env['EVAL_GPUS'].split(',')
    if len(eval_gpus) != int(env['NPROC_PER_NODE']):
        raise ValueError('NPROC_PER_NODE must match EVAL_GPUS')
    if set(eval_gpus) & set(env['EVOLVER_GPUS'].split(',')):
        raise ValueError('Persistent generation and evaluation require disjoint GPU lists')
    processes = []

    def launch(module, updates):
        settings = dict(env)
        for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
            settings.pop(key, None)
        settings.update(updates)
        process = subprocess.Popen(
            [sys.executable, '-u', __file__, '--worker', module],
            env=settings, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8',
        )
        processes.append(process)
        return process

    def invoke(workers, arguments):
        for process in workers:
            process.stdin.write(json.dumps(arguments) + '\n')
            process.stdin.flush()
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            pending = [pool.submit(process.stdout.readline) for process in workers]
            for result in as_completed(pending):
                if result.result().strip() != 'DONE':
                    for process in processes:
                        process.terminate()
                    raise RuntimeError('Persistent worker failed; inspect the preceding traceback')

    generator = None
    evaluators = []

    def evaluate(path, directory, holdout=False):
        if not evaluators:
            for rank in range(len(eval_gpus)):
                evaluators.append(launch('evaluate_generation', {
                    'CUDA_VISIBLE_DEVICES': env['EVAL_GPUS'],
                    'LOCAL_RANK': str(rank), 'RANK': str(rank),
                    'WORLD_SIZE': str(len(eval_gpus)), 'MASTER_ADDR': '127.0.0.1',
                    'MASTER_PORT': env['MASTER_PORT_BASE'], 'RUBRIC_PERSISTENT_WORKER': '1',
                }))
        arguments = [
            '--model_path', env['PM_MODEL'], '--input', str(path),
            '--generation_dir', str(directory), '--batch_size', env['EVAL_BATCH_SIZE'],
            '--max_len', env['MAX_LEN'], '--tokenizer_chat_template', env['QWEN_PM_CHAT_TEMPLATE'],
        ]
        invoke(evaluators, arguments + (['--evaluation_only'] if holdout else []))

    try:
        parent = initial
        previous = None
        pool_path = root / 'data/evolution_pool.jsonl'
        if not pool_path.is_file():
            pool_path = root / 'data/evolution_dev.jsonl'
        for generation in range(1, generations + 1):
            directory = root / f'generation_{generation:02d}'
            if not (directory / 'generation_summary.json').is_file():
                copy_rubric(parent, directory / 'parent')
                if generator is None:
                    generator = launch('generate_candidates', {'CUDA_VISIBLE_DEVICES': env['EVOLVER_GPUS']})
                arguments = [
                    '--model_path', env['EVOLVER_MODEL'], '--parent_rubric', str(directory / 'parent'),
                    '--output_dir', str(directory), '--generation', str(generation), '--resume',
                ]
                for flag, variable in [('rollouts', 'ROLLOUTS'), ('temperature', 'TEMPERATURE'),
                                       ('top_p', 'TOP_P'), ('max_similarity', 'MAX_SIMILARITY'),
                                       ('seed', 'SEED'), ('max_input_tokens', 'MAX_INPUT_TOKENS'),
                                       ('max_new_tokens', 'MAX_NEW_TOKENS'),
                                       ('max_case_chars', 'MAX_CASE_CHARS'),
                                       ('max_feedback_chars', 'MAX_FEEDBACK_CHARS')]:
                    arguments += ['--' + flag, env[variable]]
                if previous is not None:
                    arguments += ['--feedback', str(previous / 'textgrad_feedback.txt'),
                                  '--previous_generation', str(previous)]
                prompt_probe_path = directory / 'prompt_probe.jsonl'
                prompt_batch_path = directory / 'prompt_batch.jsonl'
                eval_batch_path = directory / 'evolution_batch.jsonl'
                generation_batches(pool_path, prompt_probe_path, eval_batch_path, int(env['PER_DOMAIN']), int(env['SEED']), generation)
                parent_eval_dir = directory / 'prompt_parent_eval'
                if not (parent_eval_dir / 'evaluation_summary.json').is_file():
                    copy_rubric(parent, parent_eval_dir / 'parent')
                    evaluate(prompt_probe_path, parent_eval_dir, True)
                parent_error_prompt_batch(
                    parent_eval_dir / 'parent/dev_predictions.jsonl',
                    prompt_batch_path,
                    int(env['PER_DOMAIN']),
                    int(env['SEED']),
                    generation,
                )
                arguments += ['--case_batch', str(prompt_batch_path)]
                invoke([generator], arguments)
                evaluate(eval_batch_path, directory)
            parent = directory / 'accepted_rubric'
            previous = directory
            snapshot = root / f'trajectory/round_{generation:02d}'
            copy_rubric(parent, snapshot)
            shutil.copyfile(directory / 'generation_summary.json', snapshot / 'generation_summary.json')
            subprocess.run([sys.executable, str(Path(__file__).with_name('export_trajectory.py')),
                            '--run_root', str(root), '--generations', str(generation)], check=True)
        holdout = root / 'final_holdout'
        copy_rubric(parent, holdout / 'parent')
        copy_rubric(initial, holdout / 'initial')
        evaluate(root / 'data/holdout.jsonl', holdout, True)
        final = Path(env['FINAL_RUBRIC_DIR'])
        copy_rubric(parent, final)
        for source, name in [(root / 'data/split_metadata.json', 'split_metadata.json'),
                             (root / 'rubric_trajectory.jsonl', 'rubric_trajectory.jsonl'),
                             (holdout / 'evaluation_summary.json', 'holdout_metrics.json')]:
            shutil.copyfile(source, final / name)
        print(f'Evolution run: {root}\nFinal rubric: {final}')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
