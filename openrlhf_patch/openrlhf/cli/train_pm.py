import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

from torch import nn
from transformers.trainer import get_scheduler

from openrlhf.datasets import PreferenceDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer import PreferenceModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def freeze_except_value_head(model, value_head_prefix):
    for param in model.parameters():
        param.requires_grad = False

    trainable = []
    value_head = getattr(model, value_head_prefix, None)
    if value_head is not None:
        for name, param in value_head.named_parameters():
            param.requires_grad = True
            trainable.append(f"{value_head_prefix}.{name}")
    else:
        for name, param in model.named_parameters():
            if name == value_head_prefix or name.startswith(f"{value_head_prefix}.") or f".{value_head_prefix}." in name:
                param.requires_grad = True
                trainable.append(name)

    if not trainable:
        raise ValueError(f"No trainable value head parameters found for prefix {value_head_prefix!r}")
    return trainable


def freeze_except_last_layers_and_value_head(model, value_head_prefix, num_layers):
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    for param in model.parameters():
        param.requires_grad = False

    layer_lists = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and name.endswith("layers") and len(module) > 0
    ]
    if not layer_lists:
        raise ValueError("Could not find a transformer ModuleList ending in 'layers'")
    layers_name, layers = max(layer_lists, key=lambda item: len(item[1]))
    if num_layers > len(layers):
        raise ValueError(f"Requested {num_layers} layers, but {layers_name} has {len(layers)}")

    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    value_head = getattr(model, value_head_prefix, None)
    if value_head is None:
        raise ValueError(f"No value head found for prefix {value_head_prefix!r}")
    for param in value_head.parameters():
        param.requires_grad = True

    return layers_name, len(layers)


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "preference",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=args.init_value_head,
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
        use_sn=args.use_sn,
        use_gp=args.use_gp,
        use_mcd=args.use_mcd,
        mcd_p=args.mcd_p,
        gp_amplitude=args.gp_amplitude,
        sn_range=args.sn_range,
        use_laplace=args.use_laplace,
        laplace_ridge=args.laplace_ridge,
        laplace_amplitude=args.laplace_amplitude,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    if args.train_value_head_only and args.unfreeze_last_n_layers > 0:
        raise ValueError("Use only one of --train_value_head_only and --unfreeze_last_n_layers")
    if args.train_value_head_only:
        trainable = freeze_except_value_head(model, args.value_head_prefix)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        strategy.print(
            f"Training value head only: prefix={args.value_head_prefix}, "
            f"trainable_params={trainable_params}, total_params={total_params}, "
            f"trainable_names={trainable}"
        )
    elif args.unfreeze_last_n_layers > 0:
        layers_name, total_layers = freeze_except_last_layers_and_value_head(
            model,
            args.value_head_prefix,
            args.unfreeze_last_n_layers,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        strategy.print(
            f"Training value head and last {args.unfreeze_last_n_layers}/{total_layers} layers "
            f"from {layers_name}: trainable_params={trainable_params}, total_params={total_params}"
        )

    strategy.print(model)

    ref_model = None
    if args.preference_distill_coef > 0:
        if not args.ref_pretrain:
            raise ValueError("--ref_pretrain is required when --preference_distill_coef is positive")
        ref_model = get_llm_for_sequence_regression(
            args.ref_pretrain,
            "preference",
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            init_value_head=args.init_value_head,
            value_head_prefix=args.value_head_prefix,
            packing_samples=args.packing_samples,
            use_sn=args.use_sn,
            use_gp=args.use_gp,
            use_mcd=False,
            gp_amplitude=args.gp_amplitude,
            sn_range=args.sn_range,
            use_laplace=args.use_laplace,
            laplace_ridge=args.laplace_ridge,
            laplace_amplitude=args.laplace_amplitude,
        )
        if args.ref_offload:
            ref_model._offload = True
        get_tokenizer(
            args.ref_pretrain,
            ref_model,
            "left",
            strategy,
            use_fast=not args.disable_fast_tokenizer,
        )
        strategy.print(
            f"Preference distillation enabled: ref_pretrain={args.ref_pretrain}, "
            f"coef={args.preference_distill_coef}, temperature={args.preference_distill_temperature}, "
            f"correct_weight={args.preference_distill_correct_weight}, "
            f"wrong_weight={args.preference_distill_wrong_weight}"
        )

    # configure optimizer
    # optim = strategy.create_optimizer(getattr(model, args.value_head_prefix), lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        stopping_strategy="all_exhausted",
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = PreferenceDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )
    eval_dataset = PreferenceDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    num_warmup_steps = math.ceil(max_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )
    strategy.print(
        f"LR scheduler: cosine_with_min_lr, max_steps={max_steps}, "
        f"warmup_ratio={args.warmup_ratio}, warmup_steps={num_warmup_steps}"
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # strategy prepare
    if ref_model is None:
        (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    else:
        ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = PreferenceModelTrainer(
        model=model,
        ref_model=ref_model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--use_sn", action="store_true", default=False, help="Enable Spectral Normalization")
    parser.add_argument("--sn_range", type=float, default=10., help="Spectral Normalization Range")
    parser.add_argument("--use_gp", action="store_true", default=False, help="Enable Gaussian Process")
    parser.add_argument("--gp_amplitude", type=float, default=0.1, help="Gaussian Process Amplitude")
    parser.add_argument("--use_laplace", action="store_true", default=False, help="Enable last-layer Laplace uncertainty")
    parser.add_argument("--laplace_ridge", type=float, default=0.001, help="Last-layer Laplace ridge precision")
    parser.add_argument("--laplace_amplitude", type=float, default=0.1, help="Last-layer Laplace uncertainty scale")
    parser.add_argument("--use_mcd", action="store_true", default=False, help="Enable MC Dropout")
    parser.add_argument("--mcd_p", type=float, default=0.2, help="MC Dropout rate")
    parser.add_argument("--train_value_head_only", action="store_true", default=False, help="Freeze backbone and train only the value/preference head.")
    parser.add_argument(
        "--unfreeze_last_n_layers",
        type=int,
        default=0,
        help="Freeze the backbone except its last N transformer layers and the value/preference head.",
    )
    parser.add_argument(
        "--init_value_head",
        action="store_true",
        default=False,
        help="Randomly initialize the preference/value head. Leave disabled when loading an existing PM checkpoint.",
    )
    parser.add_argument("--ref_pretrain", type=str, default=None, help="Frozen preference model used as the behavior teacher.")
    parser.add_argument("--ref_offload", action="store_true", default=False, help="Offload the frozen teacher model.")
    parser.add_argument("--preference_distill_coef", type=float, default=0.0)
    parser.add_argument("--preference_distill_temperature", type=float, default=2.0)
    parser.add_argument("--preference_distill_correct_weight", type=float, default=1.0)
    parser.add_argument("--preference_distill_wrong_weight", type=float, default=0.0)

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # RM training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--loss", type=str, default="sigmoid")  # ["sigmoid", "aps", "cls_temp", "scaled_bt"]
    parser.add_argument("--center_coef", type=float, default=1.0)
    parser.add_argument("--cls_temperature", type=float, default=1.0)
    parser.add_argument("--aps_mode", type=str, default="linear")
    parser.add_argument("--aps_rho", type=float, default=0.5)
    parser.add_argument("--aps_tau_max", type=float, default=4.0)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--context_key", type=str, default="context_messages")
    parser.add_argument("--strength_key", type=str, default="strength")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=512)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.preference_distill_coef < 0:
        raise ValueError("preference_distill_coef must be non-negative")
    if args.preference_distill_temperature <= 0:
        raise ValueError("preference_distill_temperature must be positive")
    if args.preference_distill_correct_weight < 0 or args.preference_distill_wrong_weight < 0:
        raise ValueError("preference distillation weights must be non-negative")

    train(args)
