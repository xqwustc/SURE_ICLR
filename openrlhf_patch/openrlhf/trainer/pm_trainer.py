import math
import os
from abc import ABC

import loralib as lora
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import PreferenceLoss
from openrlhf.models.ring_attn_utils import convert_ring_attn_params
from openrlhf.utils.distributed_sampler import DistributedSampler
import torch.nn.functional as F


class PreferenceModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
        ref_model=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.ref_model = ref_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        if loss in ["sigmoid", "scaled_bt"]: 
            self.loss_fn = PreferenceLoss()
        elif loss == "rdrop":
            self.loss_fn = PreferenceLoss()
        else: 
            raise "Not implemented loss function."

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model is not None:
                self.ref_model.eval()
            acc_mean = 0
            loss_mean = 0
            if self.args.loss in ["aps", "cls_temp"]:
                self.batch_tau = []
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, label, strength = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward, aux_loss = self.concatenated_forward(self.model, chosen_ids, c_mask)
                    teacher_reward = None
                    if self.ref_model is not None:
                        with torch.no_grad():
                            teacher_reward, _ = self.concatenated_forward(self.ref_model, chosen_ids, c_mask)
                    if self.args.loss == "rdrop":
                        chosen_reward_, aux_loss_ = self.concatenated_forward(self.model, chosen_ids, c_mask)
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, label, strength = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, aux_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )
                    teacher_reward = None
                    if self.ref_model is not None:
                        with torch.no_grad():
                            teacher_reward, _ = self.packed_samples_forward(
                                self.ref_model,
                                packed_input_ids,
                                packed_attention_masks,
                                packed_seq_lens,
                            )
                    if self.args.loss == "rdrop":
                        chosen_reward_, aux_loss_ = self.packed_samples_forward(
                            self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                        )

                label = torch.tensor(label).to(torch.cuda.current_device())
                strength = torch.tensor(strength).to(torch.cuda.current_device())

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()

                preference_loss, acc_raw = self.loss_fn(chosen_reward, label, strength)
                distill_loss = None
                distill_logs = {}
                if teacher_reward is not None:
                    distill_loss, distill_logs = self.preference_distillation_loss(
                        chosen_reward,
                        teacher_reward,
                        label,
                    )
                if self.args.loss == "rdrop":
                    preference_loss_, acc_raw_ = self.loss_fn(chosen_reward_, label, strength)
                    p1 = torch.sigmoid(chosen_reward)
                    p2 = torch.sigmoid(chosen_reward_)
                    align_loss = (p1 * (F.logsigmoid(p1) - F.logsigmoid(p2)) + (1-p1) * (F.logsigmoid(-p1) - F.logsigmoid(-p2)) + 
                                  p2 * (F.logsigmoid(p2) - F.logsigmoid(p1)) + (1-p2) * (F.logsigmoid(-p2) - F.logsigmoid(-p1))).mean()
                    preference_loss = 0.5 * (preference_loss + preference_loss_)
                    acc_raw = (acc_raw + acc_raw_) / 2

                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                if distill_loss is not None:
                    loss = loss + self.args.preference_distill_coef * distill_loss
                if self.args.loss == "rdrop":
                    loss = loss + 0.5 * align_loss
                
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = acc_raw.item()
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "reward_diff": chosen_reward.mean().item(),
                    "reward_diff_std": chosen_reward.std().item(),
                    "loss_mean": loss_mean,
                    "acc_mean": acc_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.args.use_gp:
                    logs_dict["out_norm"] = self.model.score.out_layer.weight.data.norm().item()

                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                if distill_loss is not None:
                    logs_dict["distill_loss"] = distill_loss.item()
                    logs_dict.update(distill_logs)

                if self.args.loss == "rdrop":
                    logs_dict["align_loss"] = align_loss.item()

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self.args.use_gp:
            self.calibrate_gp()
        if self.args.use_laplace:
            self.calibrate_laplace()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def calibrate_laplace(self):
        step_bar = tqdm(
            range(self.train_dataloader.__len__()),
            desc="Calibrate last-layer Laplace",
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, _, _ = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    self.concatenated_forward(self.model, chosen_ids, c_mask, last_epoch=True)
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, _, _ = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, last_epoch=True
                    )
                step_bar.update()

        model = self.model
        while hasattr(model, "module"):
            model = model.module
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(model.laplace_precision, op=dist.ReduceOp.SUM)

    def calibrate_gp(self):
        step_bar = tqdm(
            range(self.train_dataloader.__len__()),
            desc="Calibrate SNGP posterior",
            disable=not self.strategy.is_rank_0(),
        )
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        gp_layer = getattr(model, self.args.value_head_prefix)
        gp_layer.inv_cov.zero_()
        gp_layer.cov = None

        self.model.eval()
        with torch.no_grad():
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, _, _ = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    self.concatenated_forward(self.model, chosen_ids, c_mask, last_epoch=True)
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, _, _ = data
                    packed_input_ids = packed_input_ids.to(torch.cuda.current_device())
                    packed_attention_masks = packed_attention_masks.to(torch.cuda.current_device())
                    self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, last_epoch=True
                    )
                step_bar.update()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gp_layer.inv_cov, op=dist.ReduceOp.SUM)

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            distill_loss_sum = 0
            score_shift_sum = 0
            flip_rate_sum = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, label, strength = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask
                    )
                    teacher_reward = None
                    if self.ref_model is not None:
                        teacher_reward, _ = self.concatenated_forward(self.ref_model, chosen_ids, c_mask)
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, label, strength = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )
                    teacher_reward = None
                    if self.ref_model is not None:
                        teacher_reward, _ = self.packed_samples_forward(
                            self.ref_model,
                            packed_input_ids,
                            packed_attention_masks,
                            packed_seq_lens,
                        )

                label = torch.tensor(label).to(torch.cuda.current_device())
                strength = torch.tensor(strength).to(torch.cuda.current_device())

                loss, acc_raw = self.loss_fn(chosen_reward, label, strength)
                if teacher_reward is not None:
                    distill_loss, distill_logs = self.preference_distillation_loss(
                        chosen_reward,
                        teacher_reward,
                        label,
                    )
                    distill_loss_sum += distill_loss.item()
                    score_shift_sum += distill_logs["teacher_score_abs_shift"]
                    flip_rate_sum += distill_logs["teacher_flip_rate"]

                rewards += [chosen_reward.flatten()]
                acc += acc_raw.item()
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            if self.ref_model is not None:
                bar_dict.update(
                    {
                        "distill_loss": distill_loss_sum / self.eval_dataloader.__len__(),
                        "teacher_score_abs_shift": score_shift_sum / self.eval_dataloader.__len__(),
                        "teacher_flip_rate": flip_rate_sum / self.eval_dataloader.__len__(),
                    }
                )
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def preference_distillation_loss(self, student_reward, teacher_reward, label):
        temperature = self.args.preference_distill_temperature
        if temperature <= 0:
            raise ValueError("preference_distill_temperature must be positive")

        student_logits = student_reward.float().flatten() / temperature
        teacher_logits = teacher_reward.float().flatten() / temperature
        teacher_prob = torch.sigmoid(teacher_logits)

        kl = teacher_prob * (F.logsigmoid(teacher_logits) - F.logsigmoid(student_logits))
        kl += (1 - teacher_prob) * (F.logsigmoid(-teacher_logits) - F.logsigmoid(-student_logits))

        labels = label.float().flatten()
        teacher_prediction = (teacher_reward.float().flatten() < 0).to(labels.dtype)
        teacher_correct = teacher_prediction.eq(labels).float()
        weights = torch.where(
            teacher_correct.bool(),
            torch.full_like(teacher_correct, self.args.preference_distill_correct_weight),
            torch.full_like(teacher_correct, self.args.preference_distill_wrong_weight),
        )
        distill_loss = (kl * weights).sum() / weights.sum().clamp(min=1.0)
        distill_loss = distill_loss * temperature**2

        logs = {
            "teacher_score_abs_shift": (student_reward.float() - teacher_reward.float()).abs().mean().item(),
            "teacher_flip_rate": (
                (student_reward.float() >= 0) != (teacher_reward.float() >= 0)
            ).float().mean().item(),
            "teacher_correct_fraction": teacher_correct.mean().item(),
            "distill_active_fraction": (weights > 0).float().mean().item(),
        }
        return distill_loss, logs

    def concatenated_forward(self, model, chosen_ids, c_mask, last_epoch = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True, last_epoch=last_epoch)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], 0)
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], 0)
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0),), dim=0)
        return inputs_ids, att_masks

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, last_epoch = False):
        all_values, output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            last_epoch=last_epoch,
        )
        half_len = len(packed_seq_lens)
        chosen_rewards = all_values[:half_len]
        aux_loss = output.aux_loss if "aux_loss" in output else []

        return chosen_rewards, aux_loss
