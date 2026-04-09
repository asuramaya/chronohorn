from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from chronohorn.service_log import service_log
from chronohorn.train.runtime_config import TrainConfig


@dataclass
class RunMetrics:
    seed: int
    params: int
    train_loss: float
    test_loss: float
    overfit_pct: float
    train_time_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random_module = getattr(mx, "random", None)
    if random_module is not None and hasattr(random_module, "seed"):
        random_module.seed(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(value.size for _, value in nn.utils.tree_flatten(model.trainable_parameters()))


def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    batch_size, timesteps, vocab_size = logits.shape
    return mx.mean(
        nn.losses.cross_entropy(
            logits.reshape(batch_size * timesteps, vocab_size),
            y.reshape(batch_size * timesteps),
        )
    )


def train_loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    custom_loss = getattr(model, "supervised_loss", None)
    if callable(custom_loss):
        return custom_loss(x, y)
    return loss_fn(model, x, y)


def build_compiled_loss(model: nn.Module) -> Callable[[mx.array, mx.array], mx.array]:
    return mx.compile(lambda x, y: loss_fn(model, x, y), inputs=model.state, outputs=model.state)


def evaluate(
    model: nn.Module,
    dataset: Any,
    train_config: TrainConfig,
    split: str,
    *,
    eval_batches: int | None = None,
    compiled_loss: Callable[[mx.array, mx.array], mx.array] | None = None,
) -> float:
    batches = train_config.eval_batches if eval_batches is None else eval_batches
    total = 0.0
    loss_impl = compiled_loss or loss_fn
    for _ in range(batches):
        x, y = dataset.batch(split, train_config.batch_size, train_config.seq_len)
        loss = loss_impl(model, x, y) if compiled_loss is None else loss_impl(x, y)
        mx.eval(loss)
        total += loss.item()
    return total / batches


def _build_train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip: float,
    compile_train_step: bool,
) -> tuple[Callable[[mx.array, mx.array], mx.array], bool]:
    value_and_grad = nn.value_and_grad(model, train_loss_fn)

    def eager_train_step(x: mx.array, y: mx.array) -> mx.array:
        loss, grads = value_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=grad_clip)
        optimizer.update(model, grads)
        return loss

    if not compile_train_step:
        return eager_train_step, False

    compiled_train_step = mx.compile(
        eager_train_step,
        inputs=(model.state, optimizer.state),
        outputs=(model.state, optimizer.state),
    )
    return compiled_train_step, True


def train_model(
    model: nn.Module,
    dataset: Any,
    train_config: TrainConfig,
    seed: int,
    label: str,
    on_step: Callable[[int, nn.Module, list[float]], None] | None = None,
    on_log: Callable[[int, float, float, float, int, float], str | None] | None = None,
    *,
    optimizer_kwargs: dict[str, Any] | None = None,
    compile_train_step: bool = False,
    compiled_eval_loss: Callable[[mx.array, mx.array], mx.array] | None = None,
    final_eval_batches: int | None = None,
) -> RunMetrics:
    del label
    params = count_trainable_params(model)
    if optimizer_kwargs is None:
        optimizer_kwargs = {
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
        }
    optimizer = optim.AdamW(**optimizer_kwargs)
    log_component = "train.runtime"
    train_step, using_compiled_train_step = _build_train_step(
        model,
        optimizer,
        train_config.grad_clip,
        compile_train_step=compile_train_step,
    )
    compile_failure_reported = False

    seed_everything(seed + 1000)
    losses: list[float] = []
    best = float("inf")
    start = time.time()
    last_log_time = start
    last_log_step = 0

    for step in range(1, train_config.steps + 1):
        x, y = dataset.batch("train", train_config.batch_size, train_config.seq_len)
        try:
            loss = train_step(x, y)
        except Exception as exc:
            if not using_compiled_train_step:
                raise
            train_step, using_compiled_train_step = _build_train_step(
                model,
                optimizer,
                train_config.grad_clip,
                compile_train_step=False,
            )
            if not compile_failure_reported:
                service_log(
                    log_component,
                    "compiled train step failed; falling back to eager",
                    level="warning",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                compile_failure_reported = True
            loss = train_step(x, y)
        mx.eval(model.parameters(), optimizer.state)

        current = loss.item()
        losses.append(current)
        if current < best:
            best = current

        if on_step is not None:
            on_step(step, model, losses)

        if step % train_config.log_every == 0:
            recent = float(np.mean(losses[-train_config.log_every :]))
            now = time.time()
            elapsed = now - start
            interval_steps = step - last_log_step
            interval_elapsed = now - last_log_time
            extra = (
                on_log(step, recent, best, elapsed, interval_steps, interval_elapsed)
                if on_log is not None
                else None
            )
            if extra:
                service_log(
                    log_component,
                    "training progress",
                    step=step,
                    loss=round(recent, 6),
                    best=round(best, 6),
                    extra=extra,
                )
            else:
                speed = (step * train_config.batch_size * train_config.seq_len) / max(elapsed, 1e-9)
                service_log(
                    log_component,
                    "training progress",
                    step=step,
                    loss=round(recent, 6),
                    best=round(best, 6),
                    tokens_per_second=round(speed, 3),
                )
            last_log_time = now
            last_log_step = step

    elapsed = time.time() - start
    train_loss = evaluate(
        model,
        dataset,
        train_config,
        "train",
        eval_batches=final_eval_batches,
        compiled_loss=compiled_eval_loss,
    )
    test_loss = evaluate(
        model,
        dataset,
        train_config,
        "test",
        eval_batches=final_eval_batches,
        compiled_loss=compiled_eval_loss,
    )
    return RunMetrics(
        seed=seed,
        params=params,
        train_loss=train_loss,
        test_loss=test_loss,
        overfit_pct=(test_loss / train_loss - 1.0) * 100.0,
        train_time_sec=elapsed,
    )
