import os
import time
from typing import Tuple
from absl import logging
from absl.flags.argparse_flags import ArgumentParser

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import numpy as np
import optax
import functools
from tqdm import tqdm

import flax
from flax import linen as nn
from flax.jax_utils import replicate
from flax.training.common_utils import shard

import tensorflow as tf
import tensorflow_datasets as tfds

class MLP(nn.Module):
    hidden_sizes: Tuple[int, ...] = (1024, 512)
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

def build_dataset(split: str,
                  per_proc_batch_size: int,
                  training: bool,
                  seed: int,
                  process_index: int,
                  process_count: int):
    ds = tfds.load('mnist', split=split)
    ds = ds.shard(num_shards=process_count, index=process_index)

    def preprocess(sample_dict):
        image = tf.cast(sample_dict["image"], tf.float32) / 255.0
        image = tf.reshape(image, [28 * 28])
        label = tf.cast(sample_dict["label"], tf.int32)
        return {"image": image, "label": label}

    if training:
        ds = ds.shuffle(60_000, seed=seed, reshuffle_each_iteration=True).repeat()
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(per_proc_batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()

def main(args):
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)
    jax.distributed.initialize()
    tf.config.experimental.set_visible_devices([], "GPU")
    logging.info(f"\u001b[33mProcess {jax.process_index()} holds {jax.local_device_count()}/{jax.device_count()} devices\u001b[0m")

    local_device_count = jax.local_device_count()
    if args.per_proc_batch_size % local_device_count != 0:
        raise ValueError(
            f"--per-proc-batch-size ({args.per_proc_batch_size}) must be divisible by "
            f"jax.local_device_count() ({local_device_count})."
        )

    rng = jax.random.PRNGKey(args.seed)
    model = MLP(hidden_sizes=tuple(args.hidden_sizes))
    tx = optax.adamw(args.learning_rate)

    @functools.partial(jax.jit, backend="cpu")
    def init_params(rng):
        x = jnp.zeros((2, 28 * 28), jnp.float32)
        params = flax.core.unfreeze(model.init(rng, x))["params"]
        return params

    rng, rng_init = jax.random.split(rng)
    params_cpu = init_params({"params": rng_init})
    opt_state_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)

    params_repl = flax.jax_utils.replicate(params_cpu)
    opt_state_repl = flax.jax_utils.replicate(opt_state_cpu)

    @functools.partial(jax.pmap, axis_name="batch")
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            logits = model.apply({"params": p}, batch["image"])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["label"]
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        loss = jax.lax.pmean(loss, axis_name="batch")
        grads = jax.lax.pmean(grads, axis_name="batch")

        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @functools.partial(jax.pmap, axis_name="batch")
    def eval_step(params, batch):
        logits = model.apply({"params": params}, batch["image"])
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == batch["label"]).sum()
        total = jnp.array(batch["label"].shape[0], dtype=jnp.int32)

        correct = jax.lax.psum(correct, axis_name="batch")
        total = jax.lax.psum(total, axis_name="batch")
        return correct, total

    process_index = jax.process_index()
    process_count = jax.process_count()

    train_iter = build_dataset(
        split="train",
        per_proc_batch_size=args.per_proc_batch_size,
        training=True,
        seed=args.seed,
        process_index=process_index,
        process_count=process_count,
    )

    MNIST_TRAIN = 60_000
    MNIST_TEST = 10_000
    steps_per_epoch = (MNIST_TRAIN // process_count) // args.per_proc_batch_size
    eval_steps = max(1, (MNIST_TEST // process_count) // args.per_proc_batch_size)

    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{args.num_epochs}") if jax.process_index() == 0 else None

        epoch_loss = 0.0
        t0 = time.time()
        for _ in range(steps_per_epoch):
            batch = next(train_iter)
            batch = shard(batch)
            params_repl, opt_state_repl, loss = train_step(params_repl, opt_state_repl, batch)
            loss = jax.device_get(loss)[0]
            epoch_loss += float(loss)
            global_step += 1
            if pbar is not None:
                pbar.set_postfix(loss=f"{float(loss):.4f}")
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        multihost_utils.sync_global_devices(f"epoch_{epoch}_train_done")

        correct_total = 0
        count_total = 0
        test_iter = build_dataset(
            split="test",
            per_proc_batch_size=args.per_proc_batch_size,
            training=False,
            seed=args.seed,
            process_index=process_index,
            process_count=process_count,
        )
        for _ in range(eval_steps):
            batch = next(test_iter)
            batch = shard(batch)
            correct, total = eval_step(params_repl, batch)
            correct_total += int(jax.device_get(correct)[0])
            count_total += int(jax.device_get(total)[0])

        acc = correct_total / max(1, count_total)
        ep_time = time.time() - t0

        if jax.process_index() == 0:
            logging.info(
                f"[Epoch {epoch}] loss={epoch_loss/steps_per_epoch:.4f} "
                f"acc={acc*100:.2f}% steps={global_step} time={ep_time:.1f}s"
            )

        multihost_utils.sync_global_devices(f"epoch_{epoch}_done")

    multihost_utils.sync_global_devices("training_complete")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--per-proc-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)