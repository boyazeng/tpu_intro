# tpu_intro

**Disclaimer: this repository is intended for internal use within Professor Zhuang Liu's Group at Princeton and is not purposed as a general guide.**

## (1) Contact

For any question regarding TPU usage, please message Boya Zeng on Slack.

## (2) Structure of This Repository

It is suggested to follow the order below to go through this repository.

### (2.1) An easy, self-contained, hands-on example for TPU-based JAX training
[quick_start](quick_start): simple instructions for performing MNIST training.

### (2.2) Documentations
[CREATION.md](CREATION.md): commands for managing TPU-VMs.<br>
[USEFUL_COMMANDS.md](USEFUL_COMMANDS.md): handy commands in practical TPU usage.<br>
[TPU.md](TPU.md): general info for understanding different versions of TPU.<br>
[QUOTA.md](QUOTA.md): guide for understanding the effective TPU quota.<br>
[DATA.md](DATA.md): instructions on storing data for usage on TPUs.<br>
[COST.md](COST.md): information on what can incur cost and how much the costs are.<br>
[MONITORING.md](MONITORING.md): instructions on monitoring TPU usage.

### (2.3) Guide for job management in practical usage
[job_management](job_management): instructions for automated queueing of TPU resources.

## (3) Important Note on TPU/Bucket Naming

To facilitate usage monitoring within our group, please follow this naming convention when creating TPUs and buckets:
- TPU: please ensure that it starts with "{your name or any fixed prefix}-{the TPU version}-", e.g., "boya-V3-my-tpu".
- bucket: please ensure that it starts with your name or any fixed prefix.

## (4) Other Helpful Resources
https://cloud.google.com/tpu/docs/intro-to-tpu<br>
https://github.com/ayaka14732/tpu-starter<br>
https://jax-ml.github.io/scaling-book/