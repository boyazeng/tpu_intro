# tpu_intro

**Disclaimer: this repository is intended for internal use within Zhuang Liu's Group at Princeton and is not purposed as a general guide.**

## Contact

For any question regarding TPU usage, please add **and** message Boya Zeng on Slack.

## Structure of This Repository

[QUICK_START.md](QUICK_START.md): commands that can be ran directly to perform a simple MNIST training.<br>
[quick_start.py](quick_start.py): a standalone file for the MNIST training in JAX on TPU.<br>
[requirements.txt](requirements.txt): environment setup for MNIST training.<br>
[job_management](job_management): scripts for automated queueing of TPU resources.

[CREATION.md](CREATION.md): commands for managing TPU-VMs.<br>
[USAGE.md](USAGE.md): handy commands in practical TPU usage.<br>
[TPU.md](TPU.md): general info for understanding different versions of TPU.<br>
[QUOTA.md](QUOTA.md): guide for understanding the effective TPU quota.<br>
[DATA.md](DATA.md): instructions on storing data for usage on TPUs.<br>
[COST.md](COST.md): information on what can incur cost and how much the costs are.<br>
[MONITORING.md](MONITORING.md): instructions on monitoring TPU usage.

## Important Note on TPU/Bucket Naming

To facilitate usage monitoring within our group, please follow this naming convention when creating TPUs and buckets:
- TPU: please ensure that it starts with "{your name or any fixed prefix}-{the TPU version}-", e.g., "boya-V3-my-tpu".
- bucket: please ensure that it starts with your name or any fixed prefix.

## Other Helpful Resources
https://cloud.google.com/tpu/docs/intro-to-tpu<br>
https://github.com/ayaka14732/tpu-starter<br>
https://jax-ml.github.io/scaling-book/