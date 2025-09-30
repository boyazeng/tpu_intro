# Handy commands in practical usage

## Run commands concurrently on all workers

A single TPU-VM can have multiple workers, and SSH'ing into a single TPU-VM worker only allows the user to make local changes to that single TPU-VM. Thus, to run commands concurrently on all workers, we need to run the following on the terminal of your local machine:
```
gcloud alpha compute tpus tpu-vm ssh example_tpu \
  --zone=${e.g., europe-west4-a} \
  --ssh-key-file={e.g., ~/.ssh/google_compute_engine} \
  --worker=all \
  --command "the command you want to run"
```
the output of the command will be propagated to the terminal of your local machine.

## Scp files into the VM
```
gcloud alpha compute tpus tpu-vm scp --recurse /path/on/local/machine example_tpu:/path/on/tpu/vm \
  --zone=${e.g., europe-west4-a} \
  --worker=all
```

## Obtain the latest version of your repository

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "
if [ ! -d \"your_repo_name\" ] || [ ! -f \"your_repo_name/.git/config\" ]; then \
    echo 'Cloning repository your_repo_name...' && \
    git clone https://your_github_personal_access_token@github.com/your_username/your_repo_name; \
else \
    echo 'Repository your_repo_name already exists.' && \
    cd your_repo_name && git pull; \
fi"
```

## Install miniconda

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm ~/miniconda3/miniconda.sh && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda init"
```

## Kill the current running process

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f python3"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f python"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f jax"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs /tmp/xrt_server_log/"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo lsof -w /dev/accel0 | xargs -r -I {} -d ' ' sudo kill -9 {}"
```
Note that a JAX-related command launched immediately after killing the previous process can likely run into error (the error message would explicitly mention that "Terminating process because the JAX distributed service detected fatal errors. This most likely indicates that another task died; see the other task logs for more details."). Waiting for a few minutes would solve this issue.

## Clean up ``/var/log``
When you get something similar to
```
FileNotFoundError: [Errno 2] No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/home/your_username/your_repository']
```
it often means that the storage is full on some workers.

The storage issue could be caused by large log files. This command can clean up the log files, following https://askubuntu.com/questions/515146/very-large-log-files-what-should-i-do/515151#515151.

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "cd /var/log && sudo truncate -s 0 lastlog wtmp dpkg.log kern.log syslog"
```

## Clean up local command log
``gcloud`` writes per-command execution logs to ``~/.config/gcloud/logs/``. On servers like Della that has a file count limit (e.g., 2M) under ``/home``, the logs can quickly use up the quota. In such cases, you can clean up by simply running
```
rm -rf ~/.config/gcloud/logs/*
```