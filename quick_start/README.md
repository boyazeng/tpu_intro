# An End-to-End Example of Model Training Workflow

You can follow the instructions below to launch a TPU-based MNIST training in JAX.<br>
The [JAX code](main.py) is explained in [quick_start_explained.pdf](quick_start_explained.pdf).<br>
https://kidger.site/thoughts/torch2jax/ also provides a nice intro to JAX.

## (0) Install the Google Cloud CLI
Please follow the instructions on https://cloud.google.com/sdk/docs/install.

## (1) Request a TPU-VM
```
gcloud config set project ${id of the project}
gcloud compute tpus tpu-vm create example_tpu \
  --zone=europe-west4-a \
  --accelerator-type=v3-32 \
  --version=tpu-ubuntu2204-base
```

## (2) Install miniconda
```
gcloud alpha compute tpus tpu-vm ssh example_tpu \
--zone=europe-west4-a \
--ssh-key-file=~/.ssh/google_compute_engine \
--worker=all \
--command "mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm ~/miniconda3/miniconda.sh && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda init"
```

## (3) Install environment
```
gcloud alpha compute tpus tpu-vm ssh example_tpu \
--zone=europe-west4-a \
--ssh-key-file=~/.ssh/google_compute_engine \
--worker=all \
--command "source ~/miniconda3/etc/profile.d/conda.sh && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
conda create -n mnist_env python=3.8 -y && \
conda activate mnist_env && \
python -m pip install -U pip && \
python -m pip install "jax[tpu]==0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd tpu_intro/quick_start && \
python -m pip install -r requirements.txt"
```

## (4) Clone this codebase onto the TPU-VM
```
gcloud alpha compute tpus tpu-vm ssh example_tpu \
--zone=europe-west4-a \
--ssh-key-file=~/.ssh/google_compute_engine \
--worker=all \
--command "git clone https://github.com/boyazeng/tpu_intro"
```

## (5) Training using multiple workers
```
gcloud alpha compute tpus tpu-vm ssh example_tpu \
--zone=europe-west4-a \
--ssh-key-file=~/.ssh/google_compute_engine \
--worker=all \
--command "cd tpu_intro/quick_start && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate mnist_env && \
python main.py"
```

## (6) Remove the requested TPU-VM
```
gcloud compute tpus tpu-vm delete example_tpu --zone=europe-west4-a
```