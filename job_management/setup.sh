gcloud alpha compute tpus tpu-vm ssh boyazeng@$TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "git clone https://your_github_personal_access_token@github.com/your_username/your_repo_name && \
mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm ~/miniconda3/miniconda.sh && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda init && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
conda create -n my_env python=3.10 -y && \
conda activate my_env && \
python -m pip install -U pip && \
python -m pip install 'jax[tpu]==0.6.2' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"