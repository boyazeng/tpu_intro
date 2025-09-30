# Instruction for Creating TPU-VM

## (0) Set the project id:
```
gcloud config set project ${id of the project}
```
The id of the project can be retrieved by visiting https://console.cloud.google.com/.

## (1) Requesting TPU-VM:
```
gcloud compute tpus tpu-vm create example_tpu \
  --zone=${e.g., europe-west4-a} \
  --accelerator-type=${tpu type, e.g., v3-8} \
  --version=tpu-ubuntu2204-base \
  --spot
```
to request on demand TPU-VM, remove the "--spot" flag.
TPU software versions (``--version``) should be determined based on the TPU version (e.g., v3, v4, v6e, etc.), see https://cloud.google.com/tpu/docs/runtimes#pytorch_and_jax.

## (2) The TPU-VM request command would fail when there is no available resources. To queue for the resource:
```
# start queueing
gcloud compute tpus queued-resources create example_queue \
  --node-id=example_tpu \
  --zone=${e.g., europe-west4-a} \
  --accelerator-type=${tpu type, e.g., v3-8} \
  --runtime-version=tpu-ubuntu2204-base

# check queueing status
gcloud compute tpus queued-resources describe example_queue \
  --zone=${e.g., europe-west4-a}

# delete queue
gcloud compute tpus queued-resources delete example_queue --zone=${e.g., europe-west4-a}
```
both spot and on-demand resource can be queued.

## (3) To log in to the created TPU-VM:
```
gcloud compute tpus tpu-vm ssh example_tpu \
  --zone=${e.g., europe-west4-a} \
  --ssh-key-file={e.g., ~/.ssh/google_compute_engine} \
  --worker=0
```

## (4) To delete the TPU-VM:
```
gcloud compute tpus tpu-vm delete example_tpu \
  --zone=${e.g., europe-west4-a}
```

## (5) To use the TPU-VM with VS Code / Cursor:
Run the following to obtain the external IP of the TPU-VM:
```
gcloud compute tpus tpu-vm describe example_tpu \
  --zone=${e.g., europe-west4-a}
```

Once you have the external IP, add the following to your ssh config:
```
Host the_name_does_not_matter
  User your_username
  Hostname ${the external IP of the TPU-VM}
  IdentityFile /path/to/your/ssh/file
```