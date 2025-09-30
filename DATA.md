# Data Storage
By default, each Cloud TPU VM has a single 100 GiB boot disk. To store additional data, one can use a cloud storage bucket that will be accessed remotely during model training or a durable block storage attached to the VM. The bucket is more flexible and cheaper; the durable block storage is faster but requires a pre-determined amount and is more expensive.

## (1) Cloud storage buckets
### (1.1) To create a bucket, go to https://console.cloud.google.com/ and search "Buckets" in the search bar. Then, create a bucket in the same zone as your TPU-VM.

### (1.2) To upload data to a bucket, run
```
gsutil -m cp -r /local/path/to/dataset gs://bucket_name/path/to/dataset
```
managing data within and across buckets works similarly: they can be handled with ``gsutil cp``, ``gsutil mv``, etc.

### (1.3) Access buckets in project A from project B's TPU VM:
Go to https://console.cloud.google.com/ and search "Buckets" in the search bar, click into the bucket you're interested in.<br>
Then, go to `Permissions` page, and cick on `Grant access`. In the `New principals` field, put in `projectOwner:name-of-project-b` and set role to `Storage Legacy Object Reader`.

## (2) Durable block storage
### (2.0) Disk has two modes: single-writer mode and read-only mode

Only single-host TPU (e.g., v3-8, v4-8) can have a disk attached in single-writer mode. For multi-host TPU (e.g., v3-32, v4-64), a disk can only be attached in read-only mode.

In read-only mode, formatting a non-boot disk isn't possible. Thus, if you want to use the disk on a multi-host TPU, you should format the non-boot disk first on a single-host machine in single-writer mode first. You don't have to redo the formatting when you attach the disk to your multi-host TPU.

To create a disk, run
```
gcloud compute disks create example-disk --size={10 to 65,536, the unit is GB} --type={e.g., pd-standard}  --zone={e.g., europe-west4-a}
```
According to https://cloud.google.com/compute/docs/disks/add-persistent-disk#gcloud, acceptable sizes range, in 1 GB increments, from 10 GB to 65,536 GB inclusive.

### (2.1) First, on a single-host machine
- Attach the disk to a single-host TPU-VM
```
gcloud alpha compute tpus tpu-vm attach-disk example-tpu --disk example-disk --zone={e.g., europe-west4-a} --mode=read-write
```
`example-tpu` should be replaced with the name of your TPU-VM

- Format a non-boot disk (after logging in to the TPU-VM):
```
ls -l /dev/disk/by-id/google-*
```
This would output something like
```
lrwxrwxrwx 1 root root  9 Aug 13 08:09 /dev/disk/by-id/google-persistent-disk-0 -> ../../sda
lrwxrwxrwx 1 root root 10 Aug 13 08:09 /dev/disk/by-id/google-persistent-disk-0-part1 -> ../../sda1
lrwxrwxrwx 1 root root 11 Aug 13 08:09 /dev/disk/by-id/google-persistent-disk-0-part14 -> ../../sda14
lrwxrwxrwx 1 root root 11 Aug 13 08:09 /dev/disk/by-id/google-persistent-disk-0-part15 -> ../../sda15
lrwxrwxrwx 1 root root  9 Aug 13 08:44 /dev/disk/by-id/google-persistent-disk-1 -> ../../sdb
```
This shows that for the next step, the `DEVICE_NAME` should be `sdb`.
```
sudo mkfs.FILE_SYSTEM_TYPE -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/DEVICE_NAME
```
Normally, FILE_SYSTEM_TYPE should be `ext4` or `xfs`.

- Mount the disk onto an arbitrary location on the VM (after logging in to the TPU-VM)
```
sudo mkdir -p /mnt/disks/MOUNT_DIR
sudo mount -o discard,defaults /dev/DEVICE_NAME /mnt/disks/MOUNT_DIR
sudo chmod a+w /mnt/disks/MOUNT_DIR
```
After mounting, you can save your datasets onto the disk.

### (2.2) Then, on the multi-host machine that you're actually using for e.g. model training
- Attach the disk to a multi-host TPU-VM (from your local terminal)
```
gcloud alpha compute tpus tpu-vm attach-disk example-tpu --disk example-disk --zone={e.g., europe-west4-a} --mode=read-only
```
`example-tpu` should be replaced with the name of your TPU-VM

- Mount the disk (from your local terminal)
```
gcloud compute tpus tpu-vm ssh TPU_NAME --zone={e.g., europe-west4-a} --worker=all --command="sudo mkdir -p /mnt/disks/MOUNT_DIR"
gcloud compute tpus tpu-vm ssh TPU_NAME --zone={e.g., europe-west4-a} --worker=all --command="sudo mount -o ro,noload /dev/sdb /mnt/disks/MOUNT_DIR"
```

### (2.3) Clean up
Detach the disk from the TPU-VM
```
gcloud alpha compute tpus tpu-vm detach-disk example-tpu --disk example-disk --zone={e.g., europe-west4-a}
```
Delete the disk
```
gcloud compute disks delete example-disk --zone={e.g., europe-west4-a}
```
## Other Helpful Resources
https://cloud.google.com/tpu/docs/storage-options
https://cloud.google.com/tpu/docs/attach-durable-block-storage