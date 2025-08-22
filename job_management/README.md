## Minimal Job Management Scripts

TPU needs queueing; (1) waiting for TPU to become available for model training and (2) re-entering the queue after TPU get preempted can be tedious if done by hand.

The ``template.sh`` script automates this whole workflow with minimal abstractions. To test it, please modify ``setup.sh`` with your actual environment setup commands, and fill in the job-related information in ``template.sh``. After these updates, you can call:
```
bash template.sh
```