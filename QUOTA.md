# Understanding the Effective TPU Quota

The TPU quota can be retrieved at https://console.cloud.google.com/iam-admin/quotas.

## Resource type

**Preemptible resources** are resources that the user may lose access to at any time and has a 24 hour run time limit.<br>
https://cloud.google.com/tpu/docs/preemptible

**Spot resources** is a newer version of preemptible resources, and they share the same quotas. Generally, there's no reason to choose preemptible over spot, since spot VM can be queued while preemptible VM cannot, and spot VM can last for > 1 days, while preemptible VM cannot.

**On-demand resources** can be held by the user for as long as they wish and would not be preempted.<br>

## Estimating effective quota

The number of usable TPU cores is mainly limited by the following factors:<br>
(1) Maximum number of in-use external IP addresses according quota<br>
Normally, this number is 8 for every zone.<br>
(2) Maximum number of cores in the zone (e.g., europe-west4-a) according to quota<br>
Note that this number may not be shared between pod cores (e.g., v4-16 or above) and cores (e.g., v4-8), and may not be shared between preemptible / spot resources and on-demand resources.<br>
(3) Availability of TPUs in the zone (e.g., europe-west4-a)

## Usable TPU cores for each region and TPU version
Please request access to our group's internal spreadsheet at https://docs.google.com/spreadsheets/d/1X9GEXr0iJ9WM2GlpkwXFaLX04rYKyn-s-WaQisGpbnk/edit?usp=sharing.