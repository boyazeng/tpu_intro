# Understanding the Effective TPU Quota

The TPU quota can be retrieved at https://console.cloud.google.com/iam-admin/quotas.

## Resource type

**Preemptible resources** are resources that the user may lose access to at any time and has a 24 hour run time limit.<br>
https://cloud.google.com/tpu/docs/preemptible

**Spot resources** is a newer version of `preemptible` resources, and they share the same quotas. Generally, there's no reason to choose `preemptible` over `spot`, since `spot` VM can be queued while `preemptible` VM cannot, and `spot` VM can last for > 1 days, while `preemptible` VM cannot.

**On-demand resources** can be held by the user for as long as they wish and would not be preempted.

## Rule of thumb for estimating effective quota

The number of usable TPU cores is mainly limited by the following factors:<br>
(1) Maximum number of in-use external IP addresses according to quota<br>
By default, this number is 8 for every zone. This can be raised by contacting the TPU Research Cloud team.<br>
(2) Maximum number of cores in the zone (e.g., europe-west4-a) according to quota<br>
Note that this number is shared between pod cores (e.g., v4-16 or above) and cores (e.g., v4-8) or between `preemptible` / `spot` resources and `on-demand` resources.<br>
(3) Maximum number of VM instances in the region (e.g., europe-west4) according to quota<br>
Each host (8 cores) uses one VM instance. For instance, a single v4-64 uses 8 VM instances.<br>
(4) Persistent Disk Standard (GB) for v4 TPU / Hyperdisk Balanced Capacity (GB) for v6 TPU in a region (e.g., europe-west4) according to quota<br>
By default, each host has 100 GB of local storage space. The total VM local storage used across all VMs in a region cannot exceed the limit.<br>
(5) Availability of TPUs in the zone (e.g., europe-west4-a)

## Checking quotas on command line

### For TPU related quota
```
gcloud beta quotas info describe {the resource you're interested in} --service=tpu.googleapis.com --project={project name}
```
"the resource you're interested in" can be (a non-complete list):<br>
`TPUV5sLitepodPerProjectPerRegionForTPUAPI`<br>
`TPUV5sLitepodPerProjectPerZoneForTPUAPI`<br>
`TPUV5sPreemptibleLitepodPerProjectPerRegionForTPUAPI`<br>
`TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI`

`TPUV6EPerProjectPerRegionForTPUAPI`<br>
`TPUV6EPerProjectPerZoneForTPUAPI`<br>
`TPUV6EPreemptiblePerProjectPerRegionForTPUAPI`<br>
`TPUV6EPreemptiblePerProjectPerZoneForTPUAPI`

## Usable TPU cores for each region and TPU version
Please request access to our group's internal spreadsheet at https://docs.google.com/spreadsheets/d/1X9GEXr0iJ9WM2GlpkwXFaLX04rYKyn-s-WaQisGpbnk/edit?usp=sharing.

## All locations in Google Cloud
See https://cloud.google.com/compute/docs/regions-zones.

`africa`: `south1`<br>
`asia`: `east1, east2, northeast1, northeast2, northeast3, south1, south2, southeast1, southeast2`<br>
`australia`: `southeast1, southeast2`<br>
`europe`: `central2, north1, north2, southwest1, west1, west2, west3, west4, west6, west8, west9, west10, west12`<br>
`me`: `central1, central2, west1`<br>
`northamerica`: `northeast1, northeast2, south1`<br>
`southamerica`: `east1, west1`<br>
`us`: `central1, east1, east4, east5, south1, west1, west2, west3, west4`