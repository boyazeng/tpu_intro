# Structure of TPUs


## (1) Configuration
|TPU version|#core per chip|#process per chip|TPU memory per process|#chips per worker (`vk-8` and above)|
|-|-|-|-|-|
|v3|2|2|16 GB|4|
|v4|2|1|32 GB|4|
|v5e|1|1|16 GB|4|
|v6e|1|1|32 GB|4|

One worker uses one external IP address. The usage of external IP address is relevant for the in-use external IP address quota.

## (2) Terminology
`vk-n`: a slice of *n* cores of TPU version *k*. For example, v4-32 is a slice of 32 cores of TPU version 4.<br>
`device/slice`: a single v3-8 / v4-8 / etc. is a device, whereas v4-16 / v4-64 / etc. is a (pod) slice. Note that `device` can also have other meanings in different contexts, e.g., `len(jax.devices())` gives the number of parallel TPU workers.<br>
`multislice`: as the name suggests, can be used to run a job on multiple separate slices.

## (3) Speed (Credit: [Taiming Lu](https://taiminglu.com/))
<p align="center">
<img src="./static/images/speed_llama.jpg" width=90% height=90% 
class="center">
</p>

## Other Helpful Resources
https://github.com/jax-ml/jax/discussions/19927.