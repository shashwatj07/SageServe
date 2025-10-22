# LLM Serving Stack Simulator

## Setup

You can set up the simulator by installing its Python dependencies. We recommend starting with a fresh Python environment.

```python
# Create and activate new Python environment
conda create -n sim python=3.11
conda activate sim

# Install dependencies
pip install -r requirements.txt
```

## Inputs and Outputs

SplitwiseSim takes in a hierarchical set of YAML configuration files as input, and it produces several CSV files as output. It uses [Hydra](https://hydra.cc/) for configuration management. You can learn more about configuration management from the [Hydra docs](https://hydra.cc/docs/intro/).

The top-level configuration file for SplitwiseSim is [`config.yaml`](configs/config.yaml), which points to lower-level configurations specified by other files in the `configs/` directory. Specifically, `config.yaml` captures the following key components:

- [cluster](configs/cluster/): the provisioned server SKUs in the cluster, along with their respective counts.
- [trace](#request-traces): request trace that specifies the set of requests that arrive into the cluster.
- [router](configs/router/): the cluster-level router that routes incoming requests to application-level schedulers; currently a no-op.
- [arbiter](configs/arbiter/): the cluster-level arbiter that manages compute resources between applications to support autoscaling; currently a no-op.
- [application](configs/applications/): the logical endpoint that the requests target, which specifies the model and the set of instances on which the request runs; currently, we support only one application.
- [model_repo](configs/model_repo/): the set of models (LLMs) available to run in the cluster; used for dynamic model instantiation.
- [orchestrator_repo](configs/orchestrator_repo/): the set of application resource orchestrators (i.e., schedulers and allocators) in the cluster; used for dynamic application management.
- [hardware_repo](configs/hardware_repo/): the set of available SKUs that can be provisioned in the cluster; used for dynamic server instantiation.
- [performance_model](#performance-model): an analytical model that helps estimate request runtimes with different batch, model, and hardware configurations.
- [start_state](configs/start_state/): starting state for the cluster, which helps simplify evaluation.

Several other aspects can be configured; please see [`config.yaml`](configs/config.yaml) for details.

SplitwiseSim generates the following key outputs:

- Summary of application-level metrics (`summary.csv`)
- Per-request metrics for each completed request for each application (`detailed/{application_id}.csv`)
- Request node-level metrics (`request_nodes.csv`)
- Instance-level execution metrics (in `instances/`, with `debug` enabled)

We provide various [utility functions](notebooks/utils.py) to process outputs, as shown in [`notebooks/example.ipynb`](notebooks/example.ipynb) and [`notebooks/plots.ipynb`](notebooks/plots.ipynb).

## How to run?

Simply modify [`config.yaml`](configs/confid.yaml) to and execute ```python run.py```.

### How to run experiments with separate scaling between prod/dev?

- Use controller us3-dp in config.yaml
- Use annotated traces ES_26_dp.csv in configs/trace/enterprise_sydney.yaml

### How to configure other knobs?

The following knobs are present in [`config.yaml`](configs/confid.yaml) to freely allow changes in execution configuration.

- feed_async: True/False to enable/disable the insertion of async requests whevnever memory utilisation falls below 0.5
- feed_async_granularity: Specify the number of async requests to insert at a time.
- scaling_level: Specify 0 for no scaling, 1 for scaling from/to spot only and 2 for inter model scaling along eith spot donations.
- scaling_interval: The number of seconds to wait between two scaling events per model endpoint. Use -1 to disable this knob, i.e., no restriction on the number of scaling events w.r.t time. 


## Long term scaling
All short term scaling scripts should still run as they are!

### Scaling on 1 hour window
Run with 
```bash
python3 run_kunal.py trace.filename=ES_26 \
    short_term_scaling=False \
    long_term_scaling=True \
    global_arbiter.arima_traces=$PWD/traces/forecasts/ \
    global_arbiter.post_processing_strategy=<STRATEGY>
```
where STRATEGY can be `immediate`, `delay_changes`, `keep_maximum_instances`, `keep_minimum_instances`.

### Reactive Scaling, Proactive Guidance
Run with
```bash
python3 run.py trace.filename=final_data_day_1 short_term_scaling=True long_term_scaling=True global_arbiter.arima_traces=$PWD/traces/forecasts/ controller.regions.0.arbiter=global_arbiter_ARIMA_checking controller.regions.1.arbiter=global_arbiter_ARIMA_checking controller.regions.2.arbiter=global_arbiter_ARIMA_checking global_arbiter.arima_aware_arbiter=True

python3 run.py trace.filename=final_data_day_1 short_term_scaling=True long_term_scaling=True global_arbiter.arima_traces=$PWD/traces/forecasts/ controller.regions.0.arbiter=global_arbiter_memory_utilization controller.regions.1.arbiter=global_arbiter_memory_utilization controller.regions.2.arbiter=global_arbiter_memory_utilization global_arbiter.arima_aware_arbiter=True

python3 run.py trace.filename=final_data_day_1 short_term_scaling=True long_term_scaling=True global_arbiter.arima_traces=$PWD/traces/forecasts/ controller.regions.0.arbiter=global_arbiter_short_term_scaling controller.regions.1.arbiter=global_arbiter_short_term_scaling controller.regions.2.arbiter=global_arbiter_short_term_scaling global_arbiter.arima_aware_arbiter=True
```

## Citation
If you use our work, please consider citing our paper:
```
@article{jaiswal2026sageserve,
    title={SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling},
    author={Jaiswal, Shashwat and Jain, Kunal and Simmhan, Yogesh and Parayil, Anjaly and Mallick, Ankur and Wang, Rujia and St Amant, Renee and Bansal, Chetan and Ruhle, Victor and Kulkarni, Anoop and Kofsky, Steve and Rajmohan, Saravan},
    journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems (POMACS), 9(3), 2025},
    year={2025}
}
```
## Acknowledgement
This repository is a fork of and built on top of the [Splitwise simulator](https://github.com/Mutinifni/splitwise-sim).
