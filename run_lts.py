import logging
import os
import random
import sys

import hydra

from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from global_router import GlobalRouter
from simulator import TraceSimulator
from initialize import *

from datetime import datetime

# register custom hydra resolver
OmegaConf.register_new_resolver("eval", eval)

import utils

def run_simulation(cfg):
    cluster_repo = init_cluster_repo(cfg)
    hardware_repo = init_hardware_repo(cfg)
    model_repo = init_model_repo(cfg)
    orchestrator_repo = init_orchestrator_repo(cfg)
    region_repo = init_region_repo(cfg)
    arbiter_repo = init_arbiter_repo(cfg)
    model_endpoint_repo = init_model_endpoint_repo(cfg)
    application_repo = init_application_repo(cfg)
    start_state_repo = init_start_state_repo(cfg)
    performance_model = init_performance_model(cfg)
    power_model = init_power_model(cfg)
    controller = init_controller(cfg)
    global_router:GlobalRouter = init_global_router(cfg, controller)
    controller.set_global_router(global_router)
    trace = init_trace(cfg)
    sim = TraceSimulator(trace=trace, end_time=cfg.end_time, debug=cfg.debug)
    regions, region_clusters, model_endpoint_routers, applications = init_regions(cfg, controller)
    for region in regions.values():
        global_router.add_region(region)
        controller.add_region(region)
    global_arbiter = init_global_arbiter(cfg)
    global_router.add_global_arbiter(global_arbiter)
    sim.add_controller(controller)
    sim.add_global_router(global_router)
    sim.add_region_clusters(region_clusters)
    sim.add_regions(regions)
    sim.add_model_endpoint_routers(model_endpoint_routers)
    sim.add_applications(applications)
    sim.load_trace()
    sim.run()

    utils.save_dict_as_yaml(cfg, f"{cfg.output_dir}/config.yaml")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    # print config
    # print(OmegaConf.to_yaml(cfg, resolve=False))
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(OmegaConf.to_yaml(hydra_cfg, resolve=False))

    # initialize random number generator
    print(cfg)
    random.seed(cfg.seed)
    # delete existing oom.csv if any
    if os.path.exists("oom.csv"):
        os.remove("oom.csv")
    if os.path.exists(cfg.output_dir):
        os.remove(cfg.output_dir)

    run_simulation(cfg)


if __name__ == "__main__":
    now = datetime.now()
    sys.argv.append(f"output_dir=results/{now.year}_{now.month}_{now.day}/{now.hour}_{now.minute}_{now.second}/")
    run()