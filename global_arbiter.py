import logging

from abc import ABC
from typing import List, Tuple
import pandas as pd

from application import Application
from model import ModelParallelism
import application_repo
from simulator import clock, schedule_event, cancel_event, reschedule_event
import start_state_repo
import utils
from long_term_allocation import MilpLongTermAllocation
from forecasting import ForecastEnsembler


class GlobalArbiter(ABC):
    """
    Global Arbiter allocates Models to Regions from the global router.
    """
    def __init__(self,
                 long_term_scaling_interval,
                 max_time):
        self.regions = 3
        self.models = 2
        self.model_to_idx_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.model_list = ["A", "B", "C", "D"]

        self.long_term_scaling_interval = long_term_scaling_interval
        self.last_updated_time = -1e9
        self.last_reported_allocation_time = 0
        self.allocation_interval = 600

        self.max_time = max_time

    def scaling_interval_reached(self) -> bool:
        return (clock() >= self.last_updated_time + self.long_term_scaling_interval)
    
class MilpGlobalArbiter(GlobalArbiter):
    """MILP based global router"""
    def __init__(self,
                 arima_traces,
                 long_term_scaling_interval,
                 post_processing_strategy,
                 max_time,
                 arima_aware_arbiter):
        super().__init__(long_term_scaling_interval, max_time)
        self.arima_traces = arima_traces
        self.forecast_df = {}
        self.dev_df = {}
        self.region_clusters = None
        self.region_routers = None
        self.milp_allocator = None
        self.post_processing_strategy = post_processing_strategy
        self.num_instances_log = None
        self.arima_aware_arbiter = arima_aware_arbiter
        self.ensembler = ForecastEnsembler()
    
    def add_region_routers(self, region_routers) -> None:
        self.region_routers = region_routers
    
    def add_region_clusters(self, region_clusters) -> None:
        self.region_clusters = region_clusters

    def save_results(self):
        utils.save_dict_as_csv(self.num_instances_log, "global_ariber_logs.csv")
    
    def load_predicted_traces(self) -> None:
        self.regions = len(self.region_clusters)
        unique_models = []
        for region_id in self.region_clusters.keys():
            for model_name in self.region_routers[region_id].model_endpoint_routers:
                unique_models.append(model_name)
                filepath = f"{self.arima_traces}/final_{model_name}_prod_1minute_region{region_id}_arima.csv"
                dev_file = f"{self.arima_traces}/final_{model_name}_dev_region{region_id}_exact.csv"
                print(filepath)
                print(dev_file)
                if region_id not in self.forecast_df:
                    self.forecast_df[region_id] = {}
                if region_id not in self.dev_df:
                    self.dev_df[region_id] = {}
                self.forecast_df[region_id][model_name] = pd.read_csv(filepath)
                self.dev_df[region_id][model_name] = pd.read_csv(dev_file)
                print(region_id, model_name, len(self.forecast_df[region_id][model_name]))
                print(region_id, model_name, len(self.dev_df[region_id][model_name]))
                if self.arima_aware_arbiter:
                    self.region_clusters[region_id].arbiter.set_arima_forecast(self.forecast_df[region_id][model_name])

        unique_models = sorted(list(set(unique_models)))
        # unique_models = ["A", "B", "C", "D"]
        self.models = len(unique_models)
        self.model_list = unique_models
        for i in range(self.models):
            self.model_to_idx_mapping[unique_models[i]] = i
        self.milp_allocator = MilpLongTermAllocation(
            models=self.models,
            regions=self.regions,
            gpus=1,
            model_interchange_time=[[150] for _ in range(self.models)],
            # model_tps=[[60*11000], [60*8000], [60*1600000], [60*1600000]],
            model_tps=[[30*7516.67], [30*5398.07], [30*1600000], [30*1600000]],
            gpu_cost=[10]
        )
        self.num_instances_log = {f"region_{i}_model_{j}": [] for i in range(self.regions) for j in range(self.models)}
        self.num_instances_log["time"] = []
    def current_allocations(self) -> List[List[List[int]]]:
        ca = {}
        for region_id, region_router in self.region_routers.items():
            cur_region = [[0] for _ in range(self.models)]
            for model_name in region_router.model_endpoint_routers.keys():
                val = region_router.model_endpoint_routers[model_name].total_instances
                if val == None:
                    val = 0
                cur_region[self.model_to_idx_mapping[model_name]][0] += val
            for model_name in region_router.model_endpoint_routers.keys(): 
                self.num_instances_log[f"region_{region_id}_model_{self.model_to_idx_mapping[model_name]}"].append(cur_region[self.model_to_idx_mapping[model_name]][0])
            ca[region_id] = cur_region
            if ca[region_id] == None:
                ca[region_id] = 0
        self.num_instances_log["time"].append(clock())
        return [ca[k] for k in sorted(ca.keys())]
    
    def start_with_8_each(self) -> None:
        ca = self.current_allocations()
        actions = []
        for i in range(self.regions):
            for j in range(self.models):
                for k in range(1):
                    if ca[i][j][k] <= 8:
                        continue
                    while ca[i][j][k] > 8:
                        actions.append((False, 0, i, j, k))
                        ca[i][j][k] -= 1
        self.schedul_scaling_events(actions)
        logging.info(f"Start state: {self.current_allocations()}")

    def get_ilp_forecast(self, path) -> List[List[List[int]]]:
        cur_time = clock()
        # get current allocation
        current_allocation: List[List[List[int]]] = self.current_allocations()
        current_allocation_transposed = [[[0] for _ in range(self.regions)] for __ in range(self.models)]
        for i in range(self.models):
            for j in range(self.regions):
                for k in range(1):
                    current_allocation_transposed[i][j][k] = current_allocation[j][i][k]
        current_allocation = current_allocation_transposed
        
        # get forecast for 1 hour head
        tokens_forecast = {}
        for region_id, region_router in self.region_routers.items():
            cur_region = [[0] for _ in range(self.models)]
            for model_name in region_router.model_endpoint_routers.keys():
                df = self.forecast_df[region_id][model_name]
                forecast_val, model_outputs = self.ensembler.forecast_window(
                    df,
                    cur_time=cur_time,
                    window_start=cur_time + 20 * 60,
                    window_end=cur_time + 60 * 60,
                )
                cur_region[self.model_to_idx_mapping[model_name]][0] += forecast_val
                logging.info(f"ensemble forecast {region_id} {model_name}: {forecast_val} ({[(m.name, round(m.value, 2)) for m in model_outputs]})")
            tokens_forecast[region_id] = cur_region
        # add dev demand from 12hours ago
        for region_id, region_router in self.region_routers.items():
            cur_region = [[0] for _ in range(self.models)]
            for model_name in region_router.model_endpoint_routers.keys():
                df = self.dev_df[region_id][model_name]
                if cur_time < 3600:
                    df = df.loc[df["arrival_timestamp"] <= 3600]
                else:
                    df = df.loc[df["arrival_timestamp"] <= cur_time]
                    df = df.loc[df["arrival_timestamp"] >= cur_time - 60 * 60]
                if len(df["prompt_size"]) > 0:
                    tokens_forecast[region_id][self.model_to_idx_mapping[model_name]][0] += 0.1 * df["prompt_size"].max()
                    logging.info(f"dev sum {region_id} {model_name}: {0.1 * df['prompt_size'].max()}")
        logging.info(tokens_forecast)
        tokens_forecast_final = [[[0] for _ in range(self.regions)] for _ in range(self.models)]
        for i in range(self.models):
            for j in range(self.regions):
                tokens_forecast_final[i][j][0] = tokens_forecast[j][i][0]

        # tokens_forecast = [tokens_forecast[k] for k in sorted(tokens_forecast.keys())]
        # tokens_forecast_transpose = [[[0] for _ in range(self.regions)] for __ in range(self.models)]
        # for i in range(self.models):
        #     for j in range(self.regions):
        #         tokens_forecast_transpose[i][j][0] = tokens_forecast[j][i][0]
        #         if tokens_forecast_transpose[i][j][0] == None:
        #             tokens_forecast_transpose[i][j][0] = 0
        # tokens_forecast = tokens_forecast_transpose
        logging.info(f"Tokens forecast: {tokens_forecast_final}")
        return self.milp_allocator.get_ilp_allocations(current_allocation, tokens_forecast_final, self.arima_traces)

    def post_process_ilp(self, ilp_forecast: List[List[List[int]]]) -> List[Tuple[bool, int, int, int, int]]: 
        # (start action?, action time, region, model, gpu)
        actions = []
        indices = [(i, j, k) for i in range(self.regions) for j in range(self.models) for k in range(1)]
        if self.post_processing_strategy == "immediate":
            for idx in indices:
                if ilp_forecast[idx[0]][idx[1]][idx[2]] < 0:
                    for _ in range(-ilp_forecast[idx[0]][idx[1]][idx[2]]):
                        actions.append((False, 0, idx[0], idx[1], idx[2]))
            for idx in indices:
                if ilp_forecast[idx[0]][idx[1]][idx[2]] > 0:
                    for _ in range(ilp_forecast[idx[0]][idx[1]][idx[2]]):
                        actions.append((True, 0, idx[0], idx[1], idx[2]))
        elif self.post_processing_strategy == "evenly_distributed":
            """we should do the actions spread out even in the time period"""
            raise NotImplementedError(f"ILP post processing strategy {self.post_processing_strategy} not implemented")
        else:
            raise NotImplementedError(f"ILP post processing strategy {self.post_processing_strategy} not implemented")
        return actions

    def schedul_scaling_events(self, changes: List[Tuple[bool, int, int, int, int]]) -> None:
        for e in changes:
            action, delay, region, model, gpu = e
            if clock() + delay >= self.max_time:
                continue
            if action:
                f = lambda self=self, region=region, model=model: self.region_clusters[region].arbiter.force_scale_up(self.region_routers[region].model_endpoint_routers[self.model_list[model]])
                schedule_event(delay, f)
            else:
                f = lambda self=self, region=region, model=model: self.region_clusters[region].arbiter.force_scale_down(self.region_routers[region].model_endpoint_routers[self.model_list[model]])
                schedule_event(delay, f)

    def scale(self) -> None:
        ilp_forecast: List[List[List[int]]] = self.get_ilp_forecast(self.arima_traces)
        ilp_changes = self.post_process_ilp(ilp_forecast)
        logging.info(f"Current allocation at time {clock()}: {self.current_allocations()}")
        logging.info(f"ILP forecast at time {clock()}: {ilp_forecast}")
        for region_cluster in self.region_clusters.values():
            region_cluster.arbiter.reset_changes()
        self.schedul_scaling_events(ilp_changes)
        self.last_updated_time = clock()
