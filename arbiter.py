import logging

from abc import ABC

from application import Application
from model import ModelParallelism
import application_repo
from simulator import clock, schedule_event, cancel_event, reschedule_event
import start_state_repo
import utils

from queue import Queue

class Arbiter(ABC):
    """
    Arbiter allocates Processors to Application Allocators.
    It can be used to support application autoscaling.
    """
    def __init__(self,
                 cluster,
                 overheads,
                 **kwargs):
        self.cluster = cluster
        self.overheads = overheads
        self.servers = cluster.servers
        self.applications = []
        self.allocators = {}
        self.results = {
            "timestamp":[],
            "model":[],
            "prod":[],
            "spot":[],
            "memory_util":[]
        }
        self.last_log_iter = -1
        self.scaling_in_progress = {}
        self.next_scale_time = {}
        self.changes = {}

    def add_application(self, application):
        self.applications.append(application)
        self.allocators[application.application_id] = application.allocator
    
    def reset_changes(self):
        for model_endpoint in self.changes.keys():
            self.changes[model_endpoint] = 0


    def save_results_intermediate(self, write_itr): 
        utils.save_dict_as_csv(self.results, f"arbiters/{self.cluster.region.region_name}/{write_itr}.csv")
        self.last_log_iter = write_itr
        self.results = {
            "timestamp":[],
            "model":[],
            "prod":[],
            "spot":[],
            "memory_util": []
        }

    def save_results(self):
        utils.save_dict_as_csv(self.results, f"arbiters/{self.cluster.region.region_name}/{self.last_log_iter + 1}.csv")

    def run(self):
        pass

    def allocate(self, processors, application):
        """
        Allocates processors to the application.
        """
        pass

    def deallocate(self, processors, application):
        """
        Deallocates processors from the application.
        """
        pass

    def scale(self, model_endpoint):
        """
        Scales the application based on memory utilisation.
        """
        pass


class NoOpArbiter(Arbiter):
    """
    No-op Arbiter.
    """
    pass


class BasicArbiter(Arbiter):
    """
    Basic Arbiter.
    """
    def allocate(self, processors, application):
        """
        Allocates processors to the application.
        """
        pass

    def deallocate(self, processors, application):
        """
        Deallocates processors from the application.
        """
        pass

    def scale_up_from_spot(self, model_endpoint):
        if len(model_endpoint.applications[-1].instances) == 3:
            recent_application = model_endpoint.applications[-1]
            old_instance = recent_application.instances[-1]
            recent_application.remove_instance(old_instance)
            # start new application with old instance and reclaim spot instance
            new_application = Application.from_config(application_repo.get_application_cfg(model_endpoint.model_name),
                                    application_id=recent_application.application_id + 1,
                                    cluster=model_endpoint.region.region_cluster,
                                    region=model_endpoint.region,
                                    router=model_endpoint,
                                    arbiter=None,
                                    feed_async=recent_application.feed_async,
                                    feed_async_granularity=recent_application.feed_async_granularity)
            new_application.add_instance(old_instance) # add old instance
            old_instance.application = new_application
            new_application.allocator.start_reclaim_spot_instance(model_endpoint.model_name) # reclaim spot instance
            model_endpoint.add_application(new_application)
        elif len(model_endpoint.applications) > 1 \
                and len(model_endpoint.applications[-1].instances) == 2 \
                and len(model_endpoint.applications[-2].instances) == 2:
            recent_application = model_endpoint.applications[-2]
            assert len(recent_application.instances) == 2
            recent_application.allocator.start_reclaim_spot_instance(model_endpoint.model_name)
        elif (len(model_endpoint.applications) > 1 \
                and len(model_endpoint.applications[-1].instances) == 2 \
                and len(model_endpoint.applications[-2].instances) == 3) \
                or (len(model_endpoint.applications) == 1 \
                    and len(model_endpoint.applications[-1].instances) == 2):
            recent_application = model_endpoint.applications[-1]
            assert len(recent_application.instances) == 2
            recent_application.allocator.start_reclaim_spot_instance(model_endpoint.model_name)
    
    def spin_up_new_instance(self, model_endpoint, application, processors):
        start_state_cfg = start_state_repo.get_start_state_cfg(model_endpoint.start_state)
        instance_cfg = start_state_cfg.instance
        parallelism = ModelParallelism(pipeline_parallelism=instance_cfg.pipeline_parallelism,
                                tensor_parallelism=instance_cfg.tensor_parallelism)
        application.allocator.start_spin_up_instance(instance_cfg=instance_cfg,
                                    processors=processors,
                                    parallelism=parallelism,
                                    pre_start=True)

    def scale_up_from_spot_other(self, model_endpoint, processors):
        if len(model_endpoint.applications[-1].instances) == 3:
            recent_application = model_endpoint.applications[-1]
            old_instance = recent_application.instances[-1]
            recent_application.remove_instance(old_instance)
            # start new application with old instance and reclaim spot instance
            # new_application = Application.new_application(recent_application.application_id + 1, model_endpoint.model_name)
            new_application = Application.from_config(application_repo.get_application_cfg(model_endpoint.model_name),
                                    application_id=recent_application.application_id + 1,
                                    cluster=model_endpoint.region.region_cluster,
                                    region=model_endpoint.region,
                                    router=model_endpoint,
                                    arbiter=None,
                                    feed_async=recent_application.feed_async,
                                    feed_async_granularity=recent_application.feed_async_granularity)
            new_application.add_instance(old_instance) # add old instance
            old_instance.application = new_application
            self.spin_up_new_instance(model_endpoint, new_application, processors)
            model_endpoint.add_application(new_application)
        elif len(model_endpoint.applications) > 1 \
                and len(model_endpoint.applications[-1].instances) == 2 \
                and len(model_endpoint.applications[-2].instances) == 2:
            recent_application = model_endpoint.applications[-2]
            assert len(recent_application.instances) == 2
            self.spin_up_new_instance(model_endpoint, recent_application, processors)
        elif (len(model_endpoint.applications) > 1 \
                and len(model_endpoint.applications[-1].instances) == 2 \
                and len(model_endpoint.applications[-2].instances) == 3) \
                or (len(model_endpoint.applications) == 1 \
                    and len(model_endpoint.applications[-1].instances) == 2):
            recent_application = model_endpoint.applications[-1]
            assert len(recent_application.instances) == 2
            self.spin_up_new_instance(model_endpoint, recent_application, processors)

    def scale_down_to_spot(self, model_endpoint):
            if len(model_endpoint.applications[-1].instances) == 3:
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                recent_application = model_endpoint.applications[-1]
                old_instance = recent_application.instances[-1]
                recent_application.allocator.start_spin_down_instance(old_instance)
            elif len(model_endpoint.applications) > 1 \
                    and len(model_endpoint.applications[-1].instances) == 2 \
                    and len(model_endpoint.applications[-2].instances) == 2:
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                recent_application = model_endpoint.applications[-1]
                old_instance = recent_application.instances[-1]
                recent_application.allocator.start_spin_down_instance(old_instance)
                remaining_instance = recent_application.instances[-1]
                penultimate_application = model_endpoint.applications[-2]
                penultimate_application.add_instance(remaining_instance)
                remaining_instance.application = penultimate_application
                recent_application.remove_instance(remaining_instance)
                model_endpoint.remove_application(recent_application)
                del recent_application
            elif len(model_endpoint.applications) > 1 \
                    and len(model_endpoint.applications[-1].instances) == 2 \
                    and len(model_endpoint.applications[-2].instances) == 3:
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                recent_application = model_endpoint.applications[-2]
                assert len(recent_application.instances) == 3
                last_instance = recent_application.instances[-1]
                recent_application.allocator.start_spin_down_instance(last_instance)

    def force_scale_up(self, model_endpoint):
        """
        For the scaling up or scaling down for a model_endpoint
        """
        #logging.debug(f"Force scale up called at {model_endpoint} for {model_endpoint.model_name} at time {clock()}")
        if self.cluster.has_spot_instance(model_endpoint.model_name) and model_endpoint.scaling_level >= 1:
            #logging.debug(f"Scaling up {self.cluster.region.region_name} {model_endpoint.model_name}")
            self.scale_up_from_spot(model_endpoint)
            #logging.debug([len(app.instances) for app in model_endpoint.applications])
        elif self.cluster.has_spot_instance_any() and model_endpoint.scaling_level >= 2:
            #logging.debug(f"Scaling up other {self.cluster.region.region_name} {model_endpoint.model_name}")
            processors = self.cluster.free_processors_and_kill_spot()
            self.scale_up_from_spot_other(model_endpoint, processors)

    def force_scale_down(self, model_endpoint):
        """
        For the scaling up or scaling down for a model_endpoint
        """
        #logging.debug(f"Force scale down called at {model_endpoint} for {model_endpoint.model_name} at time {clock()}")
        if model_endpoint.scaling_level >= 1:
            self.scale_down_to_spot(model_endpoint)
        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
            self.results['memory_util'].append(model_endpoint.get_memory() / model_endpoint.get_max_memory())


    def scale(self, model_endpoint):
        """
        Scales the application based on memory utilisation.
        """
        if model_endpoint.model_name not in self.scaling_in_progress:
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0

        # reset scaling_in_progress if time has crossed
        if clock() >= self.next_scale_time[model_endpoint.model_name]:
            self.scaling_in_progress[model_endpoint.model_name] = False

        # return if scaling in progress
        if self.scaling_in_progress[model_endpoint.model_name]:
            return

        memory_utilisation = model_endpoint.get_memory() / model_endpoint.get_max_memory()
        mem_util = str([[ins.memory / ins.max_memory for ins in app.instances] for app in model_endpoint.applications])
        if memory_utilisation > 0.5:
            if self.cluster.has_spot_instance(model_endpoint.model_name) and model_endpoint.scaling_level >= 1:
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                #logging.debug(f"Scaling up {self.cluster.region.region_name} {model_endpoint.model_name} {mem_util} at {clock()}")
                # make space
                self.scale_up_from_spot(model_endpoint)
                #logging.debug([len(app.instances) for app in model_endpoint.applications])

                # set scaling progress and next scaling time
                self.scaling_in_progress[model_endpoint.model_name] = True
                self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.reclaim_spot
            elif self.cluster.has_spot_instance_any() and model_endpoint.scaling_level >= 2:
                # TODO: find number of gpus needed
                # gpus = 8
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                #logging.debug(f"Scaling up other {self.cluster.region.region_name} {model_endpoint.model_name} {mem_util} at {clock()}")
                processors = self.cluster.free_processors_and_kill_spot()
                # maintain count of free gpus per cluster
                # if insufficient, randomly free up those many gpus by freeing spot instances

                self.scale_up_from_spot_other(model_endpoint, processors)
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                # set scaling progress and next scaling time
                self.scaling_in_progress[model_endpoint.model_name] = True
                self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_up
        elif memory_utilisation < 0.25 and model_endpoint.scaling_level >= 1:
            # model_endpoint.scale_down()

            #logging.debug([len(app.instances) for app in model_endpoint.applications])
            #logging.debug(f"Scaling down {self.cluster.region.region_name} {model_endpoint.model_name} {mem_util} at {clock()}")
            self.scale_down_to_spot(model_endpoint)
            #logging.debug([len(app.instances) for app in model_endpoint.applications])
            self.scaling_in_progress[model_endpoint.model_name] = True
            self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_down
            # #logging.debug(f"Tried to scale down {self.cluster.region.region_name} {model_endpoint.model_name} {mem_util} at {clock()}")
        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
            self.results['memory_util'].append(memory_utilisation)
        
class GlobalArbiterAwareShortTermArbiter(BasicArbiter):
    """Arbiter that is aware of the recommendations by the global arbiter"""
    """
    Run with 
        long_term_scaling=True
        short_term_scaling=True
        controller.regions.0.arbiter=global_arbiter_short_term_scaling
        controller.regions.1.arbiter=global_arbiter_short_term_scaling
        controller.regions.2.arbiter=global_arbiter_short_term_scaling
    """
    
    def __init__(self,
                 cluster,
                 overheads):
        super().__init__(cluster, overheads)
    
    def scale(self, model_endpoint):
        """
        Implement scaling logic here
        """
        # add to dictionary if not there
        if model_endpoint.model_name not in self.changes:
            self.changes[model_endpoint.model_name] = 0
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0
        
        # reset scaling_in_progress if time has crossed
        if clock() >= self.next_scale_time[model_endpoint.model_name]:
            self.scaling_in_progress[model_endpoint.model_name] = False
        
        # return if scaling in progress
        if self.scaling_in_progress[model_endpoint.model_name]:
            return

        # return if no scaling to do
        if self.changes[model_endpoint.model_name] == 0:
            return
        # do scaling
        elif self.changes[model_endpoint.model_name] > 0:
            #logging.debug("Scaling up")
            if self.cluster.has_spot_instance(model_endpoint.model_name) and model_endpoint.scaling_level >= 1:
                #logging.debug(f"Scaling up {self.cluster.region.region_name} {model_endpoint.model_name}")
                self.scale_up_from_spot(model_endpoint)
                #logging.debug([len(app.instances) for app in model_endpoint.applications])
                
                # set scaling progress and next scaling time
                self.scaling_in_progress[model_endpoint.model_name] = True
                self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.reclaim_spot
            elif self.cluster.has_spot_instance_any() and model_endpoint.scaling_level >= 2:
                #logging.debug(f"Scaling up other {self.cluster.region.region_name} {model_endpoint.model_name}")
                processors = self.cluster.free_processors_and_kill_spot()
                self.scale_up_from_spot_other(model_endpoint, processors)
                
                # set scaling progress and next scaling time
                self.scaling_in_progress[model_endpoint.model_name] = True
                self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_up
            else:
                #logging.debug("Failed to up scale")
                # indicate no scaling
                self.scaling_in_progress[model_endpoint.model_name] = False
            
            # if scaled, subtract from changes to make
            if self.scaling_in_progress[model_endpoint.model_name]:
                self.changes[model_endpoint.model_name] -= 1

        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
            self.results['memory_util'].append(model_endpoint.get_memory() / model_endpoint.get_max_memory())
        

    def force_scale_up(self, model_endpoint):
        if model_endpoint.model_name not in self.changes.keys():
            self.changes[model_endpoint.model_name] = 0
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0
        self.changes[model_endpoint.model_name] += 1
        #logging.debug(f"force_scale_up for {model_endpoint.model_name} at {model_endpoint}: {self.changes[model_endpoint.model_name]}")
    
    def force_scale_down(self, model_endpoint):
        #logging.debug(f"Force scale down called at {model_endpoint} for {model_endpoint.model_name} at time {clock()}")
        if model_endpoint.scaling_level >= 1:
            self.scale_down_to_spot(model_endpoint)
        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name))) 
            self.results['memory_util'].append(model_endpoint.get_memory() / model_endpoint.get_max_memory())   


class GlobalAribiterMemoryUtilizationScaling(BasicArbiter):
    """Arbiter that is aware of the recommendations by the global arbiter"""
    """
    Run with 
        long_term_scaling=True
        short_term_scaling=True
        controller.regions.0.arbiter=global_arbiter_short_term_scaling
        controller.regions.1.arbiter=global_arbiter_short_term_scaling
        controller.regions.2.arbiter=global_arbiter_short_term_scaling
    """
    
    def __init__(self,
                 cluster,
                 overheads):
        super().__init__(cluster, overheads)            
    
    def scale_down_to_spot(self, model_endpoint):
        if len(model_endpoint.applications[-1].instances) == 3:
            #logging.debug([len(app.instances) for app in model_endpoint.applications])
            recent_application = model_endpoint.applications[-1]
            old_instance = recent_application.instances[-1]
            recent_application.allocator.start_spin_down_instance(old_instance)
            return True
        elif len(model_endpoint.applications) > 1 \
                and len(model_endpoint.applications[-1].instances) == 2 \
                and len(model_endpoint.applications[-2].instances) == 2:
            #logging.debug([len(app.instances) for app in model_endpoint.applications])
            recent_application = model_endpoint.applications[-1]
            old_instance = recent_application.instances[-1]
            recent_application.allocator.start_spin_down_instance(old_instance)
            remaining_instance = recent_application.instances[-1]
            penultimate_application = model_endpoint.applications[-2]
            penultimate_application.add_instance(remaining_instance)
            remaining_instance.application = penultimate_application
            recent_application.remove_instance(remaining_instance)
            model_endpoint.remove_application(recent_application)
            del recent_application
            return True
        elif len(model_endpoint.applications) > 1 \
                and len(model_endpoint.applications[-1].instances) == 2 \
                and len(model_endpoint.applications[-2].instances) == 3:
            #logging.debug([len(app.instances) for app in model_endpoint.applications])
            recent_application = model_endpoint.applications[-2]
            assert len(recent_application.instances) == 3
            last_instance = recent_application.instances[-1]
            recent_application.allocator.start_spin_down_instance(last_instance)
            return True
        return False


    def scale_up_logic(self, model_endpoint):
        if self.cluster.has_spot_instance(model_endpoint.model_name) and model_endpoint.scaling_level >= 1:
            #logging.debug(f"Scaling up {self.cluster.region.region_name} {model_endpoint.model_name}")
            self.scale_up_from_spot(model_endpoint)
            #logging.debug([len(app.instances) for app in model_endpoint.applications])
            
            # set scaling progress and next scaling time
            self.scaling_in_progress[model_endpoint.model_name] = True
            self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.reclaim_spot
        elif self.cluster.has_spot_instance_any() and model_endpoint.scaling_level >= 2:
            #logging.debug(f"Scaling up other {self.cluster.region.region_name} {model_endpoint.model_name}")
            processors = self.cluster.free_processors_and_kill_spot()
            self.scale_up_from_spot_other(model_endpoint, processors)
            
            # set scaling progress and next scaling time
            self.scaling_in_progress[model_endpoint.model_name] = True
            self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_up
        else:
            #logging.debug("Failed to up scale")
            # indicate no scaling
            self.scaling_in_progress[model_endpoint.model_name] = False


    def scale(self, model_endpoint):
        """
        Implement scaling logic here
        """
        # add to dictionary if not there
        if model_endpoint.model_name not in self.changes:
            self.changes[model_endpoint.model_name] = 0
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0
        
        # reset scaling_in_progress if time has crossed
        if clock() >= self.next_scale_time[model_endpoint.model_name]:
            self.scaling_in_progress[model_endpoint.model_name] = False
        
        # return if scaling in progress
        if self.scaling_in_progress[model_endpoint.model_name]:
            for model_ep in self.cluster.region.model_endpoint_routers:
                self.results['timestamp'].append(clock())
                self.results['model'].append(model_ep.model_name)
                self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
                self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
                self.results['memory_util'].append(model_endpoint.get_memory() / model_endpoint.get_max_memory())
            return

        memory_utilisation = model_endpoint.get_memory() / model_endpoint.get_max_memory()
        # return if no scaling to do
        if self.changes[model_endpoint.model_name] == 0:
            if memory_utilisation > 0.9:
                # scale up due to high memory utilization even though global arbiter put a limit
                self.scale_up_logic(model_endpoint)
                if self.scaling_in_progress[model_endpoint.model_name]:
                    self.changes[model_endpoint.model_name] -= 1
        # do scaling
        elif self.changes[model_endpoint.model_name] > 0 and memory_utilisation > 0.5:
            #logging.debug("Scaling up")
            self.scale_up_logic(model_endpoint)
            # if scaled, subtract from changes to make
            if self.scaling_in_progress[model_endpoint.model_name]:
                self.changes[model_endpoint.model_name] -= 1
        elif self.changes[model_endpoint.model_name] < 0 and memory_utilisation < 0.25:
            if self.scale_down_to_spot(model_endpoint):
                #logging.debug(f"Scaling down {model_endpoint}")
                self.changes[model_endpoint.model_name] += 1

        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
            self.results['memory_util'].append(memory_utilisation)
        

    def force_scale_up(self, model_endpoint):
        if model_endpoint.model_name not in self.changes.keys():
            self.changes[model_endpoint.model_name] = 0
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0
        self.changes[model_endpoint.model_name] += 1
        #logging.debug(f"force_scale_up for {model_endpoint.model_name} at {model_endpoint}: {self.changes[model_endpoint.model_name]}")
    
    def force_scale_down(self, model_endpoint):
        #logging.debug(f"Force scale down called at {model_endpoint} for {model_endpoint.model_name} at time {clock()}")
        if model_endpoint.scaling_level >= 1:
            self.scale_down_to_spot(model_endpoint)
        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))   
            self.results['memory_util'].append(model_endpoint.get_memory() / model_endpoint.get_max_memory()) 


class GlobalArbiterARIMAChecking(GlobalAribiterMemoryUtilizationScaling):
    def __init__(self,
                 cluster,
                 overheads,
                 scaling_threshhold):
        super().__init__(cluster, overheads)
        self.arima_forecast = None
        self.threshhold = scaling_threshhold
        print(self.threshhold)
        self.request_arrival_times = Queue()
        self.request_tokens = Queue()
        self.tokens_last_minute = 0

    def set_arima_forecast(self, forecast):
        self.arima_forecast = forecast
        print(f"Arime forecast set for length: {len(self.arima_forecast)}")

    def manage_tps(self, request):
        while self.request_arrival_times.qsize() > 0 and self.request_arrival_times.queue[0] < clock() - 60:
            self.request_arrival_times.get()
            tokens = self.request_tokens.get()
            self.tokens_last_minute -= tokens

        self.request_arrival_times.put(clock())
        self.request_tokens.put(request.prompt_size)
        self.tokens_last_minute += request.prompt_size

    def ratio_between_arima_and_actual(self):
        predicted_tokens = self.arima_forecast.loc[self.arima_forecast["Time"] == 60 * (clock() // 60 + 1)]["Predicted"].sum()
        return self.tokens_last_minute / predicted_tokens

    def scale(self, model_endpoint):
        # add to dictionary if not there
        self.manage_tps(model_endpoint.current_request)
        if model_endpoint.model_name not in self.changes:
            self.changes[model_endpoint.model_name] = 0
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0
        
        # reset scaling_in_progress if time has crossed
        if clock() >= self.next_scale_time[model_endpoint.model_name]:
            self.scaling_in_progress[model_endpoint.model_name] = False

        # return if scaling in progress
        if self.scaling_in_progress[model_endpoint.model_name]:
            for model_ep in self.cluster.region.model_endpoint_routers:
                self.results['timestamp'].append(clock())
                self.results['model'].append(model_ep.model_name)
                self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
                self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
                self.results['memory_util'].append(model_endpoint.get_memory() / model_endpoint.get_max_memory())
            return

        memory_utilisation = model_endpoint.get_memory() / model_endpoint.get_max_memory()
        # return if no scaling to do
        if self.changes[model_endpoint.model_name] == 0:
            ratio = self.ratio_between_arima_and_actual()
            if ratio > self.threshhold:
                #logging.debug(f"Ratio for upscaling: {ratio}")
                # scale up due to high memory utilization even though global arbiter put a limit
                self.scale_up_logic(model_endpoint)
                if self.scaling_in_progress[model_endpoint.model_name]:
                    self.changes[model_endpoint.model_name] -= 1
        # do scaling
        elif self.changes[model_endpoint.model_name] > 0 and memory_utilisation > 0.5: # TODO change to above
            #logging.debug(f"Memory utilization for upscaling: {memory_utilisation}")
            #logging.debug("Scaling up")
            self.scale_up_logic(model_endpoint)
            # if scaled, subtract from changes to make
            if self.scaling_in_progress[model_endpoint.model_name]:
                self.changes[model_endpoint.model_name] -= 1
        elif self.changes[model_endpoint.model_name] < 0 and memory_utilisation < 0.25: # TODO change to above
            #logging.debug(f"Memory utilization for downscaling: {memory_utilisation}")
            if self.scale_down_to_spot(model_endpoint):
                #logging.debug(f"Scaling down {model_endpoint}")
                self.changes[model_endpoint.model_name] += 1

        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
            self.results['memory_util'].append(memory_utilisation)

class ChironArbiter(BasicArbiter):
    """
    Chiron Arbiter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_write = 0
    def scale(self, model_endpoint):
        """
        Scales the application based on memory utilisation.
        """
        if model_endpoint.model_name not in self.scaling_in_progress:
            self.scaling_in_progress[model_endpoint.model_name] = False
            self.next_scale_time[model_endpoint.model_name] = 0

        # reset scaling_in_progress if time has crossed
        if clock() >= self.next_scale_time[model_endpoint.model_name]:
            self.scaling_in_progress[model_endpoint.model_name] = False

        # return if scaling in progress
        if self.scaling_in_progress[model_endpoint.model_name]:
            return
        memory_utilisation = model_endpoint.get_memory() / model_endpoint.get_max_memory()
        # if clock() - self.last_write > 5:
        # logging.info(f"{model_endpoint.model_name} {[len(app.instances) for app in model_endpoint.applications]} {round(memory_utilisation, 2)}")
        if model_endpoint.model_name.endswith("-d"):
            ttft_batch = 24*60*60
            instance_cnt = sum([len(app.instances) for app in model_endpoint.applications])
            pending_time = model_endpoint.pending_requests * model_endpoint.prompt_time + \
            model_endpoint.pending_tokens / (model_endpoint.decode_throughput * instance_cnt)
            if pending_time > ttft_batch:
                # scale up
                if self.cluster.has_spot_instance(model_endpoint.model_name) and model_endpoint.scaling_level >= 1:
                    self.scale_up_from_spot(model_endpoint)
                    self.scaling_in_progress[model_endpoint.model_name] = True
                    self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.reclaim_spot
                elif self.cluster.has_spot_instance_any() and model_endpoint.scaling_level >= 2:
                    processors = self.cluster.free_processors_and_kill_spot()
                    self.scale_up_from_spot_other(model_endpoint, processors)
                    self.scaling_in_progress[model_endpoint.model_name] = True
                    self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_up
            # if no batch requests, scale down to 0
            if model_endpoint.applications[0].router.pending_requests == 0 and model_endpoint.applications[0].instances > 0:
                # while model_endpoint.applications[0].instances > 0:
                self.scale_down_to_spot(model_endpoint)
                self.scaling_in_progress[model_endpoint.model_name] = True
                self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_down
        else:
            model_name = model_endpoint.model_name.split("-")[0]
            iw_model_endpoint = self.cluster.region.get_model_endpoint(model_name+'-p')
            mix_model_endpoint = self.cluster.region.get_model_endpoint(model_name)
            cnt_iw_total = sum([len(app.instances) for app in iw_model_endpoint.applications])
            cnt_mix_total = sum([len(app.instances) for app in mix_model_endpoint.applications])
            cnt_iw_running = 0
            for app in iw_model_endpoint.applications:
                for instance in app.instances:
                    if instance.cnt_iw_requests > 0:
                        cnt_iw_running += 1
            cnt_mix_running = 0
            for app in mix_model_endpoint.applications:
                for instance in app.instances:
                    if instance.cnt_iw_requests > 0:
                        cnt_mix_running += 1
            ibp = (cnt_iw_running + cnt_mix_running) / (cnt_iw_total + cnt_mix_total)
            # if clock() - self.last_write > 5:
            #     self.last_write = clock()
            #     logging.info(f"Time: {clock()//1}, ibp={round(ibp, 2)}, {cnt_iw_running}, {cnt_mix_running}, {cnt_iw_total}, {cnt_mix_total}")
            # print([a.model_name for a in self.cluster.region.model_endpoint_routers])
            if ibp > 0.6:
                if self.cluster.has_spot_instance(model_endpoint.model_name) and model_endpoint.scaling_level >= 1:
                    self.scale_up_from_spot(model_endpoint)
                    self.scaling_in_progress[model_endpoint.model_name] = True
                    self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.reclaim_spot
                elif self.cluster.has_spot_instance_any() and model_endpoint.scaling_level >= 2:
                    processors = self.cluster.free_processors_and_kill_spot()
                    self.scale_up_from_spot_other(model_endpoint, processors)
                    self.scaling_in_progress[model_endpoint.model_name] = True
                    self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_up
            else:
                self.scale_down_to_spot(model_endpoint)
                self.scaling_in_progress[model_endpoint.model_name] = True
                self.next_scale_time[model_endpoint.model_name] = clock() + self.overheads.spin_down
        
        for model_ep in self.cluster.region.model_endpoint_routers:
            self.results['timestamp'].append(clock())
            self.results['model'].append(model_ep.model_name)
            self.results['prod'].append(str(sum([len(app.instances) for app in model_ep.applications])))
            self.results['spot'].append(str(self.cluster.get_spot_instance_count(model_ep.model_name)))
            self.results['memory_util'].append(memory_utilisation)
