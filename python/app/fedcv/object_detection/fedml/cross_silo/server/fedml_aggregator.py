import logging
import random
import time

import copy
import numpy as np
import torch
from fedml import mlops

from ...ml.engine import ml_engine_adapter
from model.yolov5.utils.general import (LOGGER, colorstr)
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent

class FedMLAggregator(object):
    def __init__(
            self,
            train_global,
            test_global,
            all_train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            client_num,
            device,
            args,
            server_aggregator,
    ):
        self.aggregator = server_aggregator
        
        self.mAPs=dict()
        self.models = dict()
        
        self.old_models=[]
        self.old_model_scores=[]

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.client_num = client_num
        self.device = device
        self.args.device = device
        logging.info("self.device = {}".format(self.device))
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)

        # for dictionary model_params, we let the user level code to control the device
        if type(model_params) is not dict:
            model_params = ml_engine_adapter.model_params_to_device(self.args, model_params, self.device)

        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True
        
    
    
        
    def add_local_trained_mAPs(self, index, mAPs):
        logging.info("Add mAPs. index = %d" % index)
        self.mAPs[index] = mAPs



    def check_whether_all_receive(self):
        print("Check")
        logging.debug("client_num = {}".format(self.client_num))
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate_and_compare(self):
        start_time = time.time()

        model_list = []
        
        # mAPs = dict()
        # for _id in self.mAPs:
        #     client_idx = list(self.mAPs[_id].keys())[0]
        #     mAPs[client_idx] = self.mAPs[_id][client_idx]
        
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            
        model_list = self.aggregator.on_before_aggregation(model_list)
        averaged_params = self.aggregator.aggregate(model_list)

        if type(averaged_params) is dict:
            for client_index in range(len(averaged_params)):
                averaged_params[client_index] = self.aggregator.on_after_aggregation(averaged_params[client_index])
        else:
            averaged_params = self.aggregator.on_after_aggregation(averaged_params)

        # self.set_global_model_params(averaged_params)
        
        if self.args.use_best_from_server_clients:
            ValOn = self.args.data_desc[self.args.val_on]
            _model_param = [(0,averaged_params)]+model_list
            _model_map = []
            for _idx, (_,_param) in enumerate(_model_param):
                _results = []
                self.args._model_idx  = _idx
                self.args._model_desc = "Server" if _idx==0 else f"Client_{_idx}"
                if (_idx-1) in self.mAPs.keys():
                    _kkey = list(self.mAPs[_idx-1].keys())[0]
                    if ValOn in self.mAPs[_idx-1][_kkey]:
                        LOGGER.info(colorstr( "bright_green"  , "bold" , f"\n\n\t{self.args._model_desc} results for {ValOn} dataset already exist." ))
                        _results = self.mAPs[_idx-1][_kkey][ValOn]
                if _results == []:
                    self.set_global_model_params(_param)
                    results = self.test_on_server_for_all_clients()
                    _results = results[self.args.val_on] # Perform server evaluation on  0-Old, 1-New, 2-Merged
                _model_map.append(_results[2]) # 2-mAP50 3-mAP
                if _idx==0: aggr_scores=_results[2]
            
            if self.args.keep_server_model_history:
                _all_models         = _model_param  + self.old_models       
                _all_model_scores   = _model_map    + self.old_model_scores 
            else:
                _all_models         = _model_param
                _all_model_scores   = _model_map  
        
        
            # max_map = max(_model_map)
            # best_model = _model_map.index(max_map)
            max_map = max(_all_model_scores)
            best_model = _all_model_scores.index(max_map)
            
            MLOpsProfilerEvent.log_to_wandb({   f"Best_Model": best_model,
                                                f"Aggregated_Model_Score": aggr_scores,
                                                f"Best_Model_Score": max_map,
                                                f"Round_No": self.args.round_idx,
                                                f"Round x Epoch": self.args.round_idx*self.args.epochs,})
            
            if best_model>(len(_model_map)-1):
                # best_model-=len(_model_map)
                WhoGaveWeights=f"Round {best_model}"
            else:
                WhoGaveWeights = 'Server' if best_model==0 else f'Client_{best_model}'
                
            msg = colorstr( "bright_yellow"  , "bold" , f"\n\t The best weights are given by {WhoGaveWeights} with mAP: {max_map}\n" )
            LOGGER.info(msg)
            
            averaged_params = copy.deepcopy(_all_models[best_model][1])
            self.set_global_model_params(averaged_params)
            
            self.old_models.append((0,averaged_params))
            self.old_model_scores.append(max_map)
            
        else:
            self.set_global_model_params(averaged_params)
            self.args._model_idx  = 0
            self.args._model_desc = "Server" 
            ValOn = self.args.data_desc[self.args.val_on]
            results = self.test_on_server_for_all_clients()
            _results = results[self.args.val_on] # Perform server evaluation on  0-Old, 1-New, 2-Merged
            aggr_scores=_results[2]
            
        # if self.args.keep_server_model_history: self.old_models       = [(0,averaged_params)]
        # if self.args.keep_server_model_history: self.old_model_scores = max_map
        
        end_time = time.time()
        logging.info("aggregate and comparison time cost: %d" % (end_time - start_time))
        self.aggregator.test_all(self.args) #FIXME: Save Weights
        return averaged_params

    def aggregate(self):
        start_time = time.time()

        model_list = []
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        model_list = self.aggregator.on_before_aggregation(model_list)
        averaged_params = self.aggregator.aggregate(model_list)

        if type(averaged_params) is dict:
            for client_index in range(len(averaged_params)):
                averaged_params[client_index] = self.aggregator.on_after_aggregation(averaged_params[client_index])
        else:
            averaged_params = self.aggregator.on_after_aggregation(averaged_params)

        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
        """

        Args:
            round_idx: round index, starting from 0
            client_num_in_total: this is equal to the users in a synthetic data,
                                    e.g., in synthetic_1_1, this value is 30
            client_num_per_round: the number of edge devices that can train

        Returns:
            data_silo_index_list: e.g., when client_num_in_total = 30, client_num_in_total = 3,
                                        this value is the form of [0, 11, 20]

        """
        logging.info(
            "client_num_in_total = %d, client_num_per_round = %d" % (client_num_in_total, client_num_per_round)
        )
        assert client_num_in_total >= client_num_per_round

        if client_num_in_total == client_num_per_round:
            return [i for i in range(client_num_per_round)]
        else:
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(range(client_num_in_total), client_num_per_round, replace=False)
            return data_silo_index_list

    def client_selection(self, round_idx, client_id_list_in_total, client_num_per_round):
        """
        Args:
            round_idx: round index, starting from 0
            client_id_list_in_total: this is the real edge IDs.
                                    In MLOps, its element is real edge ID, e.g., [64, 65, 66, 67];
                                    in simulated mode, its element is client index starting from 1, e.g., [1, 2, 3, 4]
            client_num_per_round:

        Returns:
            client_id_list_in_this_round: sampled real edge ID list, e.g., [64, 66]
        """
        if client_num_per_round == len(client_id_list_in_total):
            return client_id_list_in_total
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        return client_id_list_in_this_round

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self):
        # logging.info("\t################test_on_server: {}".format(self.args.round_idx))
        results = self.aggregator.test_all_val_datasets(self.test_global, self.device, self.args)
        # self.aggregator.test_all(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args,) #FIXME: Save Weights
        return results
