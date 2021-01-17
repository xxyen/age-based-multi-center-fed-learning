import copy
import importlib
import os
import pickle
import numpy as np
import re
import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

from client import Client
from server import Server
from model import ServerModel
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from baseline_constants import KERNAL_WIDTH, KERNEL_HEIGHT, NUM_INPUT_CHANNEL, NUM_OUTPUT_CHANNEL

#from utils.matching.pfnm import layer_group_descent as pdm_multilayer_group_descent
from utils.matching.cnn_pfnm import layerwise_sampler
from utils.matching.cnn_retrain import reconstruct_weights, local_train, combine_network_after_matching

def saved_cls_counts(clients, file):
    net_cls_counts = {}

    for c in clients:
        unq, unq_cnt = np.unique(c.train_data['y'], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[c.id] = tmp
        
    with open(file, 'wb+') as f:
        pickle.dump(net_cls_counts, f)

def pdm_prepare_freq(cls_freqs, n_classes=10):
    freqs = []
    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs

def load_counts(dataset):
    fname = "{}_counts".format(dataset)
    with open(fname, 'rb') as f:
        loaded_cls_counts = pickle.load(f) 
    return loaded_cls_counts

def load_files():
    def get_weightfile():
        sepath = os.path.join(".", "workernn")
        j_files =  [f for f in os.listdir(sepath) if re.match("femnist_.{5}_.{2}\.pb", f)]
        return j_files

    def process_file(f):
        fname = os.path.join("workernn", f)
        with open(fname, 'rb') as file:
            w = pickle.load(file)        
        return w 
    
    return list(map(process_file, get_weightfile()))

def get_cnn_w(value):
    o_shape = value.shape
    width, height = o_shape[KERNAL_WIDTH], o_shape[KERNEL_HEIGHT]
    num_in_chn, num_out_chn = o_shape[NUM_INPUT_CHANNEL], o_shape[NUM_OUTPUT_CHANNEL]
    n_shape = (width * height * num_in_chn, num_out_chn)    
    w = value.reshape(n_shape)
    # IMPORTANAT, here need to invoke transpose , because
    # in orignal paper, they use pytorch, and the order in 
    # pytorch is different from tensorflow
    # the order is (NUM_OUTPUT_CHANNEL, NUM_INPUT_CHANNEL, ker_width, ker_height)
    w = w.transpose() 
    return w


    
def load_local_model_weight_func(model, model_summary):
    all_layers = []
    dense_varname, weight_varname = "dense", "kernel"
    for var_name, value in zip(model_summary, model):
        if var_name.startswith("conv"):
            if var_name.endswith(weight_varname):
                all_layers.append(get_cnn_w(value))
            else:
                all_layers.append(value)
        elif var_name.startswith("batch"):
            pass
        elif var_name.startswith(dense_varname):
            if var_name.endswith(weight_varname):
                all_layers.append(value.transpose())
            else:
                all_layers.append(value)
    return all_layers

class Fedbayes_Sing_Trainer:
    
    def __init__(self, users, groups, train_data, test_data):
        # matching requires num of classes to be set during 
        # model_config stage, or it can cause program failure
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        self.num_classes = 0 
        self.shape_func = None
        self.upd_collector = []
        
    def avg_cls_weights(self, batch, dataset, num_classes):
        all_class_freq = load_counts(dataset)
        J = len(batch)
        wids = [ c.id for c in batch ]
        wcnt = [ i for i in range(len(batch))]
        ret_class_freq = {c.id: all_class_freq[c.id] for c in batch}

        averaging_weights = np.zeros((J, num_classes), dtype=np.float32)
        for i in range(num_classes):
            total_num_counts = 0
            worker_class_counts = [0] * J
            for j, c in zip(wids, wcnt):
                if i in all_class_freq[j].keys():
                    total_num_counts += all_class_freq[j][i]
                    worker_class_counts[c] = all_class_freq[j][i]
                else:
                    total_num_counts += 0
                    worker_class_counts[c] = 0
            if total_num_counts == 0:
                print("#" * 5, ", strange, total 0 happened.")
            averaging_weights[:, i] = np.array(worker_class_counts) / total_num_counts

        return averaging_weights, ret_class_freq
    
    def recover_weights(self, weights, assignment, model_summary, model_meta_data):
        res_weights = []
        conv_varname, dense_varname, weight_varname = "conv", "dense", "kernel"
        #print("checking len, model summ: {}, model meta data: {}".format(len(model_summary), len(model_meta_data)))
        for var_name, o, v in zip(model_summary, model_meta_data, weights):
            print("name {}, old shape is {}, new shape is {}".format(var_name, o, v.shape))
            if var_name.startswith(conv_varname):
                if var_name.endswith(weight_varname):
                    w = v.reshape(o)
                    w = w.transpose((2, 3, 1, 0))
                else:
                    w = v
            elif var_name.startswith("batch"):
                w = np.ones(o)
            elif var_name.startswith(dense_varname):
                if var_name.endswith(weight_varname):
                    w = v.transpose()
                else:
                    w = v
            res_weights.append(w)
        # just change last layer, carefully, not sure how it works
        # do check the reason out after Jan, find a way to 
        # improve it.
        res_weights[-2] = res_weights[-2].T
        return res_weights
    
    def train_model(self, client_model, train_data, weights, assignment, config):
        # what maintain by assignment is a dictionary of 
        #{layer_name: [global_id]}
        # the meaning of it is a worker's all layer(except last) matching assignment
        epochs = config["epochs"]
        batch_size = config["batch-size"]        
        client_model.set_params(weights)
        client_model.train(train_data, num_epochs=epochs, batch_size=batch_size)
        update = client_model.get_params()
        self.upd_collector.append(update)
        
    def model_config(self, config, dataset, my_model):   
        shared_model = my_model
        model_path = '%s/%s.py' % (dataset, shared_model)
        if not os.path.exists(model_path):
            print('Please specify a valid dataset and a valid model.')
        model_path = '%s.%s' % (dataset, shared_model)

        print('############################## %s ##############################' % model_path)
        mod = importlib.import_module(model_path)
        ClientModel = getattr(mod, 'ClientModel')
        self.shape_func = getattr(mod, 'get_convolution_extractor_shape')
        # Suppress tf warnings
        tf.logging.set_verbosity(tf.logging.WARN)

        # Create 2 models
        model_params = MODEL_PARAMS[model_path]
        model_params_list = list(model_params)
        self.num_classes = model_params[1] # setting num_class to be a member of the trainer
        model_params_list.insert(0, config["seed"])
        model_params_list[1] = config["lr"]        
        model_params = tuple(model_params_list)
        tf.reset_default_graph()
        client_model = ClientModel(*model_params)

        # Create server
        server = Server(client_model)

        # Create clients
        _users = self.users
        groups = [[] for _ in _users]
        clients =  [Client(u, g, self.train_data[u], self.test_data[u], client_model) \
                    for u, g in zip(_users, groups)]
        print('%d Clients in Total' % len(clients)) 
        return clients, server, client_model
    
    def begins(self, config, args):
        clients, server, client_model = self.model_config(config, args.dataset, 'cnn')        
               
        num_rounds = config["num-rounds"]
        eval_every = config["eval-every"]
        epochs_per_round = config['epochs']
        batch_size = config['batch-size']
        clients_per_round = config["clients-per-round"]
        state_dict = {}
        
        # Test untrained model on all clients
#         stat_metrics = server.test_model(clients)
#         all_ids, all_groups, all_num_samples = server.get_clients_info(clients)
#         print_metrics(stat_metrics, all_num_samples)
        gl_weight = client_model.get_params()
        model_summary = client_model.get_summary()
        model_meta_data = client_model.get_meta_data()
        first = True
               
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i+1, num_rounds, clients_per_round))
            
            server.select_clients(clients, num_clients=clients_per_round)
            batch_clients = server.selected_clients
            if first:
                cw = gl_weight
            else:
                cw = self.recover_weights(gl_weight, assignment, model_summary, model_meta_data)
            for k in batch_clients:
                if first or not (k.id in state_dict):
                    assignment = []
                else:
                    assignment = state_dict[k.id]
                self.train_model(client_model, k.train_data, cw, assignment, config)
            first = False
                
            gl_weight = self.batch_BBPMAP(batch_clients, state_dict, client_model, config, args)
        
        client_model.close()
        
    def ends(self):
        print("experiment of Fedbayes finished.")
        return

    def batch_BBPMAP(self, batch_clients, state_dict, client_model, config, args):
        model_summary = client_model.get_summary()        
        model_meta_data = client_model.get_meta_data()
        
        n_classes = self.num_classes
        averaging_weights, cls_freqs = self.avg_cls_weights(batch_clients, args.dataset, self.num_classes)
        sigma=config["sigma"]
        sigma0=config["sigma0"]
        gamma=config["gamma"]
        it = config["sample-iter"]
        assignments_list = []
        # param names explained:
        # C is the number of layers for model structure, no counting bias
        # J is the number of clients (workers)
        C = int(len(model_meta_data) / 2)
        J = len(batch_clients)
        matching_shapes = []
        fc_pos = None        

        apply_by_j = lambda j: load_local_model_weight_func(j, model_summary)
        batch_weights = list(map(apply_by_j, self.upd_collector))
        batch_freqs = pdm_prepare_freq(cls_freqs, self.num_classes)
        
        for cur_l in range(1, C):
            layer_hungarian_weights, assignment, L_next = layerwise_sampler(
                 batch_weights=batch_weights, 
                 layer_index=cur_l,
                 sigma0_layers=sigma0, 
                 sigma_layers=sigma, 
                 batch_frequencies=batch_freqs,
                 it=it, 
                 gamma_layers=gamma, 
                 model_meta_data=model_meta_data,
                 model_layer_type= model_summary,
                 n_layers= C,
                 matching_shapes=matching_shapes,
                 )
            assignments_list.append(assignment)
            for client, a_val in zip(batch_clients, assignment):
                p_index = 2 * (cur_l -1)
                v_name = model_summary[p_index]
                if client.id in state_dict:
                    cdict = state_dict[client.id]
                else:
                    cdict = {}
                cdict.update({v_name: a_val})
                state_dict.update({client.id : cdict})
            print("Number of assignment: {}, L_next: {}, matched_weight shape: {} ".format(
                len(assignment), L_next, layer_hungarian_weights[0].shape) )

            matching_shapes.append(L_next)
            temp_network_weg = combine_network_after_matching(batch_weights, cur_l, 
                                                              model_summary, model_meta_data,
                                                              layer_hungarian_weights, L_next, assignment,
                                                             matching_shapes, self.shape_func)


            old_data = client_model.get_params()
            gl_weights = []
            for worker in range(J):
                j = worker
                gl_weights.append(reconstruct_weights(temp_network_weg[j], assignment[j], 
                                                      model_summary, old_data, 
                                                      model_summary[2 * cur_l - 2]))

            models = local_train(batch_clients, gl_weights, cur_l, config)
            batch_weights = list(map(apply_by_j, models))
        
        ## we handle the last layer carefully here ...
        ## averaging the last layer
        matched_weights = []
        last_layer_weights_collector = []

        for worker in range(J):
            # firstly we combine last layer's weight and bias
            bias_shape = batch_weights[worker][-1].shape
            last_layer_bias = batch_weights[worker][-1].reshape((1, bias_shape[0]))
            last_layer_weights = np.concatenate((batch_weights[worker][-2].T, last_layer_bias), axis=0)

            # the directed normalization doesn't work well, let's try weighted averaging
            last_layer_weights_collector.append(last_layer_weights)

        last_layer_weights_collector = np.array(last_layer_weights_collector)

        avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

        for i in range(n_classes):
            avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
            for j in range(J):
                avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
            avg_last_layer_weight[:, i] = avg_weight_collector

        #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
        for i in range(C * 2):
            if i < (C * 2 - 2):
                matched_weights.append(batch_weights[0][i])

        matched_weights.append(avg_last_layer_weight[0:-1, :])
        matched_weights.append(avg_last_layer_weight[-1, :])
        self.upd_collector = []
        
        return matched_weights
