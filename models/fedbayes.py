import importlib
import os
import pickle
import numpy as np
import tensorflow as tf

from client import Client
from server import Server
from model import ServerModel
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from baseline_constants import KERNAL_WIDTH, KERNEL_HEIGHT, NUM_INPUT_CHANNEL, NUM_OUTPUT_CHANNEL

from utils.matching.pfnm import layer_group_descent as pdm_multilayer_group_descent

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

def get_cnn_w(value):
    o_shape = value.shape
    width, height = o_shape[KERNAL_WIDTH], o_shape[KERNEL_HEIGHT]
    num_in_chn, num_out_chn = o_shape[NUM_INPUT_CHANNEL], o_shape[NUM_OUTPUT_CHANNEL]
    n_shape = (width * height * num_in_chn, num_out_chn)    
    w = value.reshape(n_shape)
    w = w.transpose()
    return w

def get_list_model_weights(models, model_summary):
    
    def _weight_func(model):
        all_layers = []
        for var_name, value in zip(model_summary, model):
            if var_name.startswith("conv"):
                if var_name.endswith("kernel"):
                    all_layers.append(get_cnn_w(value))
                else:
                    all_layers.append(value)
            elif var_name.startswith("batch"):
                pass
            elif var_name.startswith("dense"):
                if var_name.endswith("kernel"):
                    all_layers.append(value.transpose())
                else:
                    all_layers.append(value)
        return all_layers
    
    mapped = list(map(_weight_func, models)) 
    return mapped

class Fedbayes_Sing_Trainer:
    
    def __init__(self, users, groups, train_data, test_data):
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        self.num_classes = 0 # matching requires this num to be set during model_config func
        
    def model_config(self, config, dataset, my_model):   
        shared_model = my_model
        model_path = '%s/%s.py' % (dataset, shared_model)
        if not os.path.exists(model_path):
            print('Please specify a valid dataset and a valid model.')
        model_path = '%s.%s' % (dataset, shared_model)

        print('############################## %s ##############################' % model_path)
        mod = importlib.import_module(model_path)
        ClientModel = getattr(mod, 'ClientModel')  
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
        
        
        model_summary = client_model.get_summary()
        for v in model_summary:
            print(v)
                
        nets = [client_model.get_params()]
        for n in nets[0]:
            print(n.shape)
            
        meta_data = client_model.get_meta_data()
              
    
#         num_rounds = config["num-rounds"]
#         eval_every = config["eval-every"]
#         epochs_per_round = config['epochs']
#         batch_size = config['batch-size']
#         clients_per_round = config["clients-per-round"]
        
#         print('--- Round %d of %d: Training %d Clients ---' % (0+1, 1, clients_per_round))

#         joined = np.random.choice(clients, 40, replace=False)
#         for c in joined:
#             c.model.set_params(server.model)
#             comp, num_samples, update = c.train(epochs_per_round, batch_size, None, False)
            
#             saved_file = os.path.join("workernn", "{}_{}.pb".format(args.dataset, c.id))
#             with open(saved_file, 'wb+') as f:
#                 pickle.dump(update, f)
                
#         saved_cls_counts(joined, "{}_counts".format(args.dataset))
        client_model.close()
        
    def ends(self):
        print("experiment of Fedbayes finished.")
        return
            
