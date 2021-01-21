
import os
import sys
import numpy as np
import time

from utils_io import get_job_config
from utils.model_utils import read_data
from utils.args import parse_job_args


from fedsem import Fedsem_Trainer
from fedavg import Fedavg_Trainer
from fedprox import Fedprox_Trainer
from fedbayes import Fedbayes_Sing_Trainer
from modelsaver import Model_Saver

def read_yamlconfig(args):
    yaml_file = os.path.join("..", "configs", args.experiment, args.configuration)
    job = get_job_config(yaml_file)
    params = job['PARAMS']

    rounds = params['num-rounds']
    print("config rounds: ", rounds)

    lr = params['lr']
    print("config lr: ", lr )    
    
    epochs = params['epochs']
    print("config epochs: ", epochs)
    
    clients_per_round = params['clients-per-round']
    print("config clients per round: ", clients_per_round)
    
    return params

def main():
    args = parse_job_args()
    config = read_yamlconfig(args)
    
    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')   
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    exp_seeds, book_keep = config["exp-seeds"], [0.] * len(config["exp-seeds"])
    
    for j, rnd_sed in enumerate(exp_seeds):
        config["seed"] = rnd_sed
        if args.experiment == 'fedavg':
            trainer = Fedavg_Trainer(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'fedprox':
            trainer = Fedprox_Trainer(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'fedcluster':
            pass
        elif args.experiment == 'feddane':
            pass
        elif args.experiment == 'fedbayes':
            trainer = Fedbayes_Sing_Trainer(users, groups, train_data, test_data)
            metric =trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'modelsaver':
            trainer = Model_Saver(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()        
        elif args.experiment == 'fedsem':
            trainer = Fedsem_Trainer(users, groups, train_data, test_data) 
            metric = trainer.begins(config, args)
            trainer.ends() 
        else:
            print("Applications not defined. Please check configs directory if the name is right.")
            break
        book_keep[j] = metric
        
    finals = np.array(book_keep) * 100
    print(finals)
    print("{} runs - std: {}, med: {}".format(len(exp_seeds), 
                                              np.std(finals),
                                             np.median(finals)))        
main()
