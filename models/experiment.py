
import os
import sys
import time

from utils_io import get_job_config
from utils.model_utils import read_data
from utils.args import parse_job_args


from fedavg import Fedavg_Trainer
from fedprox import Fedprox_Trainer
from fedbayes import Fedbayes_Sing_Trainer
from modelsaver import Model_Saver

def read_yamlconfig(args):
    yaml_file = os.path.join("..", "configs", args.experiment, "job.yaml")
    job = get_job_config(yaml_file)
    params = job['PARAMS']

    rounds = params['num-rounds']
    print("config rounds: ", rounds)

    lr = params['lr']
    print("config lr: ", lr )    
    
    epochs = params['epochs']
    print("config epochs: ", epochs)
    
    return params

def main():
    args = parse_job_args()
    config = read_yamlconfig(args)
    
    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')   
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    if args.experiment == 'fedavg':
        trainer = Fedavg_Trainer(users, groups, train_data, test_data)
        trainer.begins(config, args)
        trainer.ends()
    elif args.experiment == 'fedprox':
        trainer = Fedprox_Trainer(users, groups, train_data, test_data)
        trainer.begins(config, args)
        trainer.ends()
    elif args.experiment == 'fedcluster':
        pass
    elif args.experiment == 'fedbayes':
        trainer = Fedbayes_Sing_Trainer(users, groups, train_data, test_data)
        trainer.begins(config, args)
        trainer.ends()
    elif args.experiment == 'modelsaver':
        trainer = Model_Saver(users, groups, train_data, test_data)
        trainer.begins(config, args)
        trainer.ends()        
    else:
        print("Applications not defined. Please check configs directory if the name is right.")
        
main()
