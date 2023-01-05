import json
import numpy as np
import os
from collections import defaultdict
import tensorflow as tf
import tensorflow_datasets
from more_itertools import grouper
import copy

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def getchunk(datalist):
    length=len(datalist)
    chunk_size = length//8
    chunks = grouper(datalist, chunk_size)
    new_lists = [list(chunk) for chunk in chunks]
    return new_lists
        
def read_dir_train():
    '''
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    '''
    groups=[]
    mnist = tensorflow_datasets.load('mnist') 
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images=np.int64(train_images>0)
    train_images_new=[]
    for i in range(len(train_images)):
        train_images_new.append(train_images[i].flatten())
    for i in range(len(train_images_new)):
        train_images_new[i]=train_images_new[i].tolist()
    #clients=['f1426_27','f1439_27','f1133_00','f1486_35','f1169_08']
    train_tmp=[]
    for i in range(len(train_labels)):
        tmp=[train_images_new[i],train_labels[i]]
        train_tmp.append(tmp)
    '''
    data={'f1426_27':{'x':[],'y':[]},'f1439_27':{'x':[],'y':[]},'f1133_00':{'x':[],'y':[]},'f1486_35':{'x':[],'y':[]},'f1169_08':{'x':[],'y':[]}}
    for item in train_tmp:
        if item[1]==0 or item[1]==1:
            data['f1426_27']['x'].append(item[0])
            data['f1426_27']['y'].append(item[1])
        elif item[1]==2 or item[1]==3:
            data['f1439_27']['x'].append(item[0])
            data['f1439_27']['y'].append(item[1])
        elif item[1]==4 or item[1]==5:
            data['f1133_00']['x'].append(item[0])
            data['f1133_00']['y'].append(item[1])
        elif item[1]==6 or item[1]==7:
            data['f1486_35']['x'].append(item[0])
            data['f1486_35']['y'].append(item[1])
        elif item[1]==8 or item[1]==9:
            data['f1169_08']['x'].append(item[0])
            data['f1169_08']['y'].append(item[1])
    '''
    clients=[]
    for i in range(40):
        clients.append(i+1)
    data_value=[{'x':[],'y':[]}]*40
    data=dict(zip(clients,data_value))
    datay={'f1426_27':{'x':[],'y':[]},'f1439_27':{'x':[],'y':[]},'f1133_00':{'x':[],'y':[]},'f1486_35':{'x':[],'y':[]},'f1169_08':{'x':[],'y':[]}}
    for item in train_tmp:
        if item[1]==0 or item[1]==1:
            datay['f1426_27']['x'].append(item[0])
            datay['f1426_27']['y'].append(item[1])
        elif item[1]==2 or item[1]==3:
            datay['f1439_27']['x'].append(item[0])
            datay['f1439_27']['y'].append(item[1])
        elif item[1]==4 or item[1]==5:
            datay['f1133_00']['x'].append(item[0])
            datay['f1133_00']['y'].append(item[1])
        elif item[1]==6 or item[1]==7:
            datay['f1486_35']['x'].append(item[0])
            datay['f1486_35']['y'].append(item[1])
        elif item[1]==8 or item[1]==9:
            datay['f1169_08']['x'].append(item[0])
            datay['f1169_08']['y'].append(item[1])
    
    tmp1x=datay['f1426_27']['x']
    tmp1y=datay['f1426_27']['y']
    tmp2x=datay['f1439_27']['x']
    tmp2y=datay['f1439_27']['y']
    tmp3x=datay['f1133_00']['x']
    tmp3y=datay['f1133_00']['y']
    tmp4x=datay['f1486_35']['x']
    tmp4y=datay['f1486_35']['y']
    tmp5x=datay['f1169_08']['x']
    tmp5y=datay['f1169_08']['y']
    
    new1x=getchunk(tmp1x)
    new1y=getchunk(tmp1y)
    new2x=getchunk(tmp2x)
    new2y=getchunk(tmp2y)
    new3x=getchunk(tmp3x)
    new3y=getchunk(tmp3y)
    new4x=getchunk(tmp4x)
    new4y=getchunk(tmp4y)
    new5x=getchunk(tmp5x)
    new5y=getchunk(tmp5y)
    
    for i in range(8):
        dic={}
        dic['x']=new1x[i]
        dic['y']=new1y[i]
        data[i+1]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new2x[i]
        dic['y']=new2y[i]
        data[i+9]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new3x[i]
        dic['y']=new3y[i]
        data[i+17]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new4x[i]
        dic['y']=new4y[i]
        data[i+25]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new5x[i]
        dic['y']=new5y[i]
        data[i+33]=copy.deepcopy(dic)
        
    return clients, groups, data

def read_dir_test():
    groups=[]
    mnist = tensorflow_datasets.load('mnist') 
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images=np.int64(test_images>0)
    test_images_new=[]
    for i in range(len(test_images)):
        test_images_new.append(test_images[i].flatten())
    for i in range(len(test_images_new)):
        test_images_new[i]=test_images_new[i].tolist()
    #clients=['f1426_27','f1439_27','f1133_00','f1486_35','f1169_08']
    test_tmp=[]
    for i in range(len(test_labels)):
        tmp=[test_images_new[i],test_labels[i]]
        test_tmp.append(tmp)
    '''
    data={'f1426_27':{'x':[],'y':[]},'f1439_27':{'x':[],'y':[]},'f1133_00':{'x':[],'y':[]},'f1486_35':{'x':[],'y':[]},'f1169_08':{'x':[],'y':[]}}
    for item in test_tmp:
        if item[1]==0 or item[1]==1:
            data['f1426_27']['x'].append(item[0])
            data['f1426_27']['y'].append(item[1])
        elif item[1]==2 or item[1]==3:
            data['f1439_27']['x'].append(item[0])
            data['f1439_27']['y'].append(item[1])
        elif item[1]==4 or item[1]==5:
            data['f1133_00']['x'].append(item[0])
            data['f1133_00']['y'].append(item[1])
        elif item[1]==6 or item[1]==7:
            data['f1486_35']['x'].append(item[0])
            data['f1486_35']['y'].append(item[1])
        elif item[1]==8 or item[1]==9:
            data['f1169_08']['x'].append(item[0])
            data['f1169_08']['y'].append(item[1])
    '''
    clients=[]
    for i in range(40):
        clients.append(i+1)
    data_value=[{'x':[],'y':[]}]*40
    data=dict(zip(clients,data_value))
    datay={'f1426_27':{'x':[],'y':[]},'f1439_27':{'x':[],'y':[]},'f1133_00':{'x':[],'y':[]},'f1486_35':{'x':[],'y':[]},'f1169_08':{'x':[],'y':[]}}
    for item in test_tmp:
        if item[1]==0 or item[1]==1:
            datay['f1426_27']['x'].append(item[0])
            datay['f1426_27']['y'].append(item[1])
        elif item[1]==2 or item[1]==3:
            datay['f1439_27']['x'].append(item[0])
            datay['f1439_27']['y'].append(item[1])
        elif item[1]==4 or item[1]==5:
            datay['f1133_00']['x'].append(item[0])
            datay['f1133_00']['y'].append(item[1])
        elif item[1]==6 or item[1]==7:
            datay['f1486_35']['x'].append(item[0])
            datay['f1486_35']['y'].append(item[1])
        elif item[1]==8 or item[1]==9:
            datay['f1169_08']['x'].append(item[0])
            datay['f1169_08']['y'].append(item[1])
    
    tmp1x=datay['f1426_27']['x']
    tmp1y=datay['f1426_27']['y']
    tmp2x=datay['f1439_27']['x']
    tmp2y=datay['f1439_27']['y']
    tmp3x=datay['f1133_00']['x']
    tmp3y=datay['f1133_00']['y']
    tmp4x=datay['f1486_35']['x']
    tmp4y=datay['f1486_35']['y']
    tmp5x=datay['f1169_08']['x']
    tmp5y=datay['f1169_08']['y']
    
    new1x=getchunk(tmp1x)
    new1y=getchunk(tmp1y)
    new2x=getchunk(tmp2x)
    new2y=getchunk(tmp2y)
    new3x=getchunk(tmp3x)
    new3y=getchunk(tmp3y)
    new4x=getchunk(tmp4x)
    new4y=getchunk(tmp4y)
    new5x=getchunk(tmp5x)
    new5y=getchunk(tmp5y)
    
    for i in range(8):
        dic={}
        dic['x']=new1x[i]
        dic['y']=new1y[i]
        data[i+1]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new2x[i]
        dic['y']=new2y[i]
        data[i+9]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new3x[i]
        dic['y']=new3y[i]
        data[i+17]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new4x[i]
        dic['y']=new4y[i]
        data[i+25]=copy.deepcopy(dic)
    for i in range(8):
        dic={}
        dic['x']=new5x[i]
        dic['y']=new5y[i]
        data[i+33]=copy.deepcopy(dic)

    return clients, groups, data
    

def read_data():
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir_train()
    test_clients, test_groups, test_data = read_dir_test()

    #assert train_clients == test_clients
    #assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
