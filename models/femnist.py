import numpy as np
import tensorflow as tf

from model import Model

NUM_CLASSES = 62
FILTER_FIRST = 32
FILTER_SECOND = 64
NUM_FC_NEURONS = 400


class ClientModel(Model):
    def __init__(self, seed, lr, train_bs=32, test_bs=16, input_shape=None, optimizer=None, sem=False):
        super(ClientModel, self).__init__(seed, lr, train_bs, test_bs, input_shape, optimizer, sem)
        
    def create_model(self):
        return self.create_CNNmodel()

    def create_CNN_ma_model(self):
        inner_model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Reshape((28, 28, 1)), 
            tf.keras.layers.Conv2D(FILTER_FIRST, 5, input_shape=(28, 28, 1) ,activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(FILTER_SECOND, 5, activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
        ]) 

        dense_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(NUM_FC_NEURONS,activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  
        ])

        out_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(784)),               
                inner_model,
                tf.keras.layers.Flatten(),
                dense_model
        ])

        self.set_model_trainable_compile(out_model)

        return inner_model, dense_model, out_model        

        
    def create_CNNmodel(self):
        """Model function for CNN."""
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Reshape((28, 28, 1)), 
            tf.keras.layers.Conv2D(FILTER_FIRST, 5, input_shape=(28, 28, 1) ,activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(FILTER_SECOND, 5, activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_FC_NEURONS,activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  
        ])

        
        self.set_model_trainable_compile(model)
        return model
    
    def create_normal_densemodel(self):        
        return self.create_cuFCmodel([256, 128])
    
    def create_cuFCmodel(self, L_n):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(L_n[0],activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(L_n[1], activation='relu'),            
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        self.set_model_trainable_compile(model)
        return model        
    
    def process_x(self, raw_x):
        return np.array(raw_x).astype(np.float32)
        
    def process_y(self, raw_y):
        return np.array(raw_y)
    
    def set_model_trainable_compile(self, model, trainable_bool=True):
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        
        return