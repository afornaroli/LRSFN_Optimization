#Import tensorflow modules
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#import other modules
from functools import partial
import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Import hessianlearn repository
sys.path.append( os.environ.get('HESSIANLEARN_PATH', "../../"))
from hessianlearn.hessianlearn.data.data import Data
from hessianlearn.hessianlearn.problem.problem import ClassificationProblem,  AutoencoderProblem
from hessianlearn.hessianlearn.problem.regularization import L2Regularization
from hessianlearn.hessianlearn.model.model import HessianlearnModelSettings, HessianlearnModel


def xor_model():
    """Initialize keras model for the XOR problem, to be used with all the different optimizers"""
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation='tanh'))
    model.add(Dense(2, activation=tf.keras.activations.sigmoid))
    return model


def predictions_xor(model, data_test):
    """Function to make predictions from a trained model"""
    y_pred = model.predict(data_test)
    predictions = []
    for i in y_pred:
        if i[0]>= i[1]:
            predictions.append([1,0])
        else:
            predictions.append([0,1])
    return np.array(predictions)
            
def train_baseline(x_train, y_train, model, optimizer):
    """Generic function to train a model with baseline optimizers"""
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    model.fit(x_train, y_train, epochs=25, verbose=2)
    return model



 
def train_LRSFN(x_train, y_train, x_val, y_val, model):
    """Function to train a model with the Low Rank Saddle Free Newton Method"""
    #Initialize problem
    problem = ClassificationProblem(model,dtype = tf.float32)
    #Initialize data objects
    train_data = {problem.x:x_train, problem.y_true:y_train}
    validation_data = {problem.x:x_val, problem.y_true:y_val}
    #Initialize model parameters
    HLModelSettings = HessianlearnModelSettings()
    HLModelSettings['hessian_low_rank'] = 5
    HLModelSettings['max_sweeps'] = 25
    HLModelSettings['alpha'] = 1e-2
    HLModelSettings['printing_sweep_frequency'] = 2
    regularization = L2Regularization(problem)
    settings = {}
    settings['batch_size'] = 128
    settings['hess_batch_size'] = 128
    #Create data object
    data = Data(train_data,settings['batch_size'], validation_data = validation_data, hessian_batch_size=4)
    #Fit model
    HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)
    HLModel.fit()
    return HLModel, model


