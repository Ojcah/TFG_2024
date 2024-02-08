# ************************************************************************************************************************* #
#                                           Copyright (C) 2022 Jorge Brenes Alfaro.
#                                           EL5617 Trabajo Final de Graduación.
#                                           Escuela de Ingeniería Electrónica.
#                                           Tecnológico de Costa Rica.
# *************************************************************************************************************************** #
#
#   This file contains the mimetic neural network (MNN), which is responsible for simulating
#   the dynamic behavior of the output of a propeller damped pendulum system (PAHM).
#
#   For the development of the MNN, recurrent neural networks (RNN) are used, which
#   have the ability to process and obtain information from sequential data, which is
#   suitable for the identification of dynamic systems.
#
#   The data used for training, validation and testing come from the PAHM, which were
#   previously collected. These data correspond to the PWM value of the motor and the
#   pendulum angle captured by the system sensor. Where the angle values are normalized
#   between -1 and 1 for better network performance. Then, the data is prepared in the
#   form of tensor, which is necessary for the GRU layer. Finally, the network model is
#   developed, training, prediction and evaluation of the model are performed.

# Libraries to process data
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Libraries to create de MNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import Adam, RMSprop

import wandb
from wandb.keras import WandbCallback

parser = argparse.ArgumentParser(description = 'Mimetic Neural Network for the physic process.')
parser.add_argument('--project_name', type = str, default = 'RNAM_', help = 'Name of the run.')
parser.add_argument('--units', type = int, default = 32, help = 'Number of the units for the RNAM.')
parser.add_argument('--epochs', type = int, default = 1000, help = 'Number of epochs for the train.')
parser.add_argument('--batch_size', type = int, default = 1, help = 'Number of batch for the train.')
parser.add_argument('--loss_name', type = str, default = 'loss_', help = 'Name for the figure of the loss (.png).')
parser.add_argument('--predict_name', type = str, default = 'Prediction_', help = 'Name for the figure of the prediction (.png).')
parser.add_argument('--model_name', type = str, default = 'Model_Synth_', help = 'Name for the RNAM model (.h5).')
parser.add_argument('--loss_type', type = str, default = 'mse', help = 'Choose a loss, mse or mae')
parser.add_argument('--load_model', type = str, default = '', help = 'Load a previously trained model (.h5).')
args = parser.parse_args()

#   The parameters are archived in Weights and Biases (W&B), as well as the results of the
#   execution for further evaluation.
wandb.login()

wandb.init(project = "RNAM Real", 
           entity = "mimetic-rna", 
           name = args.project_name,
           resume = 'Allow',
           #notes = "Se entrena la primer prueba",
           id = args.project_name)
wandb.config = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "units": args.units,
    "learning_rate":0.001,
}

#   This function is used to normalized values.
def normalizer(angle, action):
    min_val, max_val = -120, 120
    if action == 'norm':
        return (angle - min_val)/(max_val - min_val)
    else:
        return angle*(max_val - min_val)+min_val

#   This function plots training loss vs validation loss. 
def plot_loss (history):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Train vs Validation Loss for GRU network')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(args.loss_name)

#   This function plots the actual output vs the output predicted by the model. 
def plot_future(prediction, y_test):
    plt.figure(figsize=(10, 6))
    range_future = np.arange(prediction.shape[1])
    plt.plot(range_future, y_test[0,:], label='Test data', color = [1, 0.502, 0])
    plt.plot(range_future, prediction[0,:], label='Prediction',  color = [0.0502, 0.706, 0.949])
    plt.title('Predicted and true output')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Ángulo (°)')
    plt.legend(loc='lower right')
    plt.savefig(args.predict_name)

#   This function calculates performance metrics for regression problems.
def evaluate_prediction(predictions, actual):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print('GRU:')
    print('Mean Absolute Error: {:.4f}.'.format(mae))
    print('Mean Square Error: {:.4f}.'.format(mse))
    print('Root Mean Square Error: {:.4f}.'.format(rmse))

#   Separate the values in train, validation and test data/label.
def separate_values(X_train, Y_train):
    train_data, val_data, test_data = [], [], []
    train_label, val_label, test_label = [], [], []
    train_lenght = int(len(X_train)*3/5)
    val_lenght = int(len(X_train)*4/5)
    
    #   Use 3/5 of the total data set for training
    #   and 1/5 for validation and testing.
    for i,j in zip(X_train[:train_lenght], Y_train[:train_lenght]):
        train_data.append(i)
        train_label.append(j)
    
    for i,j in zip(X_train[train_lenght:val_lenght], Y_train[train_lenght:val_lenght]):
        val_data.append(i)
        val_label.append(j)
    
    for i,j in zip(X_train[val_lenght:], Y_train[val_lenght:]):
        test_data.append(i)
        test_label.append(j)
    
    train_data, val_data, test_data = np.array(train_data), np.array(val_data), np.array(test_data)
    train_label, val_label, test_label = np.array(train_label), np.array(val_label), np.array(test_label)
    
    return train_data, train_label, val_data, val_label, test_data, test_label

#   Read all the .csv files and make an nx4 array.
#   Next, separate the pwm value and angle in their respective arrays.
root = '../Datos_Recolectados/'
Dir = os.listdir(root)
pwm = np.array([])
angle = np.array([])

print('******************* Process the Dataset *******************', flush = True)
print('Recolecting Data', flush = True)
for filename in Dir:
    files = pd.read_csv(root + filename)
    pwm = np.append(pwm, np.concatenate((np.zeros(1500), files.values[:,2])))
    angle = np.append(angle, np.concatenate((np.zeros(1500), files.values[:,3])))

train_data, train_label, val_data, val_label, test_data, test_label = separate_values(pwm, angle)

#   ***************** Normalize values *****************
print('Normalizing Data',flush=True)
train_label = normalizer(train_label, 'norm')
val_label = normalizer(val_label, 'norm')
test_label = normalizer(test_label, 'norm')

#   ***************** Create the training/validation/test data *****************
print('Accommodating data for the GRU network',flush=True)

input_seq_train = train_data    # Input sequence for the train.
input_seq_val = val_data         # Input sequence for validation.
input_seq_test = test_data       # Input sequence for test.

#   ***************** Reshape label values to (nx1) *****************
train_label = np.reshape(train_label, (train_label.shape[0],1))    # Normalize angle and reshape (nx1).
val_label = np.reshape(val_label, (val_label.shape[0],1))            # Normalize angle and reshape (nx1).
test_label = np.reshape(test_label, (test_label.shape[0],1))        # Normalize angle and reshape (nx1).

y_train = np.reshape(train_label, (1, train_label.shape[0], 1))       # Label train data.
y_val = np.reshape(val_label, (1, val_label.shape[0], 1))              # Label train data.
y_test = np.reshape(test_label, (1, test_label.shape[0], 1))          # Label train data.

input_seq_train = np.reshape(input_seq_train,(input_seq_train.shape[0], 1))
input_seq_val = np.reshape(input_seq_val,(input_seq_val.shape[0], 1))
input_seq_test = np.reshape(input_seq_test,(input_seq_test.shape[0], 1))

tmp_train = np.concatenate((input_seq_train, np.zeros(shape = (input_seq_train.shape[0], 1))), axis=1)
tmp_val = np.concatenate((input_seq_val, np.zeros(shape = (input_seq_val.shape[0], 1))), axis=1)
tmp_test = np.concatenate((input_seq_test, np.zeros(shape=(input_seq_test.shape[0], 1))), axis=1)

X_train= np.reshape(tmp_train, (1, tmp_train.shape[0], tmp_train.shape[1]))     # Train Data.
X_val = np.reshape(tmp_val, (1, tmp_val.shape[0], tmp_val.shape[1]))             # Validation Data.
X_test = np.reshape(tmp_test, (1, tmp_test.shape[0], tmp_test.shape[1]))          # Test Data.

print('Total train data is: ', X_train.shape[1], flush=True)
print('Total validation data is: ', X_val.shape[1], flush=True)
print('Total testing data is:: ', X_test.shape[1], flush=True)

print('\nTrain data shape is: ', X_train.shape, flush=True)
print('Validation data shape is: ', X_val.shape, flush=True)
print('Testing data shape is:: ', X_test.shape, flush=True)
print('Train label shape is: ', y_train.shape, flush=True)
print('Validation label shape is: ', y_val.shape, flush=True)
print('Testing label shape is:: ', y_test.shape, flush=True)
print('******************* Finish *******************',flush=True)

#   ***************** Neuronal Network *****************
#   Model Creation
model = Sequential()
model.add(GRU(units=wandb.config['units'], input_shape=(None, X_train.shape[2]), use_bias=True, return_sequences=True))
#model.add(GRU(units=wandb.config['units'], return_sequences=True))

# Hidden Layer
model.add(Dense(1))

#   Compile model
if (args.loss_type == 'mae'):
    model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mae'])
else:
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])
    
model.summary()

if (args.load_model != ''):
    model.load_weights(args.load_model)

#   Train Model
history = model.fit(X_train, y_train ,
                    epochs = wandb.config['epochs'], batch_size = wandb.config['batch_size'], 
                    validation_data = (X_val, y_val),
                    verbose = 1, callbacks=[WandbCallback(save_model=False)])
model.save(args.model_name)

#   Model Prediction
testPredict = model.predict(X_test)

#   Desnormalize values
testPredict = normalizer(testPredict, '')
y_test = normalizer(y_test, '')

#   Model Evaluate
plot_loss(history)
plot_future(testPredict, y_test)
evaluate_prediction(testPredict, y_test)
