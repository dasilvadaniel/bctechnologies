# -*- coding: utf-8 -*-

'''
# To run this code on Google Colab you need to use the following code to use Tensorflow 2.0
# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')

from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
except Exception:
  print("ok")
  pass
'''

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor

#These are some ploting and dataframes visualization parameters
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None 
tf.random.set_seed(13)

#This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
#of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot. Adapted from: https://github.com/suetAndTie/ClarkeErrorGrid
def clarke_error_grid(rv, pv, title_string):

  #Checking to see if the lengths of the reference and prediction arrays are the same
  assert (len(rv) == len(pv)), 'Unequal number of values (reference : {}) (prediction : {}).'.format(len(rv), len(pv))

  #Clear plot
  plt.clf()

  #Build the plot
  for ref_values, pred_values in zip(rv, pv):
    
    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', s=2, color='cyan')
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

  #Set axes lengths
  plt.gca().set_xlim([0, 400])
  plt.gca().set_ylim([0, 400])
  plt.gca().set_aspect((400)/(400))

  #Plot zone lines
  plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
  plt.plot([0, 175/3], [70, 70], '-', c='black')
  #plt.plot([175/3, 320], [70, 400], '-', c='black')
  plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
  plt.plot([70, 70], [84, 400],'-', c='black')
  plt.plot([0, 70], [180, 180], '-', c='black')
  plt.plot([70, 290],[180, 400],'-', c='black')
  # plt.plot([70, 70], [0, 175/3], '-', c='black')
  plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
  # plt.plot([70, 400],[175/3, 320],'-', c='black')
  plt.plot([70, 400], [56, 320],'-', c='black')
  plt.plot([180, 180], [0, 70], '-', c='black')
  plt.plot([180, 400], [70, 70], '-', c='black')
  plt.plot([240, 240], [70, 180],'-', c='black')
  plt.plot([240, 400], [180, 180], '-', c='black')
  plt.plot([130, 180], [0, 70], '-', c='black')

  #Add zone titles
  plt.text(30, 15, "A", fontsize=15)
  plt.text(370, 260, "B", fontsize=15)
  plt.text(280, 370, "B", fontsize=15)
  plt.text(160, 370, "C", fontsize=15)
  plt.text(160, 15, "C", fontsize=15)
  plt.text(30, 140, "D", fontsize=15)
  plt.text(370, 120, "D", fontsize=15)
  plt.text(30, 370, "E", fontsize=15)
  plt.text(370, 15, "E", fontsize=15)
  
  #Statistics from the data
  zone = [0] * 5

  #check if rv and pv are np array used for the overall clarke grid
  #else, a list is passed, which is used for each step clarke grid
  if (np.array(rv).ndim > 1) and (np.array(pv).ndim > 1):

    for ref_values, pred_values in zip(rv,pv):
      
      for i in range(len(ref_values)):
          if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
              zone[0] += 1    #Zone A

          elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
              zone[4] += 1    #Zone E

          elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
              zone[2] += 1    #Zone C
          elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
              zone[3] += 1    #Zone D
          else:
              zone[1] += 1    #Zone B

  else:

    ref_values = rv
    pred_values = pv

    for i in range(len(ref_values)):
      if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
          zone[0] += 1    #Zone A

      elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
          zone[4] += 1    #Zone E

      elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
          zone[2] += 1    #Zone C
      elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
          zone[3] += 1    #Zone D
      else:
          zone[1] += 1    #Zone B
  
  return plt, zone

#dataset normalization
def normalization(data,mean,std):
  return (data-mean)/std

#dataset inverse normalization
def inv_normalization(data,mean,std):
  return (data-(-mean/std))/(1/std)

# evaluate one or more weekly forecasts against expected values for tensorflow models
def tf_evaluate_forecasts(model, X, y):

  print("Number of predictions to make: ", len(X))
  print("Number of data points (12/H): ", len(X)*12)

  #make predictions
  y_pred = model.predict(X)
  
  # calculate an RMSE score / Clarke Grid for each 5 minutes
  step_minute = 5
  for i in range(y.shape[1]):

      y_inv_i = [inv_normalization(val,data_mean[0],data_std[0]) for val in y[:,i]]
      y_pred_inv_i = [inv_normalization(val,data_mean[0],data_std[0]) for val in y_pred[:,i]]

      plt_i, zones_i = clarke_error_grid(y_inv_i, 
                                         y_pred_inv_i, 
                                         'Prediction for {a} minutes'.format(a=step_minute))
      
      plt_i.show()
      
      #Calculate rmse with/without normalization
      mse_i = mean_squared_error(y[:, i], y_pred[:, i])
      mse_inv_i = mean_squared_error(y_inv_i, y_pred_inv_i)
      rmse_i = sqrt(mse_i)
      rmse_inv_i = sqrt(mse_inv_i)

      #Compute accuracy for each zone
      a_accuracy_i = (sum(zones_i[:1])/sum(zones_i))*100
      ab_accuracy_i = (sum(zones_i[:2])/sum(zones_i))*100
      abc_accuracy_i = (sum(zones_i[:3])/sum(zones_i))*100
      abcd_accuracy_i = (sum(zones_i[:4])/sum(zones_i))*100
      abcde_accuracy_i = (sum(zones_i[:5])/sum(zones_i))*100
      print('Data points by zone: ', zones_i)
      print('RMSE normalized for {a} minutes: '.format(a=step_minute), rmse_i)
      print('RMSE without normalization for {a} minutes: '.format(a=step_minute), rmse_inv_i)
      print("Accuracy in zones A: ", a_accuracy_i, '%')
      print("Accuracy in zones A, B: ", ab_accuracy_i, '%')
      print("Accuracy in zones A, B, C: ", abc_accuracy_i, '%')
      print("Accuracy in zones A, B, C, D: ", abcd_accuracy_i, '%')                                     
      print("Accuracy in zones A, B, C, D, E: ", abcde_accuracy_i, '%\n\n')                                     

      step_minute += 5
 
  #Return data point to original scale
  y_inv = [inv_normalization(val,data_mean[0],data_std[0]) for val in y]
  y_pred_inv = [inv_normalization(val,data_mean[0],data_std[0]) for val in y_pred]
  
  #Calculate overall RMSE and Clarke Grid with/without normalization
  rmse = sqrt(mean_squared_error(y, y_pred))
  rmse_inv = sqrt(mean_squared_error(y_inv, y_pred_inv))
  print('Overall RMSE with normalization: ', rmse)
  print('Overall RMSE without normalization: ', rmse_inv)

  plt, zones = clarke_error_grid(y_inv, y_pred_inv, 'Overall')
  
  #Compute accuracy for each zone for all data points
  a_accuracy = (sum(zones[:1])/sum(zones))*100
  ab_accuracy = (sum(zones[:2])/sum(zones))*100
  abc_accuracy = (sum(zones[:3])/sum(zones))*100
  abcd_accuracy = (sum(zones[:4])/sum(zones))*100
  abcde_accuracy = (sum(zones[:5])/sum(zones))*100
  print('Data points by zone: ', zones)
  print("Accuracy in zones A: ", a_accuracy, '%')
  print("Accuracy in zones A, B: ", ab_accuracy, '%')
  print("Accuracy in zones A, B, C: ", abc_accuracy, '%')
  print("Accuracy in zones A, B, C, D: ", abcd_accuracy, '%')
  print("Accuracy in zones A, B, C, D, E: ", abcde_accuracy, '%')

#dataset preprocessing
def dataset_preprocessing(df_blood_glucose, df_distance_activity, df_heart_rate):

  #filters timezone_offset column. 'device' column was not used
  df_blood_glucose = df_blood_glucose[['point_timestamp','point_value(mg/dL)']]
  df_distance_activity = df_distance_activity[['point_timestamp','point_value(kilometers)']]
  df_heart_rate = df_heart_rate[['point_timestamp','point_value']]

  #changes point_timestamp format from '%Y%m%d %H:%M:%S' to '%Y%m%d %H:%M'
  #so we can join dataframes by YYYY-MM-DD HH:MM
  df_blood_glucose['point_timestamp'] = pd.to_datetime(df_blood_glucose.point_timestamp).dt.strftime('%Y%m%d %H:%M')
  df_distance_activity['point_timestamp'] = pd.to_datetime(df_distance_activity.point_timestamp).dt.strftime('%Y%m%d %H:%M')
  df_heart_rate['point_timestamp'] = pd.to_datetime(df_heart_rate.point_timestamp).dt.strftime('%Y%m%d %H:%M')

  #drops point_timestamp duplicates and sets point_timestamp as index
  #some records were duplicated as the second was removed from the 
  #timestamp, so only the first record was kept
  df_blood_glucose = df_blood_glucose.drop_duplicates(subset=['point_timestamp'], keep = 'first').set_index('point_timestamp')
  df_distance_activity = df_distance_activity.drop_duplicates(subset=['point_timestamp'], keep = 'first').set_index('point_timestamp')
  df_heart_rate = df_heart_rate.drop_duplicates(subset=['point_timestamp'], keep = 'first').set_index('point_timestamp')

  #Join dataset and fill NaN with 0
  df = df_blood_glucose.join(df_heart_rate).join(df_distance_activity).fillna(0)

  #Features that will be considered
  features_considered = ['point_value(mg/dL)',
                        'point_value',
                        'point_value(kilometers)']
  df = df[features_considered]

  #Plot used distributions
  axes = df.plot(subplots=True, layout=(3,1))

  #Dataset to array
  dataset = df.values
  data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
  data_std = dataset[:TRAIN_SPLIT].std(axis=0)

  #Dataset normalization
  dataset = normalization(dataset,data_mean,data_std)

  return dataset, data_mean, data_std, df

'''

  The following functions are used in the evaluation of a LSTM model from Tensorflow 

'''

def train_test_val_split(dataset,
                        TRAIN_SPLIT,
                        VALIDATION_SPLIT,
                        STEP,
                        BATCH_SIZE,
                        BUFFER_SIZE,
                        past_history,
                        future_target):                                                                      

  #train, test,  and validation split for multivariate timeseries                         
  x_train, y_train = multivariate_data(dataset, dataset[:, 0], 0,
                                                  TRAIN_SPLIT, past_history,
                                                  future_target, STEP)

  x_val, y_val = multivariate_data(dataset, dataset[:, 0], TRAIN_SPLIT, 
                                              VALIDATION_SPLIT, past_history,
                                              future_target, STEP)

  x_test, y_test = multivariate_data(dataset, dataset[:, 0], VALIDATION_SPLIT, 
                                              None, past_history,
                                              future_target, STEP)

  print ('Single window of past history : {}'.format(x_train[0].shape))
  print ('\n Target blood glucose to predict : {}'.format(y_train[0].shape))

  #Train, test, and validation datasets which can be used for batch/distributed processing
  train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_data = val_data.batch(BATCH_SIZE).repeat()

  test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_data = test_data.batch(BATCH_SIZE).repeat()

  return (x_train, y_train, x_val, y_val, x_test, y_test,
          train_data, val_data, test_data)

#Build timeseries windows
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):

  data = []
  labels = []

  #Sliding time windows according to history size to use
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])
  
  return np.array(data), np.array(labels)

#Return steps from history to plot in the chart
def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

#Plot examples from timeseries predictions
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()  

def multi_step_plot(history, true_future, prediction):

  #inverse normalization of blood glucose history/true_future/prediction
  history = inv_normalization(history,data_mean[0],data_std[0])
  true_future = inv_normalization(true_future,data_mean[0],data_std[0])
  prediction = inv_normalization(prediction,data_mean[0],data_std[0])

  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  
  plt.figure(figsize=(12, 6))
  plt.plot(num_in, np.array(history[:, 0]), label='History')
  plt.axhline(y=70, color='r', linestyle='-', label='Healthy Range')
  plt.axhline(y=140, color='r', linestyle='-')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')

  plt.legend(loc='upper left')
  plt.show()

def tfcallback(monitor, patience, restore_best_weights,
               checkpoint, logs, save_weights_only, save_best_only): 
    
    callback = [tf.keras.callbacks.ModelCheckpoint(
                                        path_dev + checkpoint, 
                                        save_weights_only = save_weights_only,
                                        save_best_only = save_best_only,
                                        verbose=0),
                tf.keras.callbacks.TensorBoard(
                                        log_dir=(path_dev + logs).replace('/','\\'),
                                        histogram_freq=0,
                                        write_grads=1,
                                        write_graph=True, 
                                        write_images=True),
                tf.keras.callbacks.History(),
                tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                        min_delta=0, 
                                        patience=patience, 
                                        verbose=1, 
                                        mode='auto', 
                                        baseline=None,
                                        restore_best_weights = restore_best_weights
                                        )]
    return callback     

def create_model(optimizer, loss, input_shape, size, metrics, evaluation_interval,
                 epochs, steps_per_epoch, x_val, y_val, val_steps, 
                 callbacks, dropout, regularizer, activation,  
                 train_data, val_data,x_train, y_train):

  model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32,
                                return_sequences=True,
                                input_shape=input_shape),
            tf.keras.layers.LSTM(size, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(size, 
                                  kernel_regularizer=regularizer, 
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(12)])
  
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  model.save(path_dev + "/bioconscious_model.h5")

  model.fit(train_data,
             #x_train, #these data can be used if batches are not provided
             #y_train, #these data can be used if batches are not provided
             epochs=epochs,
             steps_per_epoch=evaluation_interval,
             validation_data= val_data, #(x_val, y_val) this tuple can be used if batches are not provided
             validation_steps = val_steps,
             callbacks = callback)

  _, tf_rmse = model.evaluate(x_test, y_test)

  plot_train_history(callback[2], 'Multi-Step Training and validation loss')

  return model

'''
  The following functions are used in the evaluation of Scikit-learn models

'''

# split a univariate dataset into train/test sets
def split_dataset(data):
  
  '''

  dataset split values are specific to this experiment. 
  The number of windows in training and testing must be divisible by 12
  For example:
  - Split into standard weeks: index 8699 (train), index 8700 (test) for 14000 rows
  - Split into standard weeks: index 871 (train), index 888 (test) for 1400 rows

  '''

  train, test = data[1:-8699], data[-8700:-12]
  # restructure into windows of data
  train = np.array(np.split(train, len(train)/12))
  test = np.array(np.split(test, len(test)/12))
  return train, test

# evaluate one or more steps forecasts against expected values
def sk_evaluate_forecasts(actual, predicted):
  scores = list()
  # calculate an RMSE score for each step
  for i in range(actual.shape[1]):
    # calculate mse
    mse = mean_squared_error(actual[:, i], predicted[:, i])
    # calculate rmse
    rmse = sqrt(mse)
    # store
    scores.append(rmse)
  
  print("\n\n\nModel: ", model)
  
  plt, zones = clarke_error_grid(actual, predicted, '')
  plt.show()
  
  a_accuracy = (sum(zones[:1])/sum(zones))*100
  ab_accuracy = (sum(zones[:2])/sum(zones))*100
  abc_accuracy = (sum(zones[:3])/sum(zones))*100
  abcd_accuracy = (sum(zones[:4])/sum(zones))*100
  abcde_accuracy = (sum(zones[:5])/sum(zones))*100
  print('Data points by zone: ', zones)
  print("Accuracy in zones A: ", a_accuracy, '%')
  print("Accuracy in zones A, B: ", ab_accuracy, '%')
  print("Accuracy in zones A, B, C: ", abc_accuracy, '%')
  print("Accuracy in zones A, B, C, D: ", abcd_accuracy, '%')
  print("Accuracy in zones A, B, C, D, E: ", abcde_accuracy, '%')
  # calculate overall RMSE
  s = 0
  for row in range(actual.shape[0]):
    for col in range(actual.shape[1]):
      s += (actual[row, col] - predicted[row, col])**2
  score = sqrt(s / (actual.shape[0] * actual.shape[1]))
  return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# prepare a list of ml models
def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	models['lasso'] = Lasso()
	models['ridge'] = Ridge()
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	print('Defined %d models' % len(models))
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# # convert windows of steps multivariate data into a series of total steps
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = np.array(series).flatten()
	return series

# convert history into inputs and outputs
def to_supervised(history, n_input, output_ix):
	# convert history to a univariate series
	data = to_series(history)
	X, y = list(), list()
	ix_start = 0
	# step over the entire history one time step at a time
	for i in range(len(data)):
		# define the end of the input sequence
		ix_end = ix_start + n_input
		ix_output = ix_end + output_ix
		# ensure we have enough data for this instance
		if ix_output < len(data):
			X.append(data[ix_start:ix_end])
			y.append(data[ix_output])
		# move along one time step
		ix_start += 1
	return np.array(X), np.array(y)

# fit a model and make a forecast
def sklearn_predict(model, history, n_input):
  yhat_sequence = list()
  # fit a model for each forecast day
  for i in range(12):
    # prepare data
    train_x, train_y = to_supervised(history, n_input, i)
    # make pipeline
    pipeline = make_pipeline(model)
    # fit the model
    pipeline.fit(train_x, train_y)
    # forecast
    x_input = np.array(train_x[-1, :]).reshape(1,n_input)
    yhat = pipeline.predict(x_input)[0]
    # store
    yhat_sequence.append(yhat)
  return yhat_sequence

# evaluate a single model
def evaluate_model(model, train, test, n_input):
  # history is a list of steps data
  history = [x for x in train]
  
  # walk-forward validation over steps
  predictions = list()
  for i in range(len(test)):
    # predict the week
    yhat_sequence = sklearn_predict(model, history, n_input)
    # store the predictions
    predictions.append(yhat_sequence)
    # get real observation and add to history for predicting the next steps
    history.append(test[i, :])
  
  predictions = np.array(predictions)

  # evaluate predictions
  score, scores = sk_evaluate_forecasts(test[:, :, 0], predictions)

  return score, scores

tf.random.set_seed(13)

TRAIN_SPLIT = 6000        #resulting in 4.000 windows for training and 2000 for test
VALIDATION_SPLIT = 10000  #resulting in over 2000 windows for test
past_history = 2000       #How many data points will be assessed to 
                          #predict the next 12 data points
STEP = 1                  #How long each forecast will be made. Since
                          #the dataset has 5 minute intervals, each 
                          #step consists of 5 minutes
future_target = 12        #How many steps forward will be predicted. Since
                          #the goal is to predict 60 minutes, 
                          #12 steps of 5 minutes each will be predicted.
BATCH_SIZE = 200
BUFFER_SIZE = 100

#If you are running on Windows, be careful of the path slashes
#Data and models paths must be specified according to your environment
abs_path = os.path.abspath(os.path.curdir)

path_dev = (abs_path + "\models").replace('\\','/')
path_data = (abs_path + "\data").replace('\\','/')
date_time = datetime.datetime.today().strftime("%Y%m%d%H%M")

#open csv's as pandas dataframe
df_blood_glucose = pd.read_csv(path_data + '/blood-glucose-data.csv', sep = ',')#, nrows = 1400)
df_distance_activity = pd.read_csv(path_data + '/distance-activity-data.csv', sep = ',')#, nrows = 1400)
df_heart_rate = pd.read_csv(path_data + '/heart-rate-data.csv', sep = ',')#, nrows = 1400)

dataset, data_mean, data_std, df = dataset_preprocessing(df_blood_glucose, 
                                                     df_distance_activity, 
                                                     df_heart_rate)

(x_train, y_train, x_val, y_val, x_test, y_test,
 train_data, val_data, test_data) = train_test_val_split(dataset,
                                                        TRAIN_SPLIT,
                                                        VALIDATION_SPLIT,
                                                        STEP,
                                                        BATCH_SIZE,
                                                        BUFFER_SIZE,
                                                        past_history,
                                                        future_target)

#Let's take a look in the train, val, test length
print("X_train length: ", len(x_train))
print("X_val length: ", len(x_val))
print("X_test length: ", len(x_test))


"""
  --- Tensorflow predictions ---
  
  The following Tensorflow algorithm is the main algorithm of this project. 
  Scikit-learn algorithms were also evaluated at the end of the file, 
  for comparison purposes only.
  
  The algorithm were evaluated to predict blood glucose level 
  60 minutes (every 5 minutes) ahead based on a multivariate dataset:

  - Recurrent Neural Network (RNN) with two Long Short Term Memory (LSTM) layers

  The model is using the best parameters obtained from hyperparameter
  avaiable in hypertuning.py

"""

#Setting seed to ensure reproducibility.
tf.random.set_seed(13)

#If you are running on Windows, be careful of the slashes in the log and models
# paths in the callback. Incorrect paths will resulte in ProfilerNotRunningError error:
#https://github.com/tensorflow/tensorboard/issues/2279#issuecomment-512089344
callback = tfcallback(monitor = 'val_loss',
                      patience = 10,
                      checkpoint = "/cp.ckpt",
                      logs = "/logs/{a}".format(a=date_time),
                      restore_best_weights = True,
                      save_weights_only = True,
                      save_best_only = True,)

#model creation and fitting
model = create_model(x_train = x_train, 
                    y_train = y_train, 
                    optimizer = 'adam',
                    loss = 'mse',
                    input_shape = x_train.shape[-2:],
                    size = 32,
                    metrics = ['mse'],
                    evaluation_interval = 10,
                    epochs=100,
                    steps_per_epoch=10,
                    x_val = x_val,
                    y_val = y_val,
                    val_steps= future_target, 
                    callbacks = callback,
                    dropout = 0.1,
                    regularizer = tf.keras.regularizers.l1_l2(0.0001),
                    activation = 'sigmoid', 
                    train_data = train_data, 
                    val_data = val_data,
                    )

#Here are graphs for Clarke Error Grid for every 5 minute forecasts
#The test dataset has never been used and is completely new to the real test.
tf_evaluate_forecasts(model=model, X=x_test, y=y_test)

#Here we can analyze some individual predictions. 
#The nNumbers in the list are the position of the timeserie 
#in the validation dataset to be predicted
for i in [500,1000,1500]:
  multi_step_plot(x_test[i], y_test[i], model.predict(x_test)[i])

"""
  --- Scikit-learn predictions ---
  
  The following algorithms were also evaluated for comparison purposes only.
  The algorithms did not go through hyperparameter optimization process.
  The algorithms were evaluated to predict blood glucose level 
  60 minutes (every 5 minutes) ahead based on a multivariate dataset:

  - LinearRegression
  - Lasso
  - Ridge
  - ElasticNet
  - HuberRegressor
  - Lars
  - LassoLars
  - PassiveAggressiveRegressor
  - RANSACRegressor
  - SGDRegressor

  Adapted from:
  https://machinelearningmastery.com/multi-step-time-series-forecasting-with-machine-learning-models-for-household-electricity-consumption/

"""

#Number of rows in train/test
dataset = df.head(n=14000)

# split into train and test
train_sk, test_sk = split_dataset(dataset.values)
# prepare the models to evaluate
models = get_models()
n_input = 12
# evaluate each model
steps = ['5','10','15','20','25','30','35','40','45','50','55','60']
for name, model in models.items():
	# evaluate and get scores
	score, scores = evaluate_model(model, train_sk, test_sk, n_input)
	# summarize scores
	summarize_scores(name, score, scores)

