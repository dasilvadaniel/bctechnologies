# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:52:00 2019

@author: Daniel
"""

'''
# To run this code on google colab you need to use this code to use Tensorflow 2.0
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

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os
from tensorboard.plugins.hparams import api as hp

pd.options.mode.chained_assignment = None

#dataset normalization
def normalization(data,mean,std):
    return (data-mean)/std

#dataset inverse normalization
def inv_normalization(data,mean,std):
    return (data-(-mean/std))/(1/std)

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
  features = df[features_considered]

  #Plot used distributions
  axes = features.plot(subplots=True, layout=(3,1))

  #Dataset to array
  dataset = features.values
  data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
  data_std = dataset[:TRAIN_SPLIT].std(axis=0)

  #Dataset normalization
  dataset = normalization(dataset,data_mean,data_std)

  return dataset, data_mean, data_std

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

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):

  data = []
  labels = []

  #sliding time windows according to the size of the history to be used
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

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def train_test_model(hparams, train_data, val_data, x_train, y_train):

  model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32,
                                return_sequences=True,
                                input_shape=x_train.shape[-2:]),
            tf.keras.layers.LSTM(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(hparams[HP_NUM_UNITS], 
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001), 
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(12)])
  
  model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse', metrics=['mse'])

  model.fit(x_train, #these data can be used if batches are not provided
              y_train, #these data can be used if batches are not provided
              epochs=10,  
              callbacks = [tf.keras.callbacks.ModelCheckpoint(
                                        path_dev + "/cp.ckpt", 
                                        save_weights_only = True,
                                        save_best_only = True,
                                        verbose=0),
                tf.keras.callbacks.TensorBoard(
                                        log_dir=(path_dev + "/logs").replace('/','\\'),
                                        histogram_freq=0,
                                        write_grads=1,
                                        write_graph=True, 
                                        write_images=True),
                hp.KerasCallback((path_dev + "/logs").replace('/','\\'), hparams), 
                tf.keras.callbacks.History(),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, 
                                        patience=10, 
                                        verbose=1, 
                                        mode='auto', 
                                        baseline=None,
                                        restore_best_weights = True
                                        )])

  _, tf_mse = model.evaluate(x_val, y_val)

  return tf_mse

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    tf_mse = train_test_model(hparams, train_data, val_data, x_train, y_train)
    tf.summary.scalar(METRIC_MSE, tf_mse, step=1)

TRAIN_SPLIT = 600
VALIDATION_SPLIT = 1000
past_history = 200       #How many data points will be assessed to 
                          #...predict the next 12 data points
STEP = 1                  #How long each forecast will be made. Since
                          #...the dataset has 5 minute intervals, each 
                          #...step consists of 5 minutes
future_target = 12        #How many steps forward will be predicted. Since
                          #...the goal is to predict 60 minutes, 
                          #...12 steps of 5 minutes each will be predicted.
BATCH_SIZE = 200
BUFFER_SIZE = 100

#data and models paths must be specified according to your environment
abs_path = os.path.abspath(os.path.curdir)

path_dev = (abs_path + "\models").replace('\\','/')
path_data = (abs_path + "\data").replace('\\','/')
date_time = datetime.datetime.today().strftime("%Y%m%d%H%M")

#open csv's as pandas dataframe
df_blood_glucose = pd.read_csv(path_data + '/blood-glucose-data.csv', sep = ',')#, nrows = 1400)
df_distance_activity = pd.read_csv(path_data + '/distance-activity-data.csv', sep = ',')#, nrows = 1400)
df_heart_rate = pd.read_csv(path_data + '/heart-rate-data.csv', sep = ',')#, nrows = 1400)

dataset, data_mean, data_std = dataset_preprocessing(df_blood_glucose, 
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

#hyperparameter gridsearch
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_MSE = 'mse'

with tf.summary.create_file_writer(path_dev + '/logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_MSE, display_name='mse')],
  )

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run(path_dev + '/logs/hparam_tuning/' + run_name, hparams)
      session_num += 1

#The HParams dashboard can now be opened. 
#Example: \models\logs>tensorboard --logdir hparam_tuning
#Start TensorBoard and click on "HParams" at the top. 
