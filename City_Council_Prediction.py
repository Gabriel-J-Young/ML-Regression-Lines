from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn import metrics 
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
#pd.options.display.float_format = '{:.1f}'.format


election_dataframe = pd.read_excel('City_Council_elections_features.xlsx')
future_election_dataframe = pd.read_excel('City_Council_elections_labels.xlsx')

print(election_dataframe)
print(future_election_dataframe)
#print(election_dataframe.describe())

def preprocess_features(election_dataframe):
    """Prepares input features from the election_dataframe dataset

    Args:
        election_dataframe: A Pandas Dataframe that contains data 
            from the Seattle Council Elections Dataset
    Returns:
        A Dataframe that contains the features to be used for the model.
    """
    selected_features = election_dataframe[
    ["Age",
     "Contributions_as_percent_of_total",
     "Contributors_as_percent_of_total"]]
    return selected_features

def preprocess_targets(election_dataframe):
    """Prepares target features (lables) from the election_dataframe dataset

    Args:
        election_dataframe: A Pandas Dataframe that contains data 
            from the Seattle Council Elections Dataset
    Returns:
        A Dataframe that contains the labels to be used for the model.
    """
    output_targets =pd.DataFrame()
    output_targets["Vote"] = election_dataframe["Vote"]
    return output_targets

def postprocess_features(future_election_dataframe):
    """Prepares input features from the future_election_dataframe dataset

    Args:
        election_dataframe: A Pandas Dataframe that contains data 
            from the Seattle Council Elections Dataset 2019
    Returns:
        A Dataframe that contains the features to be used for the model predictions.
    """
    selected_features = future_election_dataframe[
    ["Age",
     "Contributions_as_percent_of_total",
     "Contributors_as_percent_of_total"]]
    
    return selected_features

def postprocess_targets(future_election_dataframe):
    """Prepares target features (lables) from the future_election_dataframe dataset

    Args:
        election_dataframe: A Pandas Dataframe that contains data 
            from the Seattle Council Elections Dataset
    Returns:
        A Dataframe that contains the labels to be used for the model predictions.
    """
    output_targets =pd.DataFrame()
    output_targets["Vote"] = future_election_dataframe["Vote"]
    
    return output_targets

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of mulitple features.

    Args:
        features: pandas Dataframe of features
        targets:pandas Dataframe of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True/False. Whether to shuffle the data.
        num_epochs: Number of epochs for which the data should be repeated. None = repeat indefinitely
     Returns:
        Tuple of (features, labels) for next data batch
    """

    #Convert panda data into a dict of np arrays.
    features = {key:np.array(value) for key, value in dict(features).items()}

    #Construct a dataset and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets)) #warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    #Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(10000)

    #Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
    """Construct the TensorFlow Featues Columns.

    Args:
        input_features: The names of the numerical input features to use.
    Returns: A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def train_model(
    learning_rate,
    steps,
    batch_size,
    preprocess_features,
    preprocess_targets,
    postprocess_features,
    postprocess_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    preprocess_features: A `DataFrame` containing one or more columns from
      `election_dataframe` to use as input features for training and validation.
    preprocess_targets: A `DataFrame` containing exactly one column from
      `election_dataframe` to use as target for training and validation.
    postprocess_features: A 'DataFrame' containing one or more columns from
      'future_election_dataframe' to use as features to predict labels with
    postprocess_targets: 'DataFrame' containing exactly one column from 
      'future_election_dataframe' to fill with labels
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  #Creates a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(preprocess_features),
      optimizer=my_optimizer
      )

  # 1. Create input functions.
  training_input_fn = lambda: my_input_fn(
      preprocess_features,
      preprocess_targets["Vote"],
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
     preprocess_features,
     preprocess_targets["Vote"],
     num_epochs=1,
     shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      preprocess_features, 
      preprocess_targets["Vote"],
      num_epochs=1,
      shuffle=False)
  vote_predictions_fn = lambda: my_input_fn(
      postprocess_features,
      postprocess_targets["Vote"],
      num_epochs=1,
      shuffle=False)

  #Train the model, but inside a loop so we can assess loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    #Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    #Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    #Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, preprocess_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, preprocess_targets))
    #Occasionally print the current loss.
    print(" period %02d: %0.2f" % (period, training_root_mean_squared_error))
    #Add the loss metrics from this period to out list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error) 
  print("Model training finished.")
  print("Generating Predictions...")
  vote_predictions = linear_regressor.predict(input_fn=vote_predictions_fn)
  vote_predictions = np.array([item['predictions'][0] for item in vote_predictions])
  print(vote_predictions)
  future_election_dataframe["Vote"] = vote_predictions

  #Output a graph of loss metrics over periods.
  print(plt.ylabel("RMSE"))
  print(plt.xlabel("Periods"))
  print(plt.title("Root Mean Squared Error vs. Periods"))
  print(plt.tight_layout())
  print(plt.plot(training_rmse, label="training"))
  print(plt.plot(validation_rmse, label="validation"))
  print(plt.legend())
  #plt.show()

  return linear_regressor

linear_regressor = train_model(
    learning_rate = 0.0001,
    steps = 50000,
    batch_size = 50,
    preprocess_features=preprocess_features(election_dataframe),
    preprocess_targets=preprocess_targets(election_dataframe),
    postprocess_features = postprocess_features(future_election_dataframe),
    postprocess_targets = postprocess_targets(future_election_dataframe)
    )
print(future_election_dataframe)

