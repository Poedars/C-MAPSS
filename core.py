"""从两个MIT项目摘录的核心函数。除np.int→int外保留原逻辑和函数名称。
具体文件、版本与位置见SOURCES.md；许可见upstream/。
"""
import numpy as np
import pandas as pd
from random import shuffle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
window_length = 30
def calc_training_rul(df):
    # Data Labeling - generate column RUL
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

def split_data(df, randomize=True, train_pct=0.8):
    """Given a pandas DataFrame of time series data that is organized with a unique 'id' column,
    split the data into a training and a validation set.

    Args:
        df (DataFrame) - Data to split
        randomize (bool) -- Shuffle the order of the engines while retaining the times series order (default True)
        train_pct (float) -- Percentage as float to allocate as training data.

    Returns:
        DataFrame: Training data DataFrame
        DataFrame: Validation data DataFrame
    """

    all_ids = df['id'].unique()

    if randomize:
        shuffle(all_ids)

    num_train_samples = int(all_ids.shape[0] * train_pct)
    train_sample_ids = all_ids[:num_train_samples]
    test_sample_ids = all_ids[num_train_samples:]

    train_data_df = df[df['id'].isin(train_sample_ids)]
    val_data_df = df[df['id'].isin(test_sample_ids)]

    assert df['id'].unique().shape[0] == train_data_df['id'].unique().shape[0] + val_data_df['id'].unique().shape[0]

    return train_data_df, val_data_df

def process_input_data_with_targets(input_data, target_data = None, window_length = 1, shift = 1):
    """Depending on values of window_length and shift, this function generates batchs of data and targets 
    from input_data and target_data.
    
    Number of batches = np.floor((len(input_data) - window_length)/shift) + 1
    
    **We don't check input dimensions uisng exception handling. So readers should be careful while using these
    functions. If input data are not of desired dimension, either error occurs or something undesirable is 
    produced as output.**
    
    Arguments:
        input_data: input data to function (Must be 2 dimensional)
        target_data: input rul values (Must be 1D array)s
        window_length: window length of data
        shift: Distance by which the window moves for next batch. This is closely related to overlap
               between data. For example, if window length is 30 and shift is 1, there is an overlap of 
               29 data points between two consecutive batches.
        
    """
    num_batches = int(np.floor((len(input_data) - window_length)/shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches, window_length,
                                                                                                  num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats = num_batches)
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
            output_targets[batch] = target_data[(shift*batch + (window_length-1))]
        return output_data, output_targets

def create_compiled_model():
    model = Sequential([
        layers.LSTM(128, input_shape = (window_length, 14), return_sequences=True, activation = "tanh"),
        layers.LSTM(64, activation = "tanh", return_sequences = True),
        layers.LSTM(32, activation = "tanh"),
        layers.Dense(96, activation = "relu"),
        layers.Dense(128, activation = "relu"),
        layers.Dense(1)
    ])
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.0001

def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
