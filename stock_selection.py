import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model
from keras import regularizers, optimizers

from sklearn import preprocessing


def defineAutoencoder(num_stock, encoding_dim=5, verbose=0):
    """
    Function for fitting an Autoencoder
    """

    # connect all layers
    input = Input(shape=(num_stock,))

    encoded = Dense(encoding_dim, kernel_regularizer=regularizers.l2(
        0.00001), name='Encoder_Input')(input)

    decoded = Dense(num_stock, kernel_regularizer=regularizers.l2(
        0.00001), name='Decoder_Input')(encoded)
    decoded = Activation("linear", name='Decoder_Activation_function')(decoded)

    # construct and compile AE model
    autoencoder = Model(inputs=input, outputs=decoded)
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
#    adam = optimizers.Adam(lr=0.0005)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    if verbose != 0:
        autoencoder.summary()

    return autoencoder


def getReconstructionErrorsDF(df_pct_change, reconstructed_data):
    """
    Function for calculating the reconstruction Errors
    """
    array = []
    stocks_ranked = []
    num_columns = reconstructed_data.shape[1]
    for i in range(0, num_columns):
        # 2 norm difference
        diff = np.linalg.norm(
            (df_pct_change.iloc[:, i] - reconstructed_data[:, i]))
        array.append(float(diff))

    ranking = np.array(array).argsort()
    r = 1
    for stock_index in ranking:
        stocks_ranked.append([r, stock_index, df_pct_change.iloc[:, stock_index].name, array[stock_index]
                              ])
        r = r + 1

    columns = ['ranking', 'stock_index', 'stock_name', 'recreation_error']
    df = pd.DataFrame(stocks_ranked, columns=columns)
    df = df.set_index('stock_name')
    return df


def Autoencoder():
    prices_data = pd.read_csv('./datasets/close_prices.csv')

    df = prices_data.copy()
    df = df.reset_index(drop=True).set_index(['date'])
    col_names = df.columns.to_list()

    df_pct_change = df.pct_change(1).astype(float)
    df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
    df_pct_change = df_pct_change.fillna(method='ffill')
    df_pct_change = df_pct_change.tail(len(df_pct_change) - 2)

    df_pct_change = df_pct_change[df_pct_change.columns[(
        (df_pct_change == 0).mean() <= 0.05)]]

    hidden_layers = 5
    batch_size = 500
    epochs = 500
    stock_selection_number = 20
    num_stock = df_pct_change.shape[1]
    verbose = 0

    df_scaler = preprocessing.MinMaxScaler()
    df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)
    num_stock = len(df_pct_change.columns)
    autoencoder = defineAutoencoder(
        num_stock=num_stock, encoding_dim=hidden_layers, verbose=verbose)
    autoencoder.fit(df_pct_change_normalised, df_pct_change_normalised, shuffle=False, epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose)
    reconstruct = autoencoder.predict(df_pct_change_normalised)
    reconstruct_real = df_scaler.inverse_transform(reconstruct)
    df_reconstruct_real = pd.DataFrame(
        data=reconstruct_real, columns=df_pct_change.columns)
    df_recreation_error = getReconstructionErrorsDF(df_pct_change=df_pct_change,
                                                    reconstructed_data=reconstruct_real)
    # print(df_recreation_error)
    filters = df_recreation_error
    filters.to_csv('datasets/filters.csv')
    # filtered_stocks = df_recreation_error.head(stock_selection_number).index


if __name__ == "__main__":
    Autoencoder()
