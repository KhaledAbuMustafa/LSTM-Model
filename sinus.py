import os
from scipy.signal import hilbert
import time
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt

# Generate random amplitude between 0.8 and 1.2
def ampl():
    return np.random.uniform(low=0.8, high=1.2)

# Generate random frequency between 0.9 and 1.1
def freq():
    return np.random.uniform(low=0.9, high=1.1)

def freq2(n):
    freq_slope = np.random.uniform(low=-0.02, high=0.02)
    initial_freq = np.random.uniform(low=0.9, high=1.1)

    return initial_freq + freq_slope * n

# Generate a random length
def length():
    n= np.random.uniform(low=10.5, high=11.5)
    rounded_number = round(n * 10) / 10
    return rounded_number

# Generate sine waves with different amplitudes and frequencies
def sine():
    n = length()
    n_100 = int(n*100)
    X = np.linspace(start=0, stop=n * np.pi,num=n_100)

    y = []
    # Sine waves with different amplitudes and frequencies
    y.append(ampl() * np.sin(X[:100] * freq()) ** 4)
    if n < 2:
        y.append(ampl() * np.sin(X[100:n_100] * freq()) ** 4)
    else:
        y.append(ampl() * np.sin(X[100:200] * freq()) ** 4)
        y.append(ampl() * np.sin(X[200:] * freq()) ** 4)

    
    array_list = [np.array(sublist) for sublist in y]

    
    y = np.concatenate(array_list)
    y = np.array([x*np.random.normal(1,0.05) for x in y]) # Measurement noise
    plt.plot(X, y)
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.grid(True)
    plt.show()
    return X, y


def sine2():
    n = length()
    n_100 = int(n*100)
    X = np.linspace(start=0, stop=n * np.pi,num=n_100)

    y = []
    y.append(ampl() * np.sin(X[:100] * freq2(n)) ** 4)
    if n < 2:
        y.append(ampl() * np.sin(X[100:n_100] * freq2(n)) ** 4)
    else:
        y.append(ampl() * np.sin(X[100:200] * freq2(n)) ** 4)
        y.append(ampl() * np.sin(X[200:] * freq2(n)) ** 4)

    indices = np.arange(2, len(y), 4)
    for i in indices:
       y[i] += np.random.choice([-0.08, 0.08])


    
    array_list = [np.array(sublist) for sublist in y]

    
    y = np.concatenate(array_list)
    y = np.array([x*np.random.normal(1,0.05) for x in y])

    # Add baseline
    slope = np.random.uniform(low=-0.01, high=0.01)
    intercept = np.random.uniform(low=-0.5, high=0.5)
    baseline = slope * X + intercept

    # Function with baseline
    y = y + baseline

    plt.plot(X, y)
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.title('sine wave')
    plt.grid(True)
    plt.show()
    return X, y



# Convert data into sequences for model input (sliding window approach)
def df_to_X_y(X,y, window_size=75):
    X_new = []
    y_new = []
    for i in range(len(y) - window_size):
        row = [[a] for a in y[i:i+window_size]]
        X_new.append(row)
        y_new.append(y[i+window_size])
    return np.array(X_new), np.array(y_new)


# Split dataset into training, validation, and test setst
def datasets(X,y):
    X_new, y_new = df_to_X_y(X, y)
    global q_70, q_80
    q_70 = int(len(X_new) * 0.7)
    q_80 = int(len(X_new) * 0.8)
    X_train, y_train = X_new[:q_70], y_new[:q_70]
    X_val, y_val = X_new[q_70:q_80], y_new[q_70:q_80]
    X_test, y_test = X_new[q_80:], y_new[q_80:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Build and train the model
def model(X_train, y_train, X_val, y_val,X, X_test, y_test):

    model = tf.keras.Sequential([layers.Input((75,1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dropout(0.2),
                        layers.Dense(1)])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    plt.plot(X[:q_70], train_pred, label='train_pred')
    plt.plot(X[q_70:q_80], val_pred, label='val_pred')
    plt.plot(X[q_80:-75], test_pred, label='test_pred')
    plt.plot(X[:q_70], y_train, label='train_original')
    plt.plot(X[q_70:q_80], y_val, label='val_original')
    plt.plot(X[q_80:-75], y_test, label='test_original')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()
    return train_pred, val_pred, test_pred



def run_pred(X,y):
    X_train, y_train, X_val, y_val, X_test, y_test = datasets(X, y)

    train_pred, val_pred, test_pred = model(X_train, y_train, X_val, y_val, X, X_test, y_test)

    print(mean_squared_error(y_train, train_pred))
    print(mean_squared_error(y_val, val_pred))
    print(mean_squared_error(y_test, test_pred))




X,y = sine()
run_pred(X,y)
