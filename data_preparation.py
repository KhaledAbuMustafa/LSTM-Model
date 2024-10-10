import numpy as np
import os


class Data_prep:

    target_steps = 1
    def datasets(self, X, y):
        X, y = self.df_to_X_y(X,y)
        q_70 = int(0.7 * len(X))
        q_85 = int(0.85 * len(X))
        X_train, y_train = X[:q_70], y[:q_70]
        X_val, y_val = X[q_70:q_85], y[q_70:q_85]
        X_test, y_test = X[q_85:], y[q_85:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def df_to_X_y(self, X,y):
        window_size = 32
        X_new = []
        y_new = []
        for i in range(len(y) - window_size-self.target_steps+1):
            row = [[a] for a in y[i:i + window_size]]
            X_new.append(row)
            y_new.append([y[i + window_size]])


        return np.array(X_new), np.array(y_new)