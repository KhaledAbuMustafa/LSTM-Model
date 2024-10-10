import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import Data_prep
from tensorflow.keras.layers import Bidirectional
from tuner import Tuner
import time





class ModelGenerator:
        def load_model(self, X_train, y_train, X_val, y_val, X, X_test, y_test,y):
                ws = 32
                data_prep = Data_prep()
                loaded_model = tf.keras.models.load_model('final_tune_weirdos_pls.h5')
                

                loaded_model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=1e-6),
                                metrics=['mean_absolute_error'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                start_time = time.time()
                #loaded_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, callbacks=[early_stopping])

                train_pred = loaded_model.predict(X_train)
                val_pred = loaded_model.predict(X_val)
                test_pred = loaded_model.predict(X_test)
                end_time = time.time()
                prediction_time = end_time - start_time
                print(f"Prediction Time: {prediction_time}")

                
                plt.figure(figsize=(14, 7))
                plt.subplot(3, 1, 1)
                plt.plot(X[:q_70], y_train[:, 0], label='True Value  (Train)')
                plt.plot(X[:q_70], train_pred[:, 0], label='Predicted Value  (Train)', linestyle='dashed')
                plt.title('Train data')
                plt.grid(True)
                plt.xlabel('time[s]')
                plt.ylabel('position[mm]')
                plt.legend(loc="upper left")

                plt.subplot(3, 1, 2)
                plt.plot(X[q_70:q_85], y_val[:, 0], label='True Value ')
                plt.plot(X[q_70:q_85], val_pred[:, 0], label='Predicted Value  ', linestyle='dashed')
                plt.title('Validation Data')
                plt.grid(True)
                plt.xlabel('time[s]')
                plt.ylabel('position[mm]')
                plt.legend(loc="upper left")


                plt.subplot(3, 1, 3)
                plt.plot(X[q_85:-ws - data_prep.target_steps + 1], y_test[:, 0], label='True Value ')
                plt.plot(X[q_85:-ws - data_prep.target_steps + 1], test_pred[:, 0], label='Predicted Value ',
                        linestyle='dashed')
                plt.title('Test Data')
                plt.xlabel('time[s]')
                plt.ylabel('position[mm]')
                plt.grid(True)
                plt.legend(loc="upper left")

                plt.tight_layout()
                plt.show()

                return train_pred, val_pred, test_pred

        def run_pred(self,X,y):
                ws = 32
                data_prep = Data_prep()
                tuner = Tuner()
                global future
                future = data_prep.target_steps - 1
                global q_70, q_85
                q_70 = int(0.7 * (len(X)-ws-data_prep.target_steps+1))
                q_85 = int(0.85 * (len(X) - ws - data_prep.target_steps + 1))
                X_train, y_train, X_val, y_val, X_test, y_test = data_prep.datasets(X, y)
                #model = self.build_model(X_train,y_train,X_val,y_val)
                train_pred, val_pred, test_pred = self.load_model(X_train, y_train, X_val, y_val, X, X_test,y_test,y)

                mean_err_train = mean_absolute_error(y_train[:,0], train_pred[:,0])
                mean_err_val = mean_absolute_error(y_val[:,0], val_pred[:,0])
                mean_err_test = mean_absolute_error(y_test[:,0], test_pred[:,0])

                abs_train_error_1 = np.abs(y_train[:, 0] - train_pred[:, 0])
                abs_val_error_1 = np.abs(y_val[:, 0]- val_pred[:, 0])
                abs_test_error_1 = np.abs(y_test[:, 0]- test_pred[:, 0])

                plt.figure(figsize=(14, 7))
                plt.subplot(3, 1, 1)
                plt.scatter(X[:q_70], abs_train_error_1, label='Absolute Error')
                plt.title('Train data')
                plt.xlabel('time[s]')
                plt.ylabel('absolute error[mm]')
                plt.grid(True)
                plt.legend(loc="upper left")

                plt.subplot(3, 1, 2)
                plt.scatter(X[q_70:q_85], abs_val_error_1, label='Absolute Error')
                plt.title('Validation Data')
                plt.xlabel('time[s]')
                plt.ylabel('absolute error[mm]')
                plt.grid(True)
                plt.legend(loc="upper left")

                plt.subplot(3, 1, 3)
                plt.scatter(X[q_85:-ws - data_prep.target_steps + 1], abs_test_error_1, label='Absolute Error')
                plt.title('Test Data')
                plt.grid(True)
                plt.xlabel('time[s]')
                plt.ylabel('absolute error[mm]')
                plt.legend(loc="upper left")
                plt.tight_layout()
                plt.show()

                print(f'Max train error 1: {np.max(abs_train_error_1)} /n Min train error 1: {np.min(abs_train_error_1)}')
                print(f'Max val error 1: {np.max(abs_val_error_1)} /n Min val error 1: {np.min(abs_val_error_1)}')
                print(f'Max test error 1: {np.max(abs_test_error_1)} /n Min test error 1: {np.min(abs_test_error_1)}')


                print(f"train1 {mean_err_train}\nval1 {mean_err_val}\npred1 {mean_err_test}")

                return mean_absolute_error(y_train, train_pred), mean_absolute_error(y_val, val_pred), mean_absolute_error(y_test,test_pred)


        def build_model(self,X_train,y_train,X_val,y_val):
                tuner = Tuner()
                build_model = tuner.build_model

                
                tuner = kt.Hyperband(
                build_model,
                objective='val_loss',
                max_epochs=130,  
                factor=3,
                directory='my_dir',
                project_name='motion_prediction_weirdos_pls'
                )

                early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

                tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=130, callbacks=[early_stopping],
                        verbose=2)

                best_model = tuner.get_best_models(num_models=1)[0]


                best_model.summary()

                best_model.save('final_tune_weirdos_pls.h5')
                print('Best model saved')
