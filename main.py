import time
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import Lookahead
import optuna
import scikit_posthocs as sp
optuna.logging.set_verbosity(optuna.logging.WARN)
from statistics import mean
from scipy import stats
from tensorflow.keras.utils import to_categorical


class lookAhead:
    name = ""
    shape = ""
    classes = 0
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    data_array = []
    data = []
    lookAhead_model = False
    improved_model = False
    simple_model = False
    auc_lookAhead = []
    auc_improved_lookAhead = []
    auc_adam = []

    def getModel(self):
        # Define the model architecture
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    def objective(self, trial):

        model = self.getModel()
        # Compile the model
        if self.lookAhead_model:
            hp_innerOptimizer = trial.suggest_categorical("optimizer", ["adam", "SGD"])
            hp_slowStepSize = trial.suggest_float("slow_step_size", 0.1, 0.9, log=True)
            model.compile(loss=categorical_crossentropy,
                          optimizer=Lookahead(optimizer=hp_innerOptimizer, slow_step_size=hp_slowStepSize),
                          metrics=['accuracy'])

        if self.improved_model:
            hp_syncPeriod = trial.suggest_int("sync_period", 1, 10)
            hp_slowStepSize = trial.suggest_float("slow_step_size", 0.1, 0.9, log=True)
            model.compile(loss=categorical_crossentropy,
                          optimizer=Lookahead(optimizer=Lookahead(optimizer='adam'), sync_period=hp_syncPeriod,
                                              slow_step_size=hp_slowStepSize), metrics=['accuracy'])

        if self.simple_model:
            hp_learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            hp_epsilon = trial.suggest_float("epsilon", 1e-7, 1e-3, log=True)
            model.compile(loss=categorical_crossentropy,
                          optimizer=Adam(learning_rate=hp_learning_rate, epsilon=hp_epsilon), metrics=['accuracy'])

        inter_kfold = KFold(n_splits=3, shuffle=True)

        # K-fold Cross Validation model evaluation
        acc_per_fold = []

        for train_in, test_in in inter_kfold.split(self.X_train, self.y_train):
            model.fit(self.X_train[train_in], to_categorical(self.y_train[train_in]),
                      validation_data=(self.X_train[test_in], to_categorical(self.y_train[test_in])), batch_size=128,
                      epochs=3)

            scores = model.evaluate(self.X_test, to_categorical(self.y_test), verbose=0)
            acc_per_fold.append(scores[1] * 100)

        return mean(acc_per_fold)

    def normalize_data(self, X_train, X_test):
        # Parse numbers as floats
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # Normalize data
        X_train = X_train / 255
        X_test = X_test / 255

        return X_train, X_test

    def model_training(self, dataset_name, algorithm_name):

        # Load dataset
        if dataset_name == 'cifar10':
            (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
            self.shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

        if dataset_name == 'fashion_mnist':
            (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            self.shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

        # normalize
        X_train, X_test = self.normalize_data(X_train, X_test)

        # num of classes
        self.classes = np.unique(y_test).size

        # splitting data to smaller datasets
        X_train_split = np.array_split(X_train, 10)
        y_train_split = np.array_split(y_train, 10)
        X_test_split = np.array_split(X_test, 10)
        y_test_split = np.array_split(y_test, 10)
        datasets_amount = 10

        for i in range(datasets_amount):

            self.X_train = X_train_split[i]
            self.y_train = y_train_split[i]
            self.X_test = X_test_split[i]
            self.y_test = y_test_split[i]

            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=10, shuffle=True)

            # K-fold Cross Validation model evaluation
            fold_no = 1

            for train, test in kfold.split(self.X_train, self.y_train):

                self.data_array = []
                self.data_array.append(dataset_name + "_part_" + str(i))
                self.data_array.append(algorithm_name)
                self.data_array.append(fold_no)
                study = optuna.create_study(direction="maximize")
                study.optimize(self.objective, n_trials=50)
                self.data_array.append(study.best_params)

                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')

                model = self.getModel()
                # Compile the model
                if self.lookAhead_model:
                    model.compile(loss=categorical_crossentropy,
                                  optimizer=Lookahead(optimizer=study.best_params['optimizer'],
                                                      slow_step_size=study.best_params['slow_step_size']),
                                  metrics=['accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.5),
                                           tf.keras.metrics.SpecificityAtSensitivity(0.5), tf.keras.metrics.Precision(),
                                           tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR')])
                if self.improved_model:
                    model.compile(loss=categorical_crossentropy,
                                  optimizer=Lookahead(optimizer=Lookahead(optimizer='adam'),
                                                      sync_period=study.best_params['sync_period'],
                                                      slow_step_size=study.best_params['slow_step_size']),
                                  metrics=['accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.5),
                                           tf.keras.metrics.SpecificityAtSensitivity(0.5), tf.keras.metrics.Precision(),
                                           tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR')])

                if self.simple_model:
                    model.compile(loss=categorical_crossentropy,
                                  optimizer=Adam(learning_rate=study.best_params['lr'],
                                                 epsilon=study.best_params['epsilon']),
                                  metrics=['accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.5),
                                           tf.keras.metrics.SpecificityAtSensitivity(0.5), tf.keras.metrics.Precision(),
                                           tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR')])

                fit_start = time.time()
                # Fit data to model
                model.fit(self.X_train[train], to_categorical(self.y_train[train]),
                          validation_data=(self.X_train[test], to_categorical(self.y_train[test])), batch_size=64,
                          epochs=10, verbose=1)
                stop_fit = time.time() - fit_start

                # Generate generalization metrics
                scores = model.evaluate(self.X_test, to_categorical(self.y_test), verbose=0)

                inference_start = time.time()
                model.predict(self.X_test)
                inference_stop = time.time() - inference_start

                if self.lookAhead_model:
                    self.auc_lookAhead.append(scores[5])

                if self.improved_model:
                    self.auc_improved_lookAhead.append(scores[5])

                if self.simple_model:
                    self.auc_adam.append(scores[5])

                self.data_array.append(scores[1] * 100)
                self.data_array.append(scores[2])
                self.data_array.append(scores[3])
                self.data_array.append(scores[4])
                self.data_array.append(scores[5])
                self.data_array.append(scores[6])
                self.data_array.append(stop_fit)
                self.data_array.append(inference_stop)

                self.data.append(self.data_array)
                fold_no = fold_no + 1

    def friedman_test(self):
        results = stats.friedmanchisquare(self.auc_adam, self.auc_lookAhead, self.auc_improved_lookAhead)
        print(results)
        if results[1] < 0.05:
            return False
        return True

    def hoc_test(self):
        data = np.array([self.auc_lookAhead, self.auc_adam, self.auc_improved_lookAhead])
        hoc = sp.posthoc_nemenyi_friedman(data)
        print(hoc)


if __name__ == '__main__':
    datasets = ['cifar10','fashion_mnist']  # 'emnist', 'fashion_mnist']  #,'cmaterdb' 'kmnist', 'mnist', 'mnist_corrupted','svhn_cropped']
    look = lookAhead()
    for x in range(3):
        if x == 0:
            look.lookAhead_model = True
            algorithm_name = 'lookAhead'
        if x == 1:
            look.lookAhead_model = False
            look.improved_model = True
            algorithm_name = 'lookAhead_improved'
        if x == 2:
            look.lookAhead_model = False
            look.improved_model = False
            look.simple_model = True
            algorithm_name = 'baseline_model'
        for dataset in datasets:
            look.model_training('fashion_mnist', algorithm_name)

    df = pd.DataFrame(look.data,
                      columns=['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values',
                               'Accuracy', 'TPR', 'FPR',
                               'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time'])

    df.to_csv('results.csv', index=False)
    friedman = look.friedman_test()
    if friedman:
        look.hoc_test()
