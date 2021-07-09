from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import tensorflow_datasets as tfds
# from keras.wrappers.scikit_learn import KerasClassifier
# import tensorflow as tf
from tensorflow_addons.optimizers import Lookahead
# from tensorflow.keras.backend import clear_session
# import keras_tuner as kt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARN)
# from keras.metrics import Precision, Recall
# from optkeras.optkeras import OptKeras
from statistics import mean
import pandas as pd
from sklearn import metrics

class lookHead:
    name = ""
    shape = ""
    classes = ""
    inputs=""
    targets=""
    # optimize_curr= None
    data_array = []
    data = []
    loohead_model=False
    upgrade_model=False
    simple_model=False

    def load_dataset(self,dataset):
        train_dataset = tfds.load(dataset)

        train, test = train_dataset['train'], train_dataset['test']
        train_numpy = np.vstack(tfds.as_numpy(train))
        test_numpy = np.vstack(tfds.as_numpy(test))

        X_train = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
        y_train = np.array(list(map(lambda x: x[0]['label'], train_numpy)))

        X_test = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
        y_test = np.array(list(map(lambda x: x[0]['label'], test_numpy)))

        return (X_train, y_train), (X_test, y_test)

    def getModel(self):
        # Define the model architecture
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    def objective(self,trial):

        model=self.getModel()
        # Compile the model
        if self.loohead_model:
            hp_innerOptimizer = trial.suggest_categorical("optimizer", ["adam", "SGD"])
            hp_slowStepSize = trial.suggest_float("slow_step_size", 0.1, 0.9, log=True)
            model.compile(loss=sparse_categorical_crossentropy,optimizer=Lookahead(optimizer=hp_innerOptimizer, slow_step_size=hp_slowStepSize), metrics=['accuracy'])

        if self.upgrade_model:
            hp_syncPeriod = trial.suggest_int("sync_period", 1, 10)
            hp_slowStepSize = trial.suggest_float("slow_step_size", 0.1, 0.9, log=True)
            model.compile(loss=sparse_categorical_crossentropy,optimizer=Lookahead(optimizer=Lookahead(optimizer='adam'),sync_period=hp_syncPeriod ,slow_step_size=hp_slowStepSize),metrics=['accuracy'])

        if self.simple_model:
            hp_learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            hp_epsilon = trial.suggest_float("epsilon", 1e-7, 1e-3, log=True)
            model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate=hp_learning_rate,epsilon=hp_epsilon),metrics=['accuracy'])


        inter_kfold = KFold(n_splits=3, shuffle=True)
        # K-fold Cross Validation model evaluation
        acc_per_fold = []

        for train_in, test_in in inter_kfold.split(self.inputs, self.targets):

            model.fit(self.inputs[train_in], self.targets[train_in],
                      batch_size=128, epochs=3)

            scores = model.evaluate(self.inputs[test_in], self.targets[test_in], verbose=0)
            acc_per_fold.append(scores[1] * 100)

        return mean(acc_per_fold)


    # def print_scores(self,acc_per_fold, loss_per_fold):
    #     # == Provide average scores ==
    #     print('------------------------------------------------------------------------')
    #     print('Score per fold')
    #     for i in range(0, len(acc_per_fold)):
    #         print('------------------------------------------------------------------------')
    #         print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    #     print('------------------------------------------------------------------------')
    #     print('Average scores for all folds:')
    #     print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    #     print(f'> Loss: {np.mean(loss_per_fold)}')
    #     print('------------------------------------------------------------------------')


    def normalize_data(self,input_train, input_test, target_train, target_test):
        # Parse numbers as floats
        input_train = input_train.astype('float32')
        input_test = input_test.astype('float32')

        # Normalize data
        input_train = input_train / 255
        input_test = input_test / 255

        # Merge inputs and targets
        inputs = np.concatenate((input_train, input_test), axis=0)
        targets = np.concatenate((target_train, target_test), axis=0)

        return inputs, targets


    def model_training(self,dataset_name):
        # Load dataset
        (input_train, target_train), (input_test, target_test) = self.load_dataset(dataset_name)
        data_set_amount=0
        if dataset_name =='cifar10' or dataset_name== 'fashion_mnist':
            newarr_input_train = np.array_split(input_train,10)
            newarr_target_train = np.array_split(target_train,10)
            newarr_input_test = np.array_split(input_test,10)
            newarr_target_test = np.array_split(target_test,10)
            data_set_amount=5

        else:
            newarr_input_train = np.array_split(input_train, 10)
            newarr_target_train = np.array_split(target_train, 10)
            newarr_input_test = np.array_split(input_test, 10)
            newarr_target_test = np.array_split(target_test, 10)
            data_set_amount=10
        for i in range(data_set_amount):
            self.classes=np.unique(newarr_target_test[i]).size
            # Determine shape of the data
            samples, img_width, img_height, img_num_channels = newarr_input_train[i].shape
            self.shape = (img_width, img_height, img_num_channels)
            # Merge inputs and targets
            self.inputs, self.targets = self.normalize_data(newarr_input_train[i], newarr_input_test[i], newarr_target_train[i], newarr_target_test[i])

            # Define per-fold score containers
            # acc_per_fold = []
            # loss_per_fold = []

            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=3, shuffle=True)

            # K-fold Cross Validation model evaluation
            fold_no = 1

            for train, test in kfold.split(self.inputs, self.targets):
                self.data_array=[]
                self.data_array.append(fold_no)
                study = optuna.create_study(direction="maximize")
                study.optimize(self.objective, n_trials=3)



                self.data_array.append(study.best_params)
                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')

                model= self.getModel()
                # Compile the model
                if self.loohead_model:
                    model.compile(loss=sparse_categorical_crossentropy,
                                  optimizer=Lookahead(optimizer=study.best_params['optimizer'], slow_step_size=study.best_params['slow_step_size']),
                                  metrics=['accuracy'])
                if self.upgrade_model:
                    model.compile(loss=sparse_categorical_crossentropy,
                                  optimizer=Lookahead(optimizer=Lookahead(optimizer='adam'), sync_period=study.best_params['sync_period'],slow_step_size=study.best_params['slow_step_size']), metrics=['accuracy'])

                if self.simple_model:
                    model.compile(loss=sparse_categorical_crossentropy,
                                  optimizer=Adam(learning_rate=study.best_params['learning_rate'], epsilon=study.best_params['epsilon']), metrics=['accuracy'])
                # Fit data to model
                model.fit(self.inputs[train], self.targets[train],
                                    batch_size=64,
                                    epochs=10,
                                    verbose=1)

                # Generate generalization metrics
                scores = model.evaluate(self.inputs[test], self.targets[test], verbose=0)

                # print(
                #     f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

                self.data_array.append(scores[1] * 100)
                # loss_per_fold.append(scores[0])


                # fpr, tpr, thresholds = metrics.roc_curve(newarr_input_test[i], newarr_target_test[i], pos_label=2)
                # # average_precision = average_precision_score(input_test, target_test)
                # average_precision = tpr / (fpr + tpr)
                # print("fpr:" + fpr + " tpr:" + tpr + " auc:" + metrics.auc(fpr, tpr))
                # print("precision:" + str(average_precision))

                self.data.append(self.data_array)
                fold_no = fold_no + 1


        # self.print_scores(acc_per_fold, loss_per_fold)


if __name__ == '__main__':
    datasets = ['cifar10','fashion_mnist'] #'emnist', 'fashion_mnist']  #,'cmaterdb' 'kmnist', 'mnist', 'mnist_corrupted','svhn_cropped']
    look =lookHead()
    for x in range(3):
        if x==0:
            look.loohead_model = True
        if x==1:
            look.loohead_model = False
            look.upgrade_model = True
        if x==2:
            look.loohead_model = False
            look.upgrade_model = False
            look.simple_model = True
        for dataset in datasets:
            look.model_training(dataset)

    df = pd.DataFrame(look.data, columns=['Dataset Name', 'Description', 'val_loss', 'loss', 'val_accuracy', 'accuracy', 'lr',
                                     'batch_size', 'batch-norm Y/N'])
    print(df)