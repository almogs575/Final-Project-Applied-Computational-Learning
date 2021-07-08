from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import tensorflow_datasets as tfds

from tensorflow_addons.optimizers import Lookahead

from keras.metrics import Precision, Recall


def load_cifar(dataset):
    train_dataset = tfds.load(dataset)

    train, test = train_dataset['train'], train_dataset['test']
    train_numpy = np.vstack(tfds.as_numpy(test))
    test_numpy = np.vstack(tfds.as_numpy(test))

    X_train = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
    y_train = np.array(list(map(lambda x: x[0]['label'], train_numpy)))

    X_test = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
    y_test = np.array(list(map(lambda x: x[0]['label'], test_numpy)))

    return (X_train, y_train), (X_test, y_test)


def create_model(input_shape, optimizer, loss_function, no_classes):
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model


def print_scores(acc_per_fold, loss_per_fold):
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


def normalize_data(input_train, input_test, target_train, target_test):
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


def model_training(dataset_name, optimizer):
    # Model configuration
    batch_size = 64
    img_width, img_height, img_num_channels = 32, 32, 3
    loss_function = sparse_categorical_crossentropy
    no_classes = 10
    no_epochs = 10
    verbosity = 1
    num_folds = 10

    # Load dataset
    (input_train, target_train), (input_test, target_test) = load_cifar(dataset_name)

    # Merge inputs and targets
    inputs, targets = normalize_data(input_train, input_test, target_train, target_test)

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)

    for train, test in kfold.split(inputs, targets):
        model = create_model(input_shape, optimizer, loss_function, no_classes)

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=batch_size,
                            epochs=no_epochs,
                            verbose=verbosity)

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    print_scores(acc_per_fold, loss_per_fold)


if __name__ == '__main__':
    datasets = ['cats_vs_dogs', 'rock_paper_scissors', 'plant_leaves', 'beans', 'cars196', 'cifar10', 'citrus_leaves',
                'food101', 'horses_or_humans', 'malaria', 'pet_finder', 'plant_village', 'stanford_dogs', 'tf_flowers',
                'imagenet2012', 'deep_weeds', 'caltech101', 'cassava', 'eurosat', 'lfw']
    optimizer = Adam()
    opt = Lookahead(optimizer)
    opt2 = Lookahead(opt)
    for dataset in datasets:
        model_training(dataset, optimizer)
        model_training(dataset, opt)
        model_training(dataset, opt2)
