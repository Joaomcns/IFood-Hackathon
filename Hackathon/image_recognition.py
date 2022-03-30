import os
import pickle
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from keras.applications.vgg19 import VGG19

physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


def read_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    base_model = VGG19(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(5, activation='softmax')(x)
    model = Model(base_model.input, x)

    X_train = read_data('X_train.pickle')
    y_train = read_data('y_train.pickle')
    X_test = read_data('X_test.pickle')
    y_test = read_data('y_test.pickle')

    model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=False,
        name='SGD'
    ), loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(x=X_train,
                     y=y_train,
                     epochs=100,
                     verbose=1,
                     shuffle=True,
                     validation_data=(X_test, y_test),
                     use_multiprocessing=False
                     )

    model.save('initial_cnn.h5', save_format='h5')

    # plot training and validation loss
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot training and validation accuracy
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
