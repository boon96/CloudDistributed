from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import concatenate
from numpy import argmax
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 20
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = './models/imageclassifier_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

# define stacked model from multiple member input models


def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(64, activation='relu')(merge)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    #plot_model(model, show_shapes=True, to_file='model_graph.png')
    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


def fit_stacked_model(model, inputX, inputy, val):

    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    V = [list(val)[0][0] for _ in range(len(model.input))]
    history = model.fit(X, inputy, validation_data=(
        V, list(val)[0][1]), epochs=epochs, callbacks=[tensorboard_callback])
    return history

# make prediction with stacked model


def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


data = tf.keras.utils.image_dataset_from_directory('Images')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
data = data.map(lambda x, y: (x/255, y))
train_size = int(len(data)*.6)
val_size = int(len(data)*.2)
test_size = int(len(data)*.2)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
trainX = list(train)[0][0]
trainy = list(train)[0][1]
test = data.skip(train_size+val_size).take(test_size)
testX = list(test)[0][0]
testy = list(test)[0][1]

n_members = 2
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
for model in members:
    _, acc = model.evaluate(test)
    print('Model Accuracy: %.3f' % acc)

stacked_model = define_stacked_model(members)
history = fit_stacked_model(stacked_model, trainX, trainy, val)

yhat = predict_stacked_model(stacked_model, testX)
yhat = argmax(yhat, axis=1)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# plot_history(history)
