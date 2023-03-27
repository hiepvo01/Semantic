import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models, optimizers

# from setup_fashion import FASHION
# from setup_faces import FACES
import os
from setup_mnist import MNIST

IMAGEW = 28
IMAGEH = 28
CHANNELS = 1
learning_rate = 1e-2
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    # tf.keras.metrics.AUC(name='auc')
]


# def trainMNIST(data):


def train(data, file_name, params, num_epochs=20, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(data.train_data.shape)

    model.add(Conv2D(params[0], (3, 3),
                     input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(params[5]))
    # model.add(Activation('relu'))
    model.add(Dense(10))
    # model.add(Activation('softmax'))

    model.summary()

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted / train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=sgd,
                  metrics=['accuracy'])#sparse_categorical_accuracy

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)

    return model




if __name__ == '__main__':
    print(tf.__version__)

    # mnist = tf.keras.datasets.mnist
    #
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # # print(type(x_train))
    '''
    # show top 20 data
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(20):
        img = x_train[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    '''
    # data = FASHION()
    data = MNIST()
    # 模型结构来自文章Towards Evaluating the Robustness of Neural Networks
    model = train(data, "/home/models/MNIST20", [32, 32, 64, 128, 128], num_epochs=20)
    # trainMNIST(MNIST())
    # trainVGG(FACES())

    from tensorflow.keras.models import load_model

    privacymodel = load_model('/home/models/MNIST20')
    privacymodel.trainable = False
    pred = privacymodel(data.validation_data)
    acc = tf.keras.metrics.categorical_accuracy(data.validation_labels, pred)
    acc = tf.reduce_mean(acc)
    print('final acc')
    print(acc)
