import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import LeakyReLU
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from numpy import expand_dims
from matplotlib import pyplot
from numpy.random import randn
from numpy.random import randint
from numpy import asarray
# tf.random.set_seed(22)
# np.random.seed(22)
# from keras.preprocessing import image
import cv2
from setup_mnist import MNIST
from tqdm.autonotebook import tqdm
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
print(np.__version__)
print(tf.__version__)


# 把多张image保存达到一张image里面去。
def save_images(img, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = img[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


def define_trans():
    model = Sequential()
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(2048))
    model.add(Activation('relu'))

    return model


def define_recive():
    model = Sequential()

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(params[5]))
    # model.add(Activation('relu'))
    # model.add(Dense(10))

    return model


def define_endec():
    encoder = [
        tf.keras.layers.Dense(units=4096, activation="relu"),
        tf.keras.layers.Dense(units=2048, activation="relu"),
    ]

    decoder = [
        tf.keras.layers.Dense(units=2048, activation="relu"),
        tf.keras.layers.Dense(units=4096, activation="relu"),
    ]

    return encoder, decoder


class Channel_Endec(keras.Model):

    def __init__(self, **kwargs):
        super(Channel_Endec, self).__init__()
        self.__dict__.update(kwargs)

        self.encoder = tf.keras.Sequential(self.encoder)
        self.decoder = tf.keras.Sequential(self.decoder)

    # encoder传播的过程
    def encoding(self, x):
        encoded = self.encoder(x)

        return encoded

    # decoder传播的过程
    def decoding(self, z):
        out = self.decoder(z)

        return out

    def call(self, inputs, training=None):
        z = T(inputs)

        z_channel = self.encoding(z)

        # z_channel = AWGN(z_channel, 0, seed=10)

        x_channel = self.decoding(z_channel)

        x_hat = R(x_channel)

        return x_hat

    def apply_gradients(self, en_gradients, de_gradients):
        self.optimizer.apply_gradients(
            zip(en_gradients, self.encoder.trainable_variables)
        )
        self.optimizer.apply_gradients(
            zip(de_gradients, self.decoder.trainable_variables)
        )

    def training_log(self, train_x, test, train_loss, x_hat, step, epoch):

        train_acc = tf.keras.metrics.categorical_accuracy(train_x[1], x_hat)
        print('model train acc is :')
        print(tf.reduce_mean(train_acc))

        z = T(test[0])

        z_channel = self.encoding(z)

        # z_channel = AWGN(z_channel, 15, seed=10)
        # z_channel = awgn(z_channel, 15)

        x_channel = self.decoding(z_channel)

        y_hat = R(x_channel)

        test_acc = tf.keras.metrics.categorical_accuracy(test[1], y_hat)
        print('model test acc is :')
        print(tf.reduce_mean(test_acc))

        test_loss = loss_obj(test[1], y_hat)

        return [epoch, step + (1+N_TRAIN_BATCHES)*epoch +1, train_loss, tf.reduce_mean(train_acc), test_loss, tf.reduce_mean(test_acc)]





    # @tf.function
    def train(self, train_x, test_db, step, epoch):
        samples, labels = train_x[0], train_x[1]
        with tf.GradientTape() as en_tape, tf.GradientTape() as de_tape:
            z = T(samples)

            z_channel = self.encoding(z)

            # z_channel = AWGN(z_channel, 15, seed=10)
            z_channel = awgn(z_channel, 15)

            x_channel = self.decoding(z_channel)

            x_hat = R(x_channel)

            decoder_loss = loss_obj(labels, x_hat)
            # encoder_loss = AWGN(decoder_loss, 15, seed=10)
            encoder_loss = decoder_loss

        en_gradients = en_tape.gradient(encoder_loss, self.encoder.trainable_variables)

        de_gradients = de_tape.gradient(decoder_loss, self.decoder.trainable_variables)

        self.apply_gradients(en_gradients, de_gradients)

        return self.training_log(train_x,test_db,encoder_loss,x_hat, step, epoch)



    def attacking(self, message):
        index = tf.argmax(message, axis=1)

        malicious = np.ones_like(message) * -1

        # for i in range(len(malicious)):
        #     malicious[i][index[i]] *= -1
        M = message * malicious

        return M

    def attack_forward_train(self, train_x, test_db, step, epoch):
        samples, labels = train_x[0], train_x[1]
        with tf.GradientTape() as en_tape:
            z = T(samples)

            z_channel = self.encoding(z)

            # z_channel = AWGN(z_channel, 15, seed=10)
            # z_channel = awgn(z_channel, 15)
            z_channel_attacked = self.attacking(z_channel)

            x_channel = self.decoding(z_channel_attacked)

            x_hat = R(x_channel)

            decoder_loss = loss_obj(labels, x_hat)
            # encoder_loss = AWGN(decoder_loss, 15, seed=10)
            encoder_loss = decoder_loss

        en_gradients = en_tape.gradient(encoder_loss, model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(en_gradients, model.trainable_variables)
        )

        return self.training_log(train_x, test_db, encoder_loss, x_hat, step, epoch)


    def attack_backward_train(self, train_x, test_db, step, epoch):
        samples, labels = train_x[0], train_x[1]
        with tf.GradientTape() as en_tape:
            z = T(samples)

            z_channel = self.encoding(z)

            # z_channel = AWGN(z_channel, 15, seed=10)
            # z_channel = awgn(z_channel, 15)
            # z_channel_attacked = self.attacking(z_channel)

            x_channel = self.decoding(z_channel)

            x_hat = R(x_channel)

            decoder_loss = loss_obj(labels, x_hat)
            # encoder_loss = AWGN(decoder_loss, 15, seed=10)
            encoder_loss = decoder_loss
        en_gradients = en_tape.gradient(encoder_loss, model.trainable_variables)

        malicious = np.ones_like(en_gradients[0:2]) * -1

        en_gradients[0:2] *= malicious

        self.optimizer.apply_gradients(
            zip(en_gradients, model.trainable_variables)
        )

        return self.training_log(train_x, test_db, encoder_loss, x_hat, step, epoch)



def get_data_all():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    num_classes = len(np.unique(y_train))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # fig = plt.figure(figsize=(8, 3))
    # for i in range(num_classes):
    #     ax = plt.subplot(2, 5, 1 + i, xticks=[], yticks=[])
    #     idx = np.where(y_train[:] == i)[0]
    #     features_idx = X_train[idx, ::]
    #     img_num = np.random.randint(features_idx.shape[0])
    #     img = features_idx[img_num, ::]
    #     ax.set_title(class_names[i])
    #     plt.imshow(img)
    #
    # plt.tight_layout()

    # %% md

    #### Reshaping and normalizing the inputs

    # %%

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # %%

    # X = np.float32(X_train)
    # X = (X / 255 - 0.5) * 2
    # X = np.clip(X, -1, 1)
    #
    # X_t = np.float32(X_test)
    # X_t = (X_t / 255 - 0.5) * 2
    # X_t = np.clip(X_t, -1, 1)

    # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, num_classes)
    # Y_test = np_utils.to_categorical(y_test, num_classes)
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # the generator is using tanh activation, for which we need to preprocess
    # the image data into the range between -1 and 1.

    # X_train = np.float32(X_train)
    # X_train = (X_train / 255 - 0.5) * 2
    # X_train = np.clip(X_train, -1, 1)
    #
    # X_test = np.float32(X_test)
    # X_test = (X_test / 255 - 0.5) * 2
    # X_test = np.clip(X_test, -1, 1)

    print('X_train reshape:', X_train.shape)
    print('X_test reshape:', X_test.shape)

    # %%

    print(X_train[0].shape)
    return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))


def get_data(classes):
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    X = expand_dims(train_images, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5

    X_test = expand_dims(test_images, axis=-1)
    # convert from ints to floats
    X_test = X_test.astype('float32')
    # scale from [0,255] to [-1,1]
    X_test = (X_test - 127.5) / 127.5

    # sorting based on index
    idx = np.argsort(train_labels)
    train_images = X[idx]
    train_labels = train_labels[idx]

    idx = np.argsort(test_labels)
    test_images = X_test[idx]
    test_labels = test_labels[idx]

    labels = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    labels_num = ["0", "1", "2", "3", "4",
                  "5", "6", "7", "8", "9"]

    label_mapping = dict(zip(labels_num, range(10)))
    X_train, X_test, y_train, y_test = [], [], [], []

    for cls in classes:
        idx = label_mapping[cls]
        start = idx * 6000
        end = idx * 6000 + 6000
        X_train.extend(train_images[start: end])
        y_train.extend(train_labels[start: end])
        start = idx * 1000
        end = idx * 1000 + 1000
        X_test.extend(test_images[start: end])
        y_test.extend(test_labels[start: end])

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))


def train():
    for epoch in range(20):

        for step, x in enumerate(train_db):
            images, labels = x[0], x[1]

            # x = tf.reshape(x, [-1, 784])
            with tf.GradientTape() as tape:
                # shape
                x_hat = model(images)
                # 把每个像素点当成一个二分类的问题；
                # loss = tf.losses.mean_squared_error(x, x_hat)
                # rec_loss = tf.losses.MSE(x, x_rec_logits)
                # loss = tf.reduce_mean(loss)

                loss = tf.keras.losses.categorical_crossentropy(labels, x_hat)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print('epoch: %3d, step:%4d, rec_loss:%9f' % (epoch, step, float(tf.reduce_mean(loss))))

            # if step % 100 == 0:
            #     print('epoch: %3d, step:%4d, rec_loss:%9f' % (epoch, step, float(tf.reduce_mean(loss))))

            # viltrainedmodel(epoch, 'adv')
            # #z = np.array([0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
            # #z = tf.convert_to_tensor(z)
            # #z = np.expand_dims(z, axis=0)
            # #x_gen = model.gen(z)
            # #save_bmp(x_gen, 'test', epoch)
    # model.gen.save('/home/tianzhiyi/ReverseLearning/attack_models/NNInum_%s' % target, save_format='tf')


def display_images(dis_image, description):
    pre = C(dis_image)
    label, confidence = np.argmax(pre), pre
    pyplot.figure()
    pyplot.imshow(dis_image[0])
    pyplot.title(description + '\n label is:' + str(label) + '\n' + str(confidence))
    pyplot.colorbar()
    pyplot.show()


def save_bmp(img, name, epoch):
    # 保存中间文件
    img_save = img.numpy()
    ymax = 255
    ymin = 0
    xmax = np.amax(img_save)
    xmin = np.amin(img_save)
    img_save = np.round((ymax - ymin) * (img_save - xmin) / (xmax - xmin) + ymin)
    # cv2.imshow('image', img_save[0])
    cv2.imwrite('/home/c_experiment/c_NNI/MNIST/c_NNI_%d/%s_%s.bmp' % (trnum, name, epoch), img_save)

    # img = image.array_to_img(img[0]*255., scale=False)
    # img.save('/home/tianzhiyi/ReverseLearning/testAE_GAN/NNI/%s_%s.bmp' % (name, epoch))


def data_process(img_path):
    # 图像预处理
    def preprocess(image):
        # print(image)
        image = tf.cast(image, tf.float32)
        image = image / 255
        image = tf.image.resize(image, (28, 28))
        image = image[None, ...]
        # print(image)
        return image

    image_raw = tf.io.read_file(img_path)
    image = tf.image.decode_image(image_raw)

    image = preprocess(image)

    return image


def clip_to_save(image):
    # gen_input = np.clip(image, -1, 1)
    # gen_input = (gen_input + 1) * 127
    # gen_input = np.round(gen_input).astype('uint8')

    img_save = image.numpy()
    ymax = 255
    ymin = 0
    xmax = np.amax(img_save)
    xmin = np.amin(img_save)
    img_save = np.round((ymax - ymin) * (img_save - xmin) / (xmax - xmin) + ymin)
    img_save = np.round(img_save).astype('uint8')
    return img_save


def AWGN(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    shape = np.array(x.shape)
    snr = 10 ** (snr / 10.0)
    xpower = tf.reduce_sum(x ** 2) / shape[0]
    npower = xpower / snr
    if len(shape) == 2:
        noise = tf.random.normal(shape=(shape[0], shape[1])) * np.sqrt(npower)
    else:
        noise = tf.random.normal(shape=(shape[0],)) * np.sqrt(npower)

    return x + noise


from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal


def awgn(s, SNRdB, L=1):
    """
    https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
"""
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if isrealobj(s):  # check if input is real/complex object type
        n = sqrt(N0 / 2) * standard_normal(s.shape)  # computed noise
    else:
        n = sqrt(N0 / 2) * (standard_normal(s.shape) + 1j * standard_normal(s.shape))
    r = s + n  # received signal
    return r


if __name__ == '__main__':
    # 定义超参数
    batchsz = 1024  # mnist
    lr = 0.004
    TRAIN_BUF = 50000
    BATCH_SIZE = 1024
    TEST_BUF = 10000
    DIMS = (32, 32, 3)
    N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)

    # # 数据集加载
    labels_num = ["0", "1", "2", "3", "4",
                  "5", "6", "7", "8", "9"]
    # classl = ["T-Shirt", "Trouser", "Pullover", "Dress",
    #           "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    # del labels_num[target]
    (x_train, y_train), (x_test, y_test) = get_data_all()

    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

    # we do not need label auto-encoder大家可以理解为无监督学习,标签其实就是本身，和自己对比；
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(TRAIN_BUF).batch(batchsz)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(TEST_BUF).batch(batchsz)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # # 搭建模型

    # encoder = define_trans()
    # decoder = define_recive()
    T = load_model('/home/Attack_sc/models/CIFAR-classification-task/trans')
    T.trainable = False
    R = load_model('/home/Attack_sc/models/CIFAR-classification-task/recive')
    R.trainable = False
    evadata = x_test[:3000]
    z = T(evadata)
    X_hat = R(z)
    acc = tf.keras.metrics.categorical_accuracy(y_test[:3000], X_hat)
    print('model acc is :')
    print(tf.reduce_mean(acc))

    encoder, decoder = define_endec()

    optimizer = keras.optimizers.SGD(lr=lr)
    # optimizer = keras.optimizers.Adam(lr=lr)
    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = Channel_Endec(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
    )

    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #           batch_size=batchsz,
    #           validation_data=(x_test, y_test),
    #           epochs=50,
    #           shuffle=True)

    # test = (x_test, y_test)
    train_log = []

    test = list(enumerate(test_db))
    ### Train the model
    for epoch in range(50):
        for step, x in enumerate(train_db):
            test_data = test[(step+1) % (N_TEST_BATCHES+1)]
            # log = model.train(x, test_data[1], step, epoch)
            # log = model.attack_forward_train(x, test_data[1], step, epoch)
            log = model.attack_backward_train(x, test_data[1], step, epoch)
            train_log.append(log)
            save = pd.DataFrame(data=train_log)
            save.to_csv('/home/Attack_sc/models/log-backward.csv')


    '''
    # a pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns=['encoder_loss', 'decoder_loss'])

    n_epochs = 50
    for epoch in range(n_epochs):
        # train
        for batch, train_x in tqdm(
                zip(range(N_TRAIN_BATCHES), train_db), total=N_TRAIN_BATCHES
        ):
            model.train(train_x)
        # test on holdout
        loss = []
        for batch, test_x in tqdm(
                zip(range(N_TEST_BATCHES), test_db), total=N_TEST_BATCHES
        ):
            loss.append(tf.reduce_mean(model.compute_loss(test_x)))
        losses.loc[len(losses)] = np.mean(loss, axis=0)
        # plot results
        # display.clear_output()
        print(
            "Epoch: {} | encoder_loss: {} | decoder_loss: {}".format(
                epoch, losses.encoder_loss.values[-1], losses.decoder_loss.values[-1]
            )
        )

        evadata = x_test[:1000]
        z_channel = model.encoder(T(evadata))
        z_channel = AWGN(z_channel, 0, seed=10)
        X_hat = R(model.decoder(z_channel))
        acc = tf.keras.metrics.categorical_accuracy(y_test[:1000], X_hat)
        print('model acc is :')
        print(tf.reduce_mean(acc))
    '''

    '''
    optimizer = keras.optimizers.SGD(lr=lr)

    # train()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batchsz,
              validation_data=(x_test, y_test),
              epochs=50,
              shuffle=True)
    '''

    model.encoder.save('/home/Attack_sc/models/CIFAR-classification-task-backward/channel_encoder', save_format='tf')
    model.decoder.save('/home/Attack_sc/models/CIFAR-classification-task-backward/channel_decoder', save_format='tf')

    E = load_model('/home/Attack_sc/models/CIFAR-classification-task-backward/channel_encoder')
    E.trainable = False
    D = load_model('/home/Attack_sc/models/CIFAR-classification-task-backward/channel_decoder')
    D.trainable = False

    evadata = x_test[:1000]
    z_channel = E(T(evadata))
    # z_channel = AWGN(z_channel, 0, seed=10)
    X_hat = R(D(z_channel))
    acc = tf.keras.metrics.categorical_accuracy(y_test[:1000], X_hat)
    print('model acc is :')
    print(tf.reduce_mean(acc))

