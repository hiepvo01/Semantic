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

    def apply_gradients(self, en_gradients, de_gradients):
        self.optimizer.apply_gradients(
            zip(en_gradients, self.encoder.trainable_variables)
        )
        self.optimizer.apply_gradients(
            zip(de_gradients, self.decoder.trainable_variables)
        )

    # @tf.function
    def train(self, train_x):
        samples, labels = train_x[0], train_x[1]
        with tf.GradientTape() as en_tape, tf.GradientTape() as de_tape:
            # encoder_loss, decoder_loss = self.compute_loss(train_x)

            z = T(samples)

            z_channel = self.encoding(z)

            # z_channel = AWGN(z_channel, 0, seed=10)

            x_channel = self.decoding(z_channel)

            x_hat = R(x_channel)

            decoder_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, x_hat))
            # encoder_loss = AWGN(decoder_loss, 0, seed=10)
            encoder_loss = decoder_loss


        en_gradients = en_tape.gradient(encoder_loss, self.encoder.trainable_variables)

        de_gradients = de_tape.gradient(decoder_loss, self.decoder.trainable_variables)

        self.apply_gradients(en_gradients, de_gradients)

        print(encoder_loss)

        acc = tf.keras.metrics.categorical_accuracy(labels, x_hat)
        print('model acc is :')
        print(tf.reduce_mean(acc))







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
    if len(shape)==2:
        noise = tf.random.normal(shape=(shape[0], shape[1])) * np.sqrt(npower)
    else:
        noise = tf.random.normal(shape=(shape[0],)) * np.sqrt(npower)

    return x + noise

if __name__ == '__main__':
    # 定义超参数
    batchsz = 512  # mnist
    lr = 0.001
    TRAIN_BUF = 60000
    BATCH_SIZE = 512
    TEST_BUF = 10000
    DIMS = (28, 28, 1)
    N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)

    # # 数据集加载
    labels_num = ["0", "1", "2", "3", "4",
                  "5", "6", "7", "8", "9"]
    # classl = ["T-Shirt", "Trouser", "Pullover", "Dress",
    #           "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    # del labels_num[target]
    (x_train, y_train), (x_test, y_test) = get_data(classes=labels_num)

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
    T = load_model('/home/Attack_sc/models/classification-task/trans')
    T.trainable = False
    R = load_model('/home/Attack_sc/models/classification-task/recive')
    R.trainable = False
    evadata = x_test[:1000]
    z = T(evadata)
    X_hat = R(z)
    acc = tf.keras.metrics.categorical_accuracy(y_test[:1000], X_hat)
    print('model acc is :')
    print(tf.reduce_mean(acc))


    encoder, decoder = define_endec()

    # optimizer = keras.optimizers.SGD(lr=lr)
    optimizer = keras.optimizers.Adam(lr=lr)

    model = Channel_Endec(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
    )


    ### Train the model
    for epoch in range(20):
        for step, x in enumerate(train_db):
            model.train(x)





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


    model.trans.save('/home/Attack_sc/models/classification-task/channel_encoder', save_format='tf')
    model.recive.save('/home/Attack_sc/models/classification-task/channel_decoder', save_format='tf')



    E = load_model('/home/Attack_sc/models/classification-task/channel_encoder')
    E.trainable = False
    D = load_model('/home/Attack_sc/models/classification-task/channel_decoder')
    D.trainable = False


    evadata = x_test[:1000]
    z_channel = E(T(evadata))
    z_channel = AWGN(z_channel, 0, seed=10)
    X_hat = R(D(z_channel))
    acc = tf.keras.metrics.categorical_accuracy(y_test[:1000], X_hat)
    print('model acc is :')
    print(tf.reduce_mean(acc))

