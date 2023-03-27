import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import LeakyReLU

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

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


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

    # print(data.train_data.shape)

    model.add(Conv2D(32, (3, 3),
                     input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))

    # # label input
    # in_label = Input(shape=(10,))
    # # embedding for categorical input
    # # li = Embedding(n_classes, 50)(in_label)
    # # linear multiplication
    # n_nodes = 7 * 7 * 128
    # li = Dense(n_nodes)(in_label)
    # # reshape to additional channel
    # li = Reshape((7, 7, 128))(li)
    # gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(li)
    # gen = LeakyReLU(alpha=0.2)(gen)
    # # upsample to 28x28
    # gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    # gen = LeakyReLU(alpha=0.2)(gen)
    # # output
    # out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    # # define model
    # model = Model(in_label, out_layer)
    # # opt = Adam(lr=0.001, beta_1=0.5)
    # # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def define_recive():
    model = Sequential()
    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(params[5]))
    # model.add(Activation('relu'))
    model.add(Dense(10))

    return model


class VAE(keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)

        self.gen = tf.keras.Sequential(self.gen)
        self.disc = tf.keras.Sequential(self.disc)

    # encoder传播的过程
    def encoder(self, x):
        real_label = self.disc(x)

        return real_label

    # decoder传播的过程
    def decoder(self, z):
        out = self.gen(z)

        return out

    # def reparameterize(self, mu, log_var):
    #     eps = tf.random.normal(log_var.shape)
    #
    #     std = tf.exp(log_var)  # 去掉log, 得到方差；
    #     std = std ** 0.5  # 开根号，得到标准差；
    #
    #     z = mu + std * eps
    #     return z

    def call(self, inputs, training=None):
        # [b, 784] => [b, z_dim], [b, z_dim]
        z = tf.nn.softmax(self.disc(inputs))
        z = np.around(z, decimals=1)
        # reparameterizaion trick：最核心的部分
        # z = self.reparameterize(mu, log_var)

        # decoder 进行还原
        x_hat = self.decoder(z)

        # Variational auto-encoder除了前向传播不同之外，还有一个额外的约束；
        # 这个约束使得你的mu, var更接近正太分布；所以我们把mu, log_var返回；
        return x_hat


class Semantic_Endec(keras.Model):

    def __init__(self, **kwargs):
        super(Semantic_Endec, self).__init__()
        self.__dict__.update(kwargs)

        self.trans = self.trans
        self.recive = self.recive

    # encoder传播的过程
    def encoder(self, x):
        encoded = self.trans(x)

        return encoded

    # decoder传播的过程
    def decoder(self, z):
        out = self.recive(z)

        return out

    # def reparameterize(self, mu, log_var):
    #     eps = tf.random.normal(log_var.shape)
    #
    #     std = tf.exp(log_var)  # 去掉log, 得到方差；
    #     std = std ** 0.5  # 开根号，得到标准差；
    #
    #     z = mu + std * eps
    #     return z

    def call(self, inputs, training=None):
        z = self.encoder(inputs)

        x_hat = self.decoder(z)

        return x_hat


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


if __name__ == '__main__':
    # 定义超参数
    batchsz = 256  # fashion_mnist
    lr = 1 * 1e-4

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
    # train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
    #
    # test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_db = test_db.batch(batchsz)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # # 搭建模型

    trans = define_trans()
    recive = define_recive()

    model = Semantic_Endec(
        trans=trans,
        recive=recive,
    )

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

    model.trans.save('/home/Attack_sc/models/classification-task/trans', save_format='tf')
    model.recive.save('/home/Attack_sc/models/classification-task/recive', save_format='tf')



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

    # trnum = 12569
    # target = 4
    # NNI = load_model('/home/c_experiment/att_models/NNI_MNIST/%d_NNI_%s' % (trnum,target))
    # NNI.trainable = False
    # C = load_model('/home/models/FASHION30')
    # C.trainable = False
    #
    # z = tf.Variable(tf.random.normal(shape=(1000, 10)), name='var')
    # # z = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    # # z = tf.convert_to_tensor(z)
    # # z = np.expand_dims(z, axis=0)
    # x_gen = NNI(z)
    #
    # target = 8
    # x_input = clip_to_save(x_gen)
    # pre = C.predict(x_input)
    # # showimage(x_input, pre)
    # gen_num = 0
    # for i in range(1000):
    #     label, confidence = np.argmax(pre[i]), pre[i]
    #     print(label)
    #     # print(confidence)
    #
    #     if label == target:
    #         # save_bmp(x_gen[i], target, gen_num)
    #         # x_i = x_input[i]
    #         save_bmp(x_gen[i], target, gen_num)
    #         gen_num += 1
    #         print("a success: %d" % gen_num)
    #         print(confidence)
    #         if gen_num == 50:
    #             break
