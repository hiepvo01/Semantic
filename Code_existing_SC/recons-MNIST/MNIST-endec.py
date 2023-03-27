import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from IPython import display
import pandas as pd
# import cv2
from skimage.metrics import structural_similarity
def plot_reconstruction(model, example_data, nex=5, zm=3):
    example_data_reconstructed = model.decode(model.encode(example_data))
    fig, axs = plt.subplots(ncols=nex, nrows=2, figsize=(zm * nex, zm * 2))
    for exi in range(nex):
        axs[0, exi].matshow(
            example_data.numpy()[exi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
        )
        axs[1, exi].matshow(
            example_data_reconstructed.numpy()[exi].squeeze(),
            cmap=plt.cm.Greys,
            vmin=0,
            vmax=1,
        )

    for ax in axs.flatten():
        ax.axis("off")
    plt.show()
    temp = example_data[0]
    print('PSNR is :')
    print(tf.image.psnr(example_data[0:100],example_data_reconstructed[0:100], max_val=1))
    # print('SSIM is :')
    # print(tf.image.ssim(example_data[0:5],example_data_reconstructed[0:5], max_val=1))


class AE(tf.keras.Model):
    """a basic autoencoder class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    @tf.function
    def encode(self, x):
        return self.enc(x)

    @tf.function
    def decode(self, z):
        return self.dec(z)

    @tf.function
    def compute_loss(self, x):
        z = self.encode(x)
        # z = awgn(z,15)

        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        return ae_loss

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    # def train(self, train_x):
    #     gradients = self.compute_gradients(train_x)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    def apply_gradients(self, en_gradients, de_gradients):
        self.optimizer.apply_gradients(
            zip(en_gradients, self.enc.trainable_variables)
        )
        self.optimizer.apply_gradients(
            zip(de_gradients, self.dec.trainable_variables)
        )

    # @tf.function
    def train(self, train_x, test_db, step, epoch):
        with tf.GradientTape() as en_tape, tf.GradientTape() as de_tape:
            z = self.encode(train_x)
            z = awgn(z, 15)

            _x = self.decode(z)
            ae_loss = tf.reduce_mean(tf.square(train_x - _x))

        en_gradients = en_tape.gradient(ae_loss, self.enc.trainable_variables)

        de_gradients = de_tape.gradient(ae_loss, self.dec.trainable_variables)

        self.apply_gradients(en_gradients, de_gradients)
        # print(ae_loss)
        # return self.training_log(train_x,test_db,encoder_loss,x_hat, step, epoch)

    def attacking(self, message):
        malicious = np.ones_like(message) * -1

        # for i in range(len(malicious)):
        #     malicious[i][index[i]] *= -1
        M = message * malicious

        return M


    def attack_forward_train(self, train_x, test_db, step, epoch):

        with tf.GradientTape() as en_tape, tf.GradientTape() as de_tape:
            z = self.encode(train_x)
            z = self.attacking(z)

            _x = self.decode(z)
            ae_loss = tf.reduce_mean(tf.square(train_x - _x))

        en_gradients = en_tape.gradient(ae_loss, self.enc.trainable_variables)

        de_gradients = de_tape.gradient(ae_loss, self.dec.trainable_variables)

        self.apply_gradients(en_gradients, de_gradients)

    def attack_backward_train(self, train_x, test_db, step, epoch):



        with tf.GradientTape() as en_tape, tf.GradientTape() as de_tape:
            z = self.encode(train_x)


            _x = self.decode(z)
            ae_loss = tf.reduce_mean(tf.square(train_x - _x))

        en_gradients = en_tape.gradient(ae_loss, self.enc.trainable_variables)

        de_gradients = de_tape.gradient(ae_loss, self.dec.trainable_variables)

        en_gradients = self.attacking(en_gradients)

        self.apply_gradients(en_gradients, de_gradients)


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

    TRAIN_BUF = 60000
    BATCH_SIZE = 1024
    TEST_BUF = 10000
    DIMS = (28, 28, 1)
    N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)

    # load dataset
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    # split dataset
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    ) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

    # batch datasets
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
            .shuffle(TRAIN_BUF)
            .batch(BATCH_SIZE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
            .shuffle(TEST_BUF)
            .batch(BATCH_SIZE)
    )
    N_Z = 64
    encoder = [
        tf.keras.layers.InputLayer(input_shape=DIMS),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N_Z),
    ]

    decoder = [
        tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ),
    ]
    # the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(1e-3)
    # optimizer = tf.keras.optimizers.SGD(lr=0.004)
    # train the model
    model = AE(
        enc=encoder,
        dec=decoder,
        optimizer=optimizer,
    )



    # n_epochs = 50
    # for epoch in range(n_epochs):
    #     # train
    #     for batch, train_x in tqdm(
    #             zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    #     ):
    #         model.train(train_x)
    #     # test on holdout
    #     loss = []
    #     for batch, test_x in tqdm(
    #             zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    #     ):
    #         loss.append(model.compute_loss(train_x))
    #     losses.loc[len(losses)] = np.mean(loss, axis=0)
    #     # plot results
    #     display.clear_output()
    #     print("Epoch: {} | MSE: {}".format(epoch, losses.MSE.values[-1]))
    #     plot_reconstruction(model, example_data)
    #
    # plt.plot(losses.MSE.values)

    train_log = []
    example_data = next(iter(train_dataset))
    losses = pd.DataFrame(columns=['MSE'])

    test = list(enumerate(test_dataset))
    ### Train the model
    for epoch in range(30):
        for step, x in enumerate(train_dataset):
            test_data = test[(step+1) % (N_TEST_BATCHES+1)]
            # log = model.train(x, test_data[1], step, epoch)
            # log = model.attack_forward_train(x, test_data[1], step, epoch)
            log = model.attack_backward_train(x, test_data[1], step, epoch)
            train_log.append(log)
            save = pd.DataFrame(data=train_log)
            # save.to_csv('/home/Attack_sc/models/log-forward.csv')

        plot_reconstruction(model, test[1][1])