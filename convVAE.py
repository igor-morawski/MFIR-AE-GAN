import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

import dataset as dt

INPUT_SHAPE = (32, 32, 1)
EPOCHS = 500
LATENT_DIM = 50
NUM_EXAMPLES_TO_GENERATE = 16

OUTPUT_DIR = "output"
TMP_DIR = "tmp"


class ConvVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
        )

        self.generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps=tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar=tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps=tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits=self.generative_net(z)
        if apply_sigmoid:
            probs=tf.sigmoid(logits)
            return probs
        return logits

#END OF CLASS

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi=tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * \
                tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(model, x):
    mean, logvar=model.encode(x)
    z=model.reparameterize(mean, logvar)
    x_logit=model.decode(z)

    cross_ent=tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x)
    logpx_z=-tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz=log_normal_pdf(z, 0., 0.)
    logqz_x=log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss=compute_loss(model, x)
    gradients=tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='jet')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(os.path.join(TMP_DIR, 'image_at_epoch_{:04d}.png'.format(epoch)))
  plt.close(fig)

if __name__ == "__main__":
    def make_sure_path_exists(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
    make_sure_path_exists(TMP_DIR)
    make_sure_path_exists(OUTPUT_DIR)

    dataset = dt.Dataloader_RAM(ids = [121, 122, 123])
    processor = dt.Processor()
    data = dataset.load()
    data = processor.align_timestamps(data) # align frames ()
    data = processor.retime(data, step = 3)

    train_images = data[0][0][0]
    test_images = data[0][1][0]
    train_images = train_images.reshape(train_images.shape[0], *INPUT_SHAPE).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], *INPUT_SHAPE).astype('float32')

    #normaliation
    def minmax_norm(images, min = None, max = None):
        #interframe normalization, the set is assumed to come from the same recording here!
        if not min:
            min = images.min()
        if not max:
            max = images.max()
        return (images-min)/(max-min)
    
    min, max = 20, 40
    train_images = minmax_norm(train_images, min, max)
    test_images = minmax_norm(test_images, min, max)

    TRAIN_BUF = 60000
    BATCH_SIZE = 100

    TEST_BUF = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

    optimizer=tf.keras.optimizers.Adam(1e-4)

    epochs=EPOCHS
    num_examples_to_generate=NUM_EXAMPLES_TO_GENERATE
    latent_dim = LATENT_DIM
    
    random_vector_for_generation=tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model=ConvVAE(latent_dim)
    
    generate_and_save_images(model, 0, random_vector_for_generation)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, '
                'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            generate_and_save_images(
                model, epoch, random_vector_for_generation)

    anim_file = os.path.join(OUTPUT_DIR, 'cvae.gif')
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(TMP_DIR,'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

