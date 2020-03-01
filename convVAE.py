import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import argparse

import dataset as dt
INPUT_SHAPE = (32, 32, 1)

tf.random.set_seed(1234)

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

def generate_and_save_images(model, epoch, test_input, directory):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='jet')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(os.path.join(directory, 'image_at_epoch_{:04d}.png'.format(epoch)))
  plt.close(fig)

def plot_ELBO(train_elbo_log, test_elbo_log, model_dir, prefix="", suffix=""):
    plt.plot(np.array(train_elbo_log))
    plt.plot(np.array(test_elbo_log))
    plt.title('model ELBO')
    plt.ylabel('ELBO')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(model_dir, prefix+"model_ELBO"+suffix+".png"))
    plt.close()
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        type=str,
                        default="output",
                        help='Path to the output folder')
    parser.add_argument('--tmp_dir',
                        type=str,
                        default="tmp",
                        help='Path to the tmp files folder')
    parser.add_argument('--epochs',
                        type=int,
                        default=250,
                        help='How many epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--num_examples_to_generate',
                        type=int,
                        default=16,
                        help='How many examples to genereate in visualization gif.')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=50,
                        help='How many examples to genereate in visualization gif.')
    parser.add_argument('--prefix',
                        type=str,
                        default="",
                        help='Prefix to identify the files.')
    parser.add_argument('--suffix',
                        type=str,
                        default="",
                        help='Prefix to identify the files.')
    FLAGS, unparsed = parser.parse_known_args()
    def make_sure_path_exists(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
    make_sure_path_exists(FLAGS.tmp_dir)
    make_sure_path_exists(FLAGS.output_dir)
    filenames = glob.glob(os.path.join(FLAGS.tmp_dir,'image*.png'))
    for filename in filenames:
        os.remove(filename)

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

    optimizer=tf.keras.optimizers.Adam(FLAGS.lr)

    
    random_vector_for_generation=tf.random.normal(
        shape=[FLAGS.num_examples_to_generate, FLAGS.latent_dim])
    model=ConvVAE(FLAGS.latent_dim)
    
    generate_and_save_images(model, 0, random_vector_for_generation, FLAGS.tmp_dir)

    train_loss_log = []
    test_loss_log = []
    for epoch in range(1, FLAGS.epochs + 1):
        start_time = time.time()
        train_loss = tf.keras.metrics.Mean()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
            train_loss(compute_loss(model, train_x))
        train_elbo = -train_loss.result()
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            print('Epoch: {}, Train set ELBO: {}. Test set ELBO: {}, '
                'time elapse for current epoch {}'.format(epoch, train_elbo,
                                                            elbo,
                                                            end_time - start_time))
            generate_and_save_images(
                model, epoch, random_vector_for_generation, FLAGS.tmp_dir)
            train_loss_log.append(train_elbo)
            test_loss_log.append(elbo)
    plot_ELBO(train_loss_log, test_loss_log, FLAGS.output_dir, FLAGS.prefix, FLAGS.suffix)
            

    anim_file = os.path.join(FLAGS.output_dir, FLAGS.prefix+'convVAE'+FLAGS.suffix+'.gif')
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(FLAGS.tmp_dir,'image*.png'))
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

