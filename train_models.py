import os

import numpy as np
from PIL import Image
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Dropout, Input
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers

def build_discriminator():
    model = Sequential([
        Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Flatten(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model


def build_generator():
    model = Sequential([
        Dense(1024, input_shape=(64,)),
        Reshape((4, 4, 64)),

        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(), # Batch normalization maybe risky
        LeakyReLU(0.2),

        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', activation='tanh')
    ])
    return model

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image_array = np.array(image)
    image_normalized = (image_array / 127.5) - 1

    return image_normalized

def log(text):
    file = open("log", "a")
    print(text)
    file.write(str(text) + "\n")
    file.close()

if __name__ == '__main__':

    # dirname = "data/crawled_cakes"
    dirname = "data/generated_cakes"

    data = []
    for name in os.listdir(dirname):
        data.append(load_and_preprocess_image(os.path.join(dirname, name)))
    data = np.array(data)
    print("DATASET READY")


    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.00001))
    discriminator.trainable = False


    def custom_loss(y_true, y_pred):
        discriminator_output = discriminator(y_pred)
        return tf.keras.losses.binary_crossentropy(discriminator_output, np.ones(discriminator_output.shape))

    generator = build_generator()
    generator.compile(loss=custom_loss, optimizer=Adam(0.00003))
    generator.trainable = False

def calculate_accuracies(batch_size, generator, discriminator, data):
    noise = np.random.uniform(0, 1, size=[batch_size, 64])
    generated_data = generator.predict(noise)
    real_data = data[np.random.randint(0, data.shape[0], size=batch_size)]
    X = np.concatenate([real_data, generated_data], axis=0)
    output = discriminator.predict(X)
    correct_discriminator_guesses = (output[:batch_size] > 0.5).sum() + (output[batch_size:] < 0.5).sum()
    discriminator_accuracy = correct_discriminator_guesses / (batch_size * 2) * 100
    incorrect_dirscriminator_guesses = (output[batch_size:] > 0.5).sum()
    generator_accuracy = incorrect_dirscriminator_guesses / batch_size * 100
    return discriminator_accuracy, generator_accuracy


def train(epochs, batch_size, starting_epoch, discriminator, generator, data):
    batch_count = (data.shape[0] // batch_size) - 1
    accuracy_batch_size = batch_size

    for e in range(starting_epoch, epochs):
        random.shuffle(data)
        for i in range(batch_count):
            noise = np.random.uniform(0, 1, size=[batch_size, 64])
            generated_data = generator.predict(noise)
            real_data = data[i*batch_size: (i+1)*batch_size]

            X = np.concatenate([real_data, generated_data], axis=0)
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 1
            # for i in range(batch_size):
            #     y_dis[i] -= random.uniform(0, 0.05)
            # for i in range(batch_size, batch_size * 2):
            #     y_dis[i] += random.uniform(0, 0.05)

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)
            discriminator.trainable = False

            noise = np.random.uniform(0, 1, size=[batch_size, 64])
            y_gen = np.ones(batch_size) # TODO sketchy
            generator.trainable = True
            g_loss = generator.train_on_batch(noise, y_gen)
            generator.trainable = False

        discriminator_accuracy, generator_accuracy = calculate_accuracies(accuracy_batch_size, generator, discriminator, data)

        log("Epoch " + str(e))
        log("Discriminator accuracy:" + str(discriminator_accuracy) + "%")
        log("Generator accuracy:" + str(generator_accuracy) + "%")

        log(f"Epoch {e} - D loss: {d_loss} | G loss: {g_loss}")
        if e % 5 == 0:
            save_generated_data(e, generator)
        if e % 30 == 0 and e != 0:
            save_models(e, discriminator, generator)

def save_generated_data(epoch, generator):
    noise = np.random.uniform(0, 1, size=[1, 64])
    generated_data = generator.predict(noise)

    save_dir = 'generated_images'
    os.makedirs(save_dir, exist_ok=True)

    generated_image = (generated_data[0] + 1) * 127.5
    generated_image = generated_image.astype(np.uint8)
    image = Image.fromarray(generated_image)
    image.save(os.path.join(save_dir, f'generated_image_epoch{epoch}.png'), 'PNG')

def save_models(epoch, discriminator, generator):
    os.makedirs('saved_models', exist_ok=True)
    discriminator.save(f'saved_models/discriminator_epoch_{epoch}.h5')
    generator.save(f'saved_models/generator_epoch_{epoch}.h5')

if __name__ == '__main__':
    train(epochs=10001, batch_size=64, starting_epoch=0, discriminator=discriminator, generator=generator, data=data)
