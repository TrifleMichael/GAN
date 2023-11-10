import os

import numpy as np
from PIL import Image
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, Conv2DTranspose, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Dropout, Input
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers

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

def save_generated_data(epoch, generator):
    noise = np.random.uniform(0, 1, size=[1, 64])
    generated_data = generator.predict(noise)

    save_dir = 'regenerated_images'
    os.makedirs(save_dir, exist_ok=True)

    generated_image = (generated_data[0] + 1) * 127.5
    generated_image = generated_image.astype(np.uint8)
    image = Image.fromarray(generated_image)
    image.save(os.path.join(save_dir, f'generated_image_epoch{epoch}.png'), 'PNG')

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image_array = np.array(image)
    image_normalized = (image_array / 127.5) - 1
    return image_normalized

def train(epochs, batch_size, batch_count):
    starting_epoch = 240
    target_image = load_and_preprocess_image("target_cakes/3.png")

    def custom_old_loss(y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(target_image, y_pred)

    custom_objects = {'custom_loss': custom_old_loss}
    generator_old = load_model(f'saved_models/generator_epoch_{starting_epoch}.h5', custom_objects=custom_objects)

    generator = build_generator()
    generator.compile(loss=tf.keras.losses.mean_squared_error, optimizer=Adam(0.00001))
    generator.set_weights(generator_old.get_weights())
    generator.trainable = True

    for e in range(epochs):
        for _ in range(batch_count):
            noise = np.random.uniform(0, 1, size=[batch_size, 64])
            loss = generator.train_on_batch(noise, target_image)
        if e % 10 == 0:
            print("Loss: ", loss)
            print("----- Finished Epoch", e, "-----")
            save_generated_data(e, generator)
        
    
if __name__ == '__main__':
    train(10000, 32, 1)