import imageio
import os
import numpy as np
from keras.models import load_model

generator = load_model('people_generator_5000.h5')
random_dim = 100


def discriminator_batch(path, generator, batch_size):
    x = 0
    fake_images, images = [], []
    while True:
        files = iter(os.listdir(f'{path}'))
        while x < batch_size:
            file = next(files)
            noise = np.random.normal(0, 1, size=[1, random_dim])
            fake_image = generator.predict(noise)
            fake_images.append(fake_image)

            image = np.array(imageio.imread(os.path.join(path, file)))
            image = image / 255
            image.resize((1, 3888))
            images.append(image)
            x += 1

        imgs = np.concatenate([images, fake_images])
        zeros = np.zeros(batch_size * 2)
        zeros[:batch_size] = 0.9
        yield imgs, zeros


# g = discriminator_batch('people_emoji_final', generator, 64)
# print('g made')
# print('g ', next(g))
# print('g ', next(g))


def generator_batch(path, batch_size):
    x = 0
    images = []
    while True:
        try:
            files = iter(os.listdir(f'{path}'))
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            while x < batch_size:
                file = next(files)
                image = np.array(imageio.imread(os.path.join(path, file)))
                image = image / 255
                image.resize((1, 3888))
                images.append(image)
                x += 1
        except StopIteration:
            continue
        yield np.array(noise), np.array(images)


# g = generator_batch('final_emoji', 264)
# print(next(g))
# print(next(g))

def gan_batch(batch_size):
    while True:
        noise = np.random.normal(0, 1, size=[batch_size, random_dim])
        ones = np.ones(batch_size)
        yield noise, ones


g = gan_batch(64)
print(next(g))
