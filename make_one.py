import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

with open('generator.pickle', 'rb') as f:
    gen = pickle.load(f)


def plot_generated_images(generator,
                          num_figs=100,
                          dim=(10, 10),
                          figsize=(10, 10)):
    plt.figure(figsize=figsize)

    for i in range(num_figs):
        random_dim = 100
        noise = np.random.normal(0, 1, size=[1, random_dim])
        generated_image = generator.predict(noise)
        generated_image = generated_image.reshape(36, 36, 3)
        plt.imshow((generated_image * 255).astype(np.uint8), interpolation='nearest')
        plt.subplot(dim[0], dim[1], i + 1,
                    xticklabels='',
                    yticklabels='')

    plt.axis('off')
    plt.grid(False)
    plt.savefig(f'my_emoji/gan_generated_image_tile.png')
    plt.show()


# def main(generator):
#     random_dim = 100
#     while True:
#         try:
#             command = input("Press 'enter' to generate a new emoji\nPress 'q' to quit\n")
#             if 'q' not in command.lower():
#                 continue
#             else:
#                 sys.exit(0)
#         except ValueError:
#             sys.exit()


if __name__ == "__main__":
    plot_generated_images(gen)
