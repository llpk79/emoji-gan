import base64
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import requests

from PIL import Image
from bs4 import BeautifulSoup as Soup


# All of the emojis.
emoji_URL = "https://unicode.org/emoji/charts/full-emoji-list.html"  # lines 1 - 116
person_URL = "https://unicode.org/emoji/charts/full-emoji-modifiers.html"  # lines 196 - 835, 846 - 885, 855 - 871,
# 891 - 936, 951 - 965, 981 - 1025. whoa.


# strip transparency dimension (because RGB channels are crazy in transparencent spaces)
def strip_transparency(i):
    px_transparent = i[:, :, 3] < 0.1
    i[px_transparent, 0:3] = 1
    i = i[:, :, 0:3]
    return i


def get_raw_emoji(URL):
    request = requests.get(URL)
    soup = Soup(request.text, 'html.parser')
    tags = soup.find_all(name='tr',)
    keepers = [tag for tag in tags if tag.td and (int(tag.td.text) in range(196, 836) or
                                                  int(tag.td.text) in range(846, 886) or
                                                  int(tag.td.text) in range(855, 872) or
                                                  int(tag.td.text) in range(891, 937) or
                                                  int(tag.td.text) in range(951, 966) or
                                                  int(tag.td.text) in range(981, 1026))]
    keeper_soup = [keeper.find_all(name='img') for keeper in keepers]
    all_pngs = []
    for png_list in keeper_soup:
        for png in png_list:
            all_pngs.append(png)
    png64_encoded = [png64_['src'].split(',')[1].encode() for png64_ in all_pngs]
    png64_decoded =[base64.decodebytes(png64) for png64 in png64_encoded]
    for i, png64 in enumerate(png64_decoded):
        with open(f'people_emoji/emoji_{i}.png', 'wb') as f:
            f.write(png64)


def strip_emoji():
    for emoji in os.listdir('people_emoji'):
        try:
            image = np.asarray(imageio.imread(os.path.join('people_emoji', emoji)))
            stripped_face = strip_transparency(image)
        except IndexError as e:
            print(e)
            print(emoji)
            continue
        array_face = np.asarray(stripped_face)
        imageio.imwrite(f'people_emoji_stripped/{emoji}', array_face, 'png')


def resize_emoji():
    for emoji in os.listdir('people_emoji_stripped'):
        image = Image.open(f'people_emoji_stripped/{emoji}')
        resized_image = image.resize((36, 36), Image.ANTIALIAS)
        np_image = np.asarray(resized_image)
        imageio.imwrite(f'people_emoji_final/{emoji}', np_image, 'png')


def main():
    # get_raw_emoji(person_URL)
    strip_emoji()
    resize_emoji()


if __name__ == '__main__':
    main()

    new_emoji = imageio.imread('final_emoji/emoji_104.png')
    fig = plt.figure(dpi=15)
    ax = fig.gca()
    ax.imshow(new_emoji, interpolation='nearest')
    ax.axis('off')
