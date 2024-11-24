import nltk
from nltk.corpus import words
from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

nltk.download('words')

word_list = words.words()
random.shuffle(word_list)
word_list = [word for word in word_list if len(word) <= 10]
word_list = word_list[:100000]

output_dir = './data/interim/5/4/nltk'
os.makedirs(output_dir, exist_ok=True)

img_width, img_height = 256, 64
font_size = 40
font = ImageFont.load_default()

for i, word in tqdm(enumerate(word_list),desc="Generating Dataset of Images",unit="word",total=len(word_list)):
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(word, font=font)
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    draw.text((text_x, text_y), word, font=font, fill='black')
    img.save(os.path.join(output_dir, f'{word}.png'))

print("Dataset of Images Generated Successfully, number of images:",len(word_list),"path:",output_dir)