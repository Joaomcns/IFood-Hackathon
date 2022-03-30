import numpy as np
from PIL import Image
from keras.models import load_model

pictures = {0: 'test1.jpg', 1: 'test2.jpg', 3: 'test3.jpeg'}
model = load_model('initial_cnn.h5')
img_test = Image.open(pictures[1])
img_test = img_test.resize((64, 64))
img_arr = 1 / 255 * np.array(img_test)
img_arr = np.expand_dims(img_arr, axis=0)
pred = model.predict(img_arr, verbose=1)
labels = {0: 'baked_goods', 1: 'deserts_and_drinks', 2: 'light_meals', 3: 'main_dishes', 4: 'soups_and_spicey_food'}
print(labels[pred.argmax()])

