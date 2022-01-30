from tensorflow.keras.models import load_model
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image

model = load_model("model.h5")

labels = ['cat', 'dog']
DIMENSIONS = (150, 150)

for file in glob.glob("./dataset/examples/*"):
    random_image = cv2.imread(file)
    plt.imshow(random_image)

    random_image = cv2.resize(random_image, dsize=DIMENSIONS)
    x = image.img_to_array(random_image)
    x = np.expand_dims(random_image, axis=0)

    result = model.predict(x)
    print(result[0])
    if result[0] > 0.5:
        plt.xlabel("dog")
    else:
        plt.xlabel("cat")

    plt.show()
