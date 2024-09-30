# Anomaly Detection of Spinach Adulterated Pistachios using an Autoencoder Architecture
This notebook is also available on my Kaggle account: https://www.kaggle.com/code/asmahwimli/anomaly-detection-spinach-adulterated-pistachios
---

This project focuses on identifying spinach-adulterated pistachios using an autoencoder architecture. By training on pure pistachio images, the autoencoder is designed to detect anomalies, i.e., adulterated pistachios, using reconstruction errors.

## Dataset
The dataset consists of images of pistachios, categorized into:
* **Pure Pistachios:** Images of pure, unadulterated pistachios.
* **Spinach-Adulterated Pistachios:** These are further divided into subsets based on spinach content:
    * 10% spinach adulteration
    * 20% spinach adulteration
    * 30% spinach adulteration
    * 40% spinach adulteration
    * 50% spinach adulteration
Dataset link: https://www.kaggle.com/datasets/kazimkili/spinach-adulterated-pistachios/data
## Goal of the project: 
The aim is to build an autoencoder for anomaly detection to differentiate between pure and adulterated pistachios. The autoencoder is trained on pure pistachios, and it flags adulterated pistachios by identifying significant deviations in reconstruction error.

## 1. Loading the dataset:
Images are loaded from the directories, resized to 256x256x3, and normalized.
~~~python
targetsize=(256,256,3)
def load_images_from_directory(directory_path, targetsize):
    images = []
    img_list = os.listdir(directory_path)
    for i in tqdm(img_list):
        img = tf.keras.preprocessing.image.load_img(directory_path + '/' + str(i), target_size=targetsize[:2])
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0 #Normalization to make values between 0 and 1
        images.append(img)
    return np.array(images)
~~~
