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
We used this function to first load the pure pistachios images which will be used for model training.
### Plotting some of `pure_pistachios` images:
![pure_pistachios](https://github.com/user-attachments/assets/626dbd7c-ca6b-45c5-a32a-619386fc3892)
## 2. Data Augmentation: 
Augmentation techniques such as horizontal and vertical flips are applied to increase the training dataset's diversity.
~~~python
def h_flip(image): return np.fliplr(image)
def v_flip(image): return np.flipud(image)
~~~
### 3. Model Architecture:
This model architecture is inspired by **VGG19**, a convolutional neural network that is 19 layers deep. The VGG19 model, originally designed for image classification tasks, excels at extracting intricate features from images and classifying them into 1000 object categories, such as keyboards, animals, and other everyday objects. While the original VGG19 architecture ends with fully connected layers for classification, in my approach, I’ve adapted this structure to serve as an **autoencoder** for anomaly detection.

![vggarchitecture](https://github.com/user-attachments/assets/9e7d0b99-3c58-462c-bcbe-d48ba0561973)
In particular, I retained the **encoder** aspect of VGG19, leveraging its powerful feature extraction through convolutional layers and max-pooling operations. However, instead of terminating with classification layers, I introduced a mirrored **decoder** to reconstruct the input data from its latent representation. This design allows the network to learn a compressed version of the data and then reconstruct it, making it well-suited for identifying anomalies by analyzing the reconstruction errors. Thus, the architecture maintains the robust feature extraction capability of VGG19 while extending it into a generative process for anomaly detection.

![Autoencoder_architecture](https://github.com/user-attachments/assets/a2b0964a-d5cb-4096-aa3d-3467947788d5)
~~~python
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        #encoder
        self.encoder = Sequential([
            Conv2D(128, (3,3), activation='relu', padding='same',input_shape=targetsize),
            MaxPooling2D((2,2), padding='same'),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2), padding='same'),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2), padding='same'),
            Conv2D(16, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2), padding='same')
])
        self.decoder = Sequential([
            Conv2D(16, (3,3), activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(3,(3,3),activation='sigmoid',padding='same')])
        
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
~~~
The activation function used in the convolutional layers is ReLU, except for the last layer of the decoder, which uses a sigmoid activation function to ensure output pixel values are in the range [0, 1].
## 4. Model Training:
**visualization of Loss by epochs:**
![loss](https://github.com/user-attachments/assets/53b5b02c-49d4-48a5-a8d2-6f8da5011455)

It seems like the model is training reasonably well, with the training loss (loss) and validation loss (val_loss) decreasing over epochs. The model is learning to reconstruct the input images, as evidenced by the decreasing loss values.
#### Generate for a randomly chosen image:
![see_how_it_generates](https://github.com/user-attachments/assets/7023f411-4e33-4a6d-93fe-f68fae97811f)
Although the generated image is not as detailed as the original, we will see that the neural network still successfully detected adulterated pistachios. This shows that the model can identify anomalies even with less detailed images.
## 5. Model Evaluation:
### Distribution of reconstruction error by level of fraud:
The reconstruction error is a crucial metric in evaluating the performance of the autoencoder. It measures how well the model can recreate the input images after they pass through the encoder and decoder.
~~~python
def calculate_reconstruction_error(autoencoder, images):
    reconstructed_images = autoencoder.predict(images)
    reconstruction_errors = np.mean(np.square(images - reconstructed_images), axis=(1, 2, 3))
    return reconstruction_errors
~~~
![re_10%](https://github.com/user-attachments/assets/277ecb70-944d-4e20-a6c7-1584e309b45b)
![re_20%](https://github.com/user-attachments/assets/078ca144-0625-43cd-acbf-e1c00e08098a)
![re_30%](https://github.com/user-attachments/assets/28e75753-bb3a-45a5-9082-835cfcd094db)
![re_40%](https://github.com/user-attachments/assets/45ecad7c-a225-41e6-a269-daa3f2d767af)
![re_50%](https://github.com/user-attachments/assets/87cdcdd0-c8f9-4e15-b82b-7f8420ec733f)

The visualizations reveal that: as the amount of spinach adulteration in the pistachio samples increases, the reconstruction error does as well. **This means the model struggles more to accurately reconstruct heavily adulterated images. In simpler terms, the higher the spinach content, the harder it is for the model to tell what's going on in the images.**
### Choosing a threshold for the reconstruction error:
To detect anomalies, we need to choose a threshold for the reconstruction error. Here's how I did it:
* **Calculate Training Loss:** We first find out how well the autoencoder can reconstruct the training images. This gives us a set of error values.
~~~python
train_loss = calculate_reconstruction_error(autoencoder, training_images)
~~~
* **Set the Threshold:** We determine the threshold by taking the average of these errors and adding one standard deviation. This helps us set a limit that is higher than most of the reconstruction errors from the training data.
~~~python
threshold = np.mean(train_loss) + np.std(train_loss)
~~~
In this case, our threshold is about **0.0028552006**. Any error above this value will be considered an anomaly.

### Calculating evaluation metrics: Accuracy, Precision, Recall and F1-score:
To evaluate the performance of our autoencoder in detecting anomalies, we compute several metrics: accuracy, precision, recall, and F1 score.
The results of the evaluation reveal that our model performs exceptionally well, achieving an accuracy of approximately 97.3%, a precision of 98.5%, a recall of 98.2%, and an F1 score of 98.4%. These metrics indicate that the model is highly effective at correctly identifying both normal and anomalous images, validating its robustness for the task of anomaly detection.

![metrics](https://github.com/user-attachments/assets/6469641b-6869-411f-b20a-7e883bd78cf4)

**Overall, these evaluation metrics underscore the reliability and efficacy of the anomaly detection system in distinguishing between pure and spinach-adulterated pistachios.**

---
