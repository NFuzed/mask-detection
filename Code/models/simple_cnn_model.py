import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from skimage.io import imread
from skimage.transform import resize

from Code.models.base_model import BaseModel

class SimpleCNNModel(BaseModel):
    def __init__(self, img_size=(128, 128), num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None

    def load_data(self, images_path, labels_path):
        X = []
        y = []
        for filename in os.listdir(images_path):
            img_id = os.path.splitext(filename)[0]
            label_file = os.path.join(labels_path, f"{img_id}.txt")
            img = imread(os.path.join(images_path, filename))
            img = resize(img, self.img_size)
            X.append(img)
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def build_model(self):
        model = Sequential([
            Input(shape=(self.img_size[0], self.img_size[1], 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def predict(self, loaded_data):
        X, y = loaded_data  # loaded_data is (X, y) tuple
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y, y_pred

    def predict_single(self, image):
        image = image.reshape((1, *image.shape))  # add batch dimension
        probs = self.model.predict(image)
        return np.argmax(probs, axis=1)[0]

