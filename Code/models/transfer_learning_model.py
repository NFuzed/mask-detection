import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import class_weight

from Code.models.base_model import BaseModel

class TransferLearningModel(BaseModel):
    def __init__(self, img_size=(128, 128), num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.base_model = None

    def load_data(self, images_path, labels_path):
        X = []
        y = []
        for filename in os.listdir(images_path):
            img_id = os.path.splitext(filename)[0]
            label_file = os.path.join(labels_path, f"{img_id}.txt")
            img = imread(os.path.join(images_path, filename))
            img = resize(img, self.img_size)
            img = preprocess_input(img)
            X.append(img)
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def build_model(self):
        base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                 include_top=False,
                                 weights='imagenet')
        base_model.trainable = False  # Freeze base

        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

        self.model.compile(optimizer=Adam(learning_rate=1e-3),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.base_model = base_model

    def train(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, oversample = True):
        if self.model is None:
            self.build_model()

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split
        )

        train_generator = datagen.flow(X_train, y_train, subset='training', batch_size=batch_size)
        val_generator = datagen.flow(X_train, y_train, subset='validation', batch_size=batch_size)


        if oversample:
            weights_array  = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1, 2]),
                y=y_train
            )
            class_weights = dict(enumerate(weights_array))
            print(class_weights)

            self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                class_weight=class_weights
            )
        else:
            self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
            )

        self.fine_tune()

    def fine_tune(self):
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-20]:  # Keep first layers frozen
            layer.trainable = False
        self.model.compile(optimizer=Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def predict(self, loaded_data):
        X, y = loaded_data
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y, y_pred

    def predict_single(self, image):
        # image = preprocess_input(image)
        image = image.reshape((1, *image.shape))  # add batch dimension
        probs = self.model.predict(image)
        return np.argmax(probs, axis=1)[0]