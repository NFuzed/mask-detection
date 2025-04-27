import os
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from Code.models.base_model import BaseModel


class HogSvmModel(BaseModel):
    def __init__(self, img_size=(128, 128), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), random_state = 42):
        self.img_size = img_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.random_state = random_state
        self.model = SVC(kernel='linear', C=1.0, random_state=42, gamma=0.1)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def load_data(self, images_path, labels_path):
        """Extracts HOG features from images and labels files."""
        X = []
        y = []
        for filename in os.listdir(images_path):
            img_id = os.path.splitext(filename)[0]
            label_file = os.path.join(labels_path, f"{img_id}.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label = int(f.read().strip())
                img = imread(os.path.join(images_path, filename), as_gray=True)
                img = resize(img, self.img_size)
                features = hog(img, orientations=self.orientations,
                               pixels_per_cell=self.pixels_per_cell,
                               cells_per_block=self.cells_per_block,
                               block_norm='L2-Hys')
                X.append(features)
                y.append(label)
        return np.array(X), np.array(y)

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")

    def save_model(self, path):
        joblib.dump(self.model, path)

    def predict(self, loaded_data):
        X_test, y_test = loaded_data
        y_pred = self.model.predict(X_test)
        return y_test, y_pred

    def predict_single(self, image):
        image = image[..., 0]

        features = hog(image, orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm='L2-Hys')
        return self.model.predict([features])[0]

# Validation Accuracy: 0.8351
# Accuracy: 0.8384
# Precision: 0.8335
# Recall: 0.8384
# F1-Score: 0.8344
# Confusion Matrix:
# [[ 24  27   0]
#  [ 28 355   5]
#  [  2  12   5]]