import os
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize
import joblib

from Code.models.base_model import BaseModel


class SiftBovwSvmModel(BaseModel):
    def __init__(self, num_clusters=100, img_size=(128, 128)):
        self.num_clusters = num_clusters
        self.img_size = img_size
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
        self.svm = SVC(kernel='linear', C=1.0, random_state=42, gamma=0.1)
        self.sift = cv2.SIFT_create()

    def load_model(self, model_path):
        data = joblib.load(model_path)
        self.kmeans = data['kmeans']
        self.svm = data['svm']

    def load_data(self, images_path, labels_path):
        all_descriptors = []
        image_desc_mapping = []
        labels = []

        for filename in os.listdir(images_path):
            img_id = os.path.splitext(filename)[0]
            label_file = os.path.join(labels_path, f"{img_id}.txt")
            img = imread(os.path.join(images_path, filename), as_gray=True)
            img = resize(img, self.img_size)
            img = (img * 255).astype(np.uint8)  # SIFT needs uint8 images
            _, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)
                image_desc_mapping.append(descriptors)
                with open(label_file, 'r') as f:
                    label = int(f.read().strip())
                labels.append(label)
        return np.vstack(all_descriptors), image_desc_mapping, labels

    def create_histograms(self, image_desc_mapping):
        histograms = []
        for descriptors in image_desc_mapping:
            if descriptors is None:
                histograms.append(np.zeros(self.num_clusters))
                continue
            words = self.kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(self.num_clusters + 1))
            histograms.append(hist)
        return np.array(histograms)

    def train(self, sift_descriptors):
        all_descriptors, image_desc_mapping, labels = sift_descriptors
        self.kmeans.fit(all_descriptors)
        X = self.create_histograms(image_desc_mapping)
        y = np.array(labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.svm.fit(X_train, y_train)
        y_pred = self.svm.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")

    def save_model(self, model_path):
        joblib.dump({'kmeans': self.kmeans, 'svm': self.svm}, model_path)

    def predict(self, loaded_data):
        _, image_desc_mapping, labels = loaded_data
        X = self.create_histograms(image_desc_mapping)
        y_true = np.array(labels)
        y_pred = self.svm.predict(X)
        return y_true, y_pred

    def predict_single(self, image):
        if image is None or image.size == 0:
            raise ValueError("Image not loaded correctly!")

        if image.ndim == 3:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

        _, descriptors = self.sift.detectAndCompute(image, None)

        if descriptors is None:
            return self.majority_class

        words = self.kmeans.predict(descriptors)

        histogram = np.zeros(self.kmeans.n_clusters)
        for word in words:
            histogram[word] += 1
        histogram /= np.linalg.norm(histogram)  # Normalize the histogram

        # Use SVM to predict
        prediction = self.svm.predict([histogram])[0]
        return prediction
