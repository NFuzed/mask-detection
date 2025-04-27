import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb

class ModelVisualizer:
    def __init__(self, model, img_size=(128, 128)):
        self.model = model
        self.img_size = img_size
        self.class_names = ["No Mask", "Mask", "Incorrect"]

    def load_images(self, images_path, num_images=9):
        images = []
        labels = []
        rng = np.random.default_rng()

        all_files = [f for f in os.listdir(images_path)]
        selected_files = rng.choice(all_files, num_images, replace=False)

        for file in selected_files:
            img = imread(os.path.join(images_path, file))
            img = self.preprocess_image(img)
            images.append(img)
            labels.append(file)  # Save filename for info

        return np.array(images), labels

    def predict_and_plot(self, images_path, num_images=9):
        X, _ = self.load_images(images_path, num_images)


        _, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            prediction = self.model.predict_single(X[i])

            if i >= len(X):
                break
            ax.imshow(X[i].astype(np.uint8))
            ax.axis('off')
            predicted_class = self.class_names[prediction]
            ax.set_title(f"Predicted: {predicted_class}")

        plt.tight_layout()
        plt.show()

    def preprocess_image(self, img):
        img = resize(img, self.img_size, preserve_range=True, anti_aliasing=True)

        if img.ndim == 2:
            # Grayscale → RGB
            img = gray2rgb(img)

        elif img.ndim == 3:
            if img.shape[-1] == 1:
                img = np.concatenate([img] * 3, axis=-1)
            elif img.shape[-1] > 3:
                img = img[:, :, :3]

        img = img.astype(np.uint8)

        return img
