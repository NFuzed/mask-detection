import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, model_path):
        self.model = model
        self.trained_model = model.load_model(model_path=model_path)

    def evaluate(self, images_path, labels_path):
        X_test, y_test = self.model.load_data(images_path, labels_path)
        y_pred = self.trained_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        return y_test, y_pred

    def create_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix', figsize=(8, 6), cmap='Blues'):
        """
        Plots a confusion matrix using seaborn heatmap.
        """
        class_names = ["No Mask", "Mask", "Incorrect"]
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
