from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def load_data(self, images_path, labels_path):
        ...

    @abstractmethod
    def load_model(self, model_path):
        ...

    @abstractmethod
    def save_model(self, model_path):
        ...

    @abstractmethod
    def predict(self, loaded_data):
        ...

    @abstractmethod
    def predict_single(self, image):
        ...