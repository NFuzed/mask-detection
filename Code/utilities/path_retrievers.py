import os

class PathRetrievers:
    def __init__(self):
        root_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        dataset_path = os.path.join(root_dir_path, 'CW_Dataset')

        self.path_to_dataset_test_images = os.path.join(dataset_path, 'test', 'images')
        self.path_to_dataset_test_labels = os.path.join(dataset_path, 'test', 'labels')

        self.path_to_dataset_train_images = os.path.join(dataset_path, 'train', 'images')
        self.path_to_dataset_train_labels = os.path.join(dataset_path, 'train', 'labels')

        self.path_to_export_trained_models = os.path.join(root_dir_path, 'Models')

        self.path_to_personal_dataset_images = os.path.join(root_dir_path, 'Personal_Dataset', 'images')


if __name__ == '__main__':
    path_retriever = PathRetrievers()
    print(path_retriever.path_to_dataset_test_images)
    print(path_retriever.path_to_export_trained_models)

