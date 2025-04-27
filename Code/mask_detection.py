from Code.utilities import model_visualizer

class MaskDetection:
    def __init__(self, image_dir_path, model):
        visualizer = model_visualizer.ModelVisualizer(model=model)
        visualizer.predict_and_plot(image_dir_path)
