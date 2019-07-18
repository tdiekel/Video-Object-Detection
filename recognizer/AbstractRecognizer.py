from abc import ABC
from abc import abstractmethod


class AbstractRecognizer(ABC):
    @abstractmethod
    def load_model(self):
        """ Load model into memory
        """
        pass

    @abstractmethod
    def recognize(self, frames):
        """ Recognizes objects in frames
        """
        pass

    @abstractmethod
    def visualize(self, frame, detections):
        """ Visualize detections in frames
        """
        # TODO do
        pass

    @abstractmethod
    def process_recognitions(self, results, append_all_detections_dict, append_per_class_dict):
        """ Processes the detections
        """
        # TODO do
        pass