from abc import ABC
from abc import abstractmethod


class AbstractDetector(ABC):
    @abstractmethod
    def load_model(self, device_id):
        """ Load model into memory
        """
        pass

    @abstractmethod
    def detect(self, frames):
        """ Detect objects in frames
        """
        pass

    @abstractmethod
    def visualize(self, frame, detections):
        """ Visualize detections in frames
        """
        pass

    @abstractmethod
    def process_detections(self, results, append_all_detections_dict, append_per_class_dict):
        """ Processes the detections
        """
        pass