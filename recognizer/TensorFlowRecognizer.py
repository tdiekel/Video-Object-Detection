import logging
from abc import ABC

import tensorflow as tf

from recognizer.AbstractRecognizer import AbstractRecognizer
from utils.setup_logger import logger
from utils.util import *

# Set tensorflow logger to ERROR level
tf.logging.set_verbosity(tf.logging.ERROR)

# Create class logger
logger = logging.getLogger('TensorFlowRecognizer')


class TensorFlowRecognizer(AbstractRecognizer, ABC):
    def __init__(self):
        logger.info('Initializing')

        self.config = load_config()['road_condition']

        # Create empty attribute fields
        self.sess = None
        self.input_operation = None
        self.output_operation = None

        # Load label
        self.label = []
        self.load_label()

    @staticmethod
    def load_recognizer_label():
        """ Load road condition label map
        """

        # Load config
        config = load_config()['road_condition']

        label = []

        proto_as_ascii_lines = tf.gfile.GFile(config['label_map']).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())

        return label

    def load_label(self):
        """ Load road condition label map
        """

        logger.info('Loading label map')

        proto_as_ascii_lines = tf.gfile.GFile(self.config['label_map']).readlines()
        for l in proto_as_ascii_lines:
            self.label.append(l.rstrip())

    def load_model(self):
        """ Load the TensorFlow model into memory and get tensor names
        """

        logger.info('Loading model')

        recognition_graph = tf.Graph()

        with recognition_graph.as_default():
            rc_graph_def = tf.GraphDef()

            with tf.gfile.GFile(self.config['graph_path'], 'rb') as fid:
                serialized_graph = fid.read()
                rc_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(rc_graph_def, name='')

            # Save session for later access
            self.sess = tf.Session(graph=recognition_graph)

        self.input_operation = recognition_graph.get_operation_by_name(self.config['input_layer'])
        self.output_operation = recognition_graph.get_operation_by_name(self.config['output_layer'])

        logger.info('Model loaded')

    def preprocess(self, frames):
        """ Preprocess frames with settings of loaded model
        """

        return preprocess(frames,
                          image_width=self.config['image_width'],
                          image_height=self.config['image_height'],
                          normalize=True)

    def recognize(self, frames):
        """ Runs the recognition for one or more frames with the loaded model.

        :param frames: loaded and preprocessed frames
        :return: recognition score for each class. shape = (batch_size, class_count)
        """

        return self.sess.run(self.output_operation.outputs[0],
                             {self.input_operation.outputs[0]: frames})

    def visualize(self, frame, detections):
        """ Visualize detections in frames
        """
        # TODO do
        pass

    def process_recognitions(self, results, append_all_detections_dict, append_per_class_dict):
        """ Processes the detections
        """
        # TODO do
        pass
