import logging
from abc import ABC

import tensorflow as tf

from detectors.AbstractDetector import AbstractDetector
from utils.setup_logger import logger
from utils.tensorflow import label_map_util
from utils.util import *

# Create class logger
logger = logging.getLogger('TensorFlowDetector')


class TensorFlowDetector(AbstractDetector, ABC):
    def __init__(self):
        logger.info('Initializing')

        # Load config
        self.config = load_config()['object_detection']
        self.config_tf = self.config['tensorflow']

        # Create empty attribute fields
        self.sess = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None

        # Load label map for visualisation
        self.category_index = self.load_label_map()

    def load_label_map(self):
        """ Load the label map.
        :return: TensorFlow specific label map
        """

        logger.info('Loading label map')

        label_map = label_map_util.load_labelmap(self.config_tf['label_map'])
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.config_tf['max_class_id'],
                                                                    use_display_name=True)
        return label_map_util.create_category_index(categories)

    def load_model(self):
        """ Load the TensorFlow model into memory and get tensor names
        """
        logger.info('Loading model')

        # Set gpu device for process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config['gpu_id'])

        # with tf.device('/gpu:{}'.format(self.config['gpu_id'])):
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(self.config_tf['graph_path'], 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Save session for later access
            self.sess = tf.Session(graph=detection_graph)

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        logger.info('Model loaded')

    def detect(self, frames):
        """ Runs the detection for one or more frames with the loaded model.

        :param frames: loaded and preprocessed frames
        :return: (boxes, scores, classes, num)
        """

        # Perform the actual detection by running the model with the frame as input
        return self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frames})

    def visualize(self, frames, detections):
        """ Visualize detections in frames
        """
        # TODO create
        pass

    def process_detections(self, results, append_all_detections_dict, append_per_class_dict):
        ((boxes, scores, classes, num), video_meta) = results

        # Get detections above threshold
        detections = np.where(scores >= self.config_tf['thresh'])

        # Iterate through detections in frames
        for frame_i in np.unique(detections[0]).tolist():
            # Count objects in scene
            objects_in_scene_count = np.where(detections[0] == frame_i)[0].size

            # Create empty placeholder
            timestamp = 0
            class_id = []
            score = []
            bbox = []

            # Iterate objects in scene
            for object_i in range(objects_in_scene_count):
                # Get object information
                timestamp = video_meta['timestamps'][detections[0][object_i]]
                class_id.append(int(classes[detections][object_i]))
                score.append(scores[detections][object_i])

                # Transform relative to absolute values
                ymin, xmin, ymax, xmax = boxes[detections][object_i].tolist()
                bbox.append({
                    'ymin': ymin * video_meta['im_height'],
                    'xmin': xmin * video_meta['im_width'],
                    'ymax': ymax * video_meta['im_height'],
                    'xmax': xmax * video_meta['im_width']
                })

                # Append to all_detections dict
                append_all_detections_dict(video_meta['filename'], timestamp, class_id[object_i], score[object_i],
                                           bbox[object_i], object_i + 1, objects_in_scene_count)

            # Append to per_class_detections dict
            append_per_class_dict(video_meta['filename'], timestamp, class_id, score)
