import logging
import os
from abc import ABC

# import cv2
# import matplotlib as mpl
import numpy as np
import tensorflow as tf

from detectors.BaseDetector import BaseDetector
from utils.setup_logger import logger
from utils.tensorflow import label_map_util
# from utils.tensorflow import visualization_utils as vis_util
from utils.util import yield_images, time_from_ms, get_video_meta

# from matplotlib import pyplot as plt

# Set tensorflow logger to ERROR level
tf.logging.set_verbosity(tf.logging.ERROR)

# Create class logger
logger = logging.getLogger('TensorFlowDetector')


class TensorFlowDetector(BaseDetector, ABC):

    def __init__(self, args):
        super().__init__(args, args.tf_batch_size)

        '''Create empty attribute fields'''
        # Graph
        self.sess = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None

        # Load label map for visualisation
        self.category_index = self.load_tf_label_map()

    def load_tf_label_map(self):
        """ Load the label map.
                :return: TensorFlow specific label map
                """
        logger.info('Loading label map')

        label_map = label_map_util.load_labelmap(self.args.tf_label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.args.max_class_id,
                                                                    use_display_name=True)
        return label_map_util.create_category_index(categories)

    def load_model(self):
        """ Load the TensorFlow model into memory and get tensor names
        """
        logger.info('Loading TensorFlow model')

        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(self.args.frozen_inference_graph, 'rb') as fid:
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

    def detect(self, frames):
        """ Runs the detection for one or more frames with the loaded model.

        :param frames: loaded and preprocessed frames
        :return: (boxes, scores, classes, num)
        """
        # Perform the actual detection by running the model with the frame as input
        return self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frames})

    def run(self):
        # Load the model into memory
        self.load_model()

        # Iterate video files
        for filepath in self.get_next_video():
            filename = os.path.basename(filepath)

            # Get video meta data
            duration = float(get_video_meta(filepath, meta='duration'))
            fps = float(eval(get_video_meta(filepath, meta='fps')))
            im_height = get_video_meta(filepath, meta='height')
            im_width = get_video_meta(filepath, meta='width')

            logger.info('Running detection for {}'.format(filename))

            # Iterate frames
            for frames, timestamps in yield_images(filepath, self.args.fps_step, self.batch_size):
                # Run detection
                boxes, scores, classes, num = self.detect(frames)

                # Get detections above threshold
                detections = np.where(scores >= self.args.thresh)

                # Iterate through detections in frames
                for frame_i in np.unique(detections[0]).tolist():
                    # Count objects in scene
                    objects_in_scene_count = np.where(detections[0] == frame_i)[0].size

                    # Log detection
                    h, m, s, ms = time_from_ms(timestamps[frame_i])
                    logger.info('Detected {} object(s) at {:.0f}:{:.0f}:{:.0f}.{:.0f}'.format(objects_in_scene_count,
                                                                                              h, m, s, ms))

                    # Create empty placeholder
                    timestamp = 0
                    class_id = []
                    score = []
                    bbox = []

                    # Iterate objects in scene
                    for object_i in range(objects_in_scene_count):
                        # Get object information
                        timestamp = timestamps[detections[0][object_i]]
                        class_id.append(int(classes[detections][object_i]))
                        score.append(scores[detections][object_i])

                        # Transform relative to absolute values
                        ymin, xmin, ymax, xmax = boxes[detections][object_i].tolist()
                        bbox.append({
                            'ymin': ymin * im_height,
                            'xmin': xmin * im_width,
                            'ymax': ymax * im_height,
                            'xmax': xmax * im_width
                        })

                        # Save detail information
                        self.append_detail_dict(filename, timestamp, class_id[object_i], score[object_i],
                                                bbox[object_i], object_i + 1, objects_in_scene_count)

                    '''Save information per class'''
                    self.append_info_dict(filename, timestamp, class_id, score)

                    if self.args.visualize:  # TODO
                        # Run visualization for batch
                        # self.visualize()
                        pass

            # Save detections before loading next video
            self.save_detections(filename, duration, fps)

    def visualize(self, frame, detections):
        pass
