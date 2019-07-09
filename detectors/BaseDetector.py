import json
import logging
import os
from abc import ABC
from abc import abstractmethod
from glob import glob

import numpy as np
import pandas as pd
from bisect import bisect_left
from utils.util import create_dir

logger = logging.getLogger('BaseDetector')


class BaseDetector(ABC):

    def __init__(self, args, batch_size):
        self.args = args
        self.batch_size = batch_size

        # Load label map
        self.categories = json.load(open(self.args.label_map, 'r')).get('classes')
        self.categories = [{'id': cat['id'], 'name': cat['name']} for cat in self.categories]

        # Create dicts for simple translation between name and id
        self.cat2id = {cat['name']: cat['id'] for cat in self.categories}
        self.id2cat = {cat['id']: cat['name'] for cat in self.categories}

        # Create empty dicts for detection results
        self.info = {
            cat['id']: {
                'filename': [],
                'timestamp': [],
                'score': [],
                'objects_in_scene_count': []
            } for cat in self.categories
        }
        self.detail = {
            'filename': [],
            'timestamp': [],
            'class_id': [],
            'score': [],
            'bbox': [],
            'num_of_object_in_scene': [],
            'objects_in_scene_count': []
        }

    def get_next_video(self):
        """Generator to yield video filepaths
        :return: video filepath
        """

        if self.args.videos == 'one':
            files = [os.path.join(self.args.data_path, self.args.fname)]
        elif self.args.videos == 'list':
            files = [os.path.join(self.args.data_path, fname) for fname in self.args.list]
        elif self.args.videos == 'all':
            files = glob(os.path.join(self.args.data_path, '*' + self.args.file_type))
        else:
            files = []

        for file in files:
            yield file

    def append_detail_dict(self, filename, timestamp, class_id, score, bbox, num_of_object_in_scene,
                           objects_in_scene_count):
        """ Append information to detail dict
        """

        # Save frame information
        self.detail['filename'].append(filename)
        self.detail['timestamp'].append(timestamp)

        # Save detection information
        self.detail['class_id'].append(class_id)
        self.detail['score'].append(score)
        self.detail['bbox'].append(bbox)
        self.detail['num_of_object_in_scene'].append(num_of_object_in_scene)
        self.detail['objects_in_scene_count'].append(objects_in_scene_count)

    def append_info_dict(self, filename, timestamp, class_ids, scores):
        """ Append information to info dict
        """

        # Iterate classes
        for c_id in set(class_ids):
            # Get detections with c_id
            det_idx = np.where(np.array(class_ids) == c_id)

            # Get highscore
            class_scores = np.array(scores)[det_idx]
            highscore = class_scores[np.argmax(class_scores)]

            # Save frame information
            self.info[c_id]['filename'].append(filename)
            self.info[c_id]['timestamp'].append(timestamp)

            # Save detection information
            self.info[c_id]['score'].append(highscore)
            self.info[c_id]['objects_in_scene_count'].append(len(det_idx))

    def save_detections(self, video_filename, duration, fps):
        """Saves detections to disk
        """

        # Save detections
        video_dir = os.path.join('./output', video_filename)
        create_dir(video_dir)

        # Save detail dict
        output_path = os.path.join(video_dir, '00_details.csv')

        logger.info('Saving all detections in one file ...')

        detail_df = pd.DataFrame(data=self.detail)
        detail_df.to_csv(output_path, index=None)

        # Save info dict

        logger.info('Saving detections per class')

        for class_id in self.info:
            output_path = os.path.join(video_dir, '{}_{}.csv'.format(class_id, self.id2cat[class_id]))

            logger.info('... current class: {} - {}'.format(class_id, self.id2cat[class_id]))

            info_df = self.interpolate_info_dict(video_filename, class_id, duration, fps)
            info_df.to_csv(output_path, index=None)

        # Clear dicts for next video file
        self.info = {
            cat['id']: {
                'filename': [],
                'timestamp': [],
                'score': [],
                'objects_in_scene_count': []
            } for cat in self.categories
        }
        self.detail = {
            'filename': [],
            'timestamp': [],
            'class_id': [],
            'score': [],
            'bbox': [],
            'num_of_object_in_scene': [],
            'objects_in_scene_count': []
        }

    # TODO target_sample_rate and relevant_future in args
    def interpolate_info_dict(self, filename, class_id, duration, fps, target_sample_rate=100, relevant_future=5):
        """ Interpolates the info dict for a single class

        :param filename: Video file name
        :param class_id: Class to interpolate dict for
        :param duration: Duration of the video in seconds
        :param fps: Average frame rate of the video file
        :param target_sample_rate: Target sample rate
        :param relevant_future: Time delta [seconds] in which detected objects are counted as one. The timeseries will be interpolated between these detections.
        :return: {'filename': [],  'timestamp': [], 'score': [], 'objects_in_scene_count': []}
        """

        # Extract relevant data for readability
        data = self.info[class_id]
        # Get timestamps in seconds as np.array for use with np.where
        timestamps = np.array(data['timestamp']) / 1000

        # Calc number of necessary data points
        num_data_points = round(duration * target_sample_rate)
        # Get time interval between data points from sample rate
        sample_period = 1 / target_sample_rate

        '''Special case - No object of class class_id found'''
        if len(timestamps) == 0:
            return pd.DataFrame(data={
                'filename': [filename] * num_data_points,
                'timestamp': [i * sample_period for i in range(num_data_points)],
                'score': np.zeros(num_data_points),
                'objects_in_scene_count': np.zeros(num_data_points)
            })

        '''Normal case - Found at least one object of class class_id'''
        # Iterating over all possible timestamps of the data points twice
        # First iteration: Check if an object was detected near time t (+- max_t_delta)
        # Second iteration: Write all data points and interpolate between if necessary

        # Calc max time delta between detections
        max_t_delta = self.args.fps_step / fps

        # Sorted list with times t where an object was detected
        object_at_time = []

        # Detail information for detected objects.
        # Format: {t : {'score': <score>, 'objects_in_scene_count': <count>}}
        object_info = {}

        # First iteration
        for i in range(0, num_data_points):
            # Current time
            t = i * sample_period

            # Check if object was detected at time t (+- max_t_delta)
            t_idx = np.where((timestamps >= t - max_t_delta) & (timestamps < t + max_t_delta))[0]

            if not len(t_idx) == 0:
                # Add time t to set
                object_at_time.append(t)

                # Save detail information
                object_info[t] = {
                    # 'true_timestamp': timestamps[t_idx[0]],
                    'score': data['score'][t_idx[0]],
                    'objects_in_scene_count': data['objects_in_scene_count'][t_idx[0]]
                }

        # Prepare result dict
        result = {
            'filename': [filename] * num_data_points,
            'timestamp': [],
            'score': [],
            'objects_in_scene_count': []
        }

        def add_empty(result, t):
            """ Function to add empty data point
            """
            result['timestamp'].append(t)
            result['score'].append(0)
            result['objects_in_scene_count'].append(0)

        # Second iteration
        for i in range(0, num_data_points):
            # Current time
            t = i * sample_period

            # Check if object at current time
            if t in object_info:
                result['timestamp'].append(t)
                result['score'].append(object_info[t]['score'])
                result['objects_in_scene_count'].append(object_info[t]['objects_in_scene_count'])

                # Future not relevant
                continue

            # Check if an object exists in relevant future
            # Add empty when first object not processed
            if t < object_at_time[0] + max_t_delta:
                add_empty(result, t)

                continue

            # Get index of potential next object
            next_obj = bisect_left(object_at_time, t)

            # Check if object exists
            if len(object_at_time) > next_obj:
                # Get time of previous and next object
                t_prev = object_at_time[next_obj - 1]
                t_next = object_at_time[next_obj]

                # Calc time delta
                t_delta = t_next - t_prev

                # Check time delta
                if t_delta <= relevant_future:
                    # Add object with interpolated values
                    interp_score = np.interp(t, [t_prev, t_next],
                                             [object_info[t_prev]['score'],
                                              object_info[t_next]['score']
                                              ])
                    interp_count = np.interp(t, [t_prev, t_next],
                                             [object_info[t_prev]['objects_in_scene_count'],
                                              object_info[t_next]['objects_in_scene_count']
                                              ])

                    result['timestamp'].append(t)
                    result['score'].append(interp_score)
                    result['objects_in_scene_count'].append(interp_count)

                    continue
                else:
                    # Not in relevant future, add empty
                    add_empty(result, t)

                    continue
            else:
                # No next object, add empty
                add_empty(result, t)

                continue

        # Return result
        return pd.DataFrame(data=result)

    @abstractmethod
    def load_model(self):
        """Load model into memory"""
        pass

    @abstractmethod
    def detect(self, frames):
        """Detect objects in frames"""
        pass

    @abstractmethod
    def visualize(self, frame, detections):
        """Visualize detections in frame"""
        pass

    @abstractmethod
    def run(self):
        """Run detection and visualization process"""
        pass
