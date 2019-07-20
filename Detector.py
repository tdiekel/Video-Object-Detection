import json
import logging
import os
import threading
import time
from bisect import bisect_left
from glob import glob
from multiprocessing import Process, Queue, Pool, Lock, Event

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from detectors.TensorFlowDetector import TensorFlowDetector
from recognizer.TensorFlowRecognizer import TensorFlowRecognizer
from utils.setup_logger import logger
from utils.util import create_dir, load_config, VideoTimer
from utils.vid_util import get_video_meta_data, video_worker

# Create class logger
logger = logging.getLogger('Detector')

# Time to wait for last results to be processed
WAIT_TIME = 5.0
# Flag if we have to continue waiting
CONTINUE_WAITING = True
# Flag if timer was started
WAIT_TIMER_STARTED = False


def timer_callback():
    """ Simple callback to flip :param CONTINUE_WAITING: flag
    """
    global CONTINUE_WAITING
    CONTINUE_WAITING = False


# If processes don't finish in the allotted amount of time, then terminate all of the processes.
TIMEOUT = 10


class Detector:
    def __init__(self, args):
        logger.info('Initializing')
        self.args = args

        # Load config
        self.config = load_config()

        # Get selected detector
        if self.config['object_detection']['use_tensorflow']:
            self.detector = TensorFlowDetector()
            self.detector_class = TensorFlowDetector
            self.detector_config = self.config['object_detection']['tensorflow']

        # Load label maps
        self.categories = json.load(open(self.config['data']['label_map'], 'r')).get('classes')
        self.categories = [{'id': cat['id'], 'name': cat['name']} for cat in self.categories]
        self.road_condition_label = TensorFlowRecognizer.load_recognizer_label()

        # Create dicts for simple translation between name and id
        self.cat2id = {cat['name']: cat['id'] for cat in self.categories}
        self.id2cat = {cat['id']: cat['name'] for cat in self.categories}

        # Initialize vars
        self.per_class_detections = None
        self.all_detections = None
        self.all_recognitions = None
        self.prepare_dicts()

    def prepare_dicts(self):
        # Prepare dicts for detection results
        self.per_class_detections = {
            cat['id']: {
                'timestamp': [],
                'score': [],
                'objects_in_scene_count': []
            } for cat in self.categories
        }
        self.all_detections = {
            'timestamp': [],
            'class_id': [],
            'score': [],
            'bbox': [],
            'num_of_object_in_scene': [],
            'objects_in_scene_count': []
        }

        # Prepare dict for recognition result
        self.all_recognitions = {
            'timestamp': [],
            'label': [],
            'score': []
        }

    def get_next_video(self):
        """ Generator to yield video filepaths
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

    def run(self):
        # Get globals
        global WAIT_TIMER_STARTED, CONTINUE_WAITING

        # Get some settings for readability
        batch_size = self.config['settings']['batch_size']
        fps_step_size = self.config['data']['fps_step']
        queue_size = self.config['settings']['queue_size']

        # Start timer
        video_timer = VideoTimer().start()

        # Create queues, limit size for image queues
        q_img_rc = Queue(maxsize=queue_size // batch_size)
        q_img_det = Queue(maxsize=queue_size // batch_size)
        q_result_rc = Queue()
        q_result_det = Queue()

        lock_det = Lock()

        # Get GPU device IDs
        if isinstance(self.config['object_detection']['device_id'], int):
            device_ids = [self.config['object_detection']['device_id']]
        else:
            device_ids = [int(device_id) for device_id in self.config['object_detection']['device_id'].split(',')]

        # Create processes and events
        events = [Event()]
        processes = [Process(target=self.recognize_worker, args=(events[0], q_img_rc, q_result_rc,))]
        for i, device_id in enumerate(device_ids):
            events.append(Event())
            processes.append(Process(target=self.detection_worker,
                                     args=(events[-1], q_img_det, q_result_det, device_id, lock_det,)))

        # Start processes
        for process in processes:
            process.start()

        # Wait for processes to start
        for e in events:
            e.wait()

        # Iterate video files
        for filepath in self.get_next_video():
            logger.info('Running detection for {}'.format(os.path.basename(filepath)))

            video_meta = get_video_meta_data(filepath)

            e_vid = Event()
            p_vid = Process(target=video_worker,
                            args=(filepath, video_meta, e_vid, [q_img_rc, q_img_det], fps_step_size, batch_size,))
            p_vid.start()

            # Measure processing time
            video_timer.start_detection()
            video_timer.start_video()

            # Iterate frames
            while True:

                if not q_result_rc.empty():
                    # Process next result
                    self.process_recognitions(q_result_rc.get())

                if not q_result_det.empty():
                    # Process next result
                    self.detector.process_detections(q_result_det.get(),
                                                     self.append_all_detections_dict,
                                                     self.append_per_class_dict)

                # Check if all frames were set to queues
                if e_vid.is_set():

                    # Check if video process is alive
                    if p_vid.is_alive():
                        # Finish video process
                        p_vid.join()

                    # Check if all frames were processed
                    if q_img_rc.empty() and q_img_det.empty():
                        # Check if all results were processed
                        if q_result_rc.empty() and q_result_rc.empty():
                            # Wait for WAIT_TIME to make sure all items in the result queues where processed
                            if not WAIT_TIMER_STARTED:
                                # Start timer
                                timer = threading.Timer(WAIT_TIME, timer_callback)
                                timer.start()

                                # Set timer started
                                WAIT_TIMER_STARTED = True

                            # Wait for time to call timer_callback
                            if not CONTINUE_WAITING:
                                break
                            else:
                                time.sleep(0.1)
                        else:
                            time.sleep(0.1)
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.05)

            video_timer.stop_detection()

            # Save detections before loading next video
            self.save_detections(video_meta)

            video_timer.stop_video(video_meta['duration'])

            # Reset timer
            WAIT_TIMER_STARTED = False
            CONTINUE_WAITING = True

        # Stop timer
        video_timer.stop()

        logger.info('Finished processing videos')
        video_timer.print_times()

        logger.info('Waiting for background processes to finish')

        start = time.time()
        while time.time() - start <= TIMEOUT:
            if any(p.is_alive() for p in processes):
                time.sleep(.1)  # Just to avoid hogging the CPU
            else:
                # All the processes are done, break now.
                break
        else:
            # We only enter this if we didn't 'break' above.
            logger.warning('Timed out, killing all processes')
            for p in processes:
                p.terminate()
                p.join()

    @staticmethod
    def recognize_worker(event, q_img_rc, q_result_rc):
        """ Runs recognition for the images in queue :param q_img_rc:
                and puts results in queue :param q_append_rc:

        :param event: Event which is set when model is loaded. Type: multiprocessing.Event
        :param q_img_rc: queue with images from the video file
        :param q_result_rc: queue to fill with recognition results
        """

        # Load the recognition model into memory
        recognizer = TensorFlowRecognizer()
        recognizer.load_model()

        # Send signal the loading is finished
        event.set()

        while True:
            if not q_img_rc.empty():
                # Get next frames
                (frames, timestamps, video_meta) = q_img_rc.get()

                # Preprocess
                frames_preprocessed = recognizer.preprocess(frames)

                # Run recognition
                recognitions = recognizer.recognize(frames_preprocessed)

                # Put results in queue
                q_result_rc.put((recognitions, timestamps, video_meta))

    def detection_worker(self, event, q_img_det, q_result_det, device_id, lock):
        """ Runs detection for the images in queue :param q_img_det:
            and puts results in queue :param q_result_det:

        :param event: Event which is set when model is loaded. Type: multiprocessing.Event
        :param q_img_det: queue with images from the video file
        :param q_result_det: queue to fill with detection results
        :param device_id: ID of the device to use
        :param lock: Used to block the q_img_det queue. Type: multiprocessing.Lock
        """

        # Load the detection model into memory
        detector = self.detector_class()
        detector.load_model(device_id)

        # Send signal the loading is finished
        event.set()

        while True:
            with lock:
                # Get next frames
                (frames, timestamps, video_meta) = q_img_det.get()

            # Run recognition
            detections = detector.detect(frames)

            # Put results in queue
            q_result_det.put((detections, timestamps, video_meta))

    def process_recognitions(self, results):
        """ Append information to all_recognitions dict
        """

        # Unpack values
        (recognitions, timestamps, video_meta) = results

        # Iterate all recognitions
        for i, recognition in enumerate(recognitions):
            # Save frame information
            self.all_recognitions['timestamp'].append(timestamps[i])

            # Save recognition information for all classes
            self.all_recognitions['label'].append([])
            self.all_recognitions['score'].append([])

            for j, label in enumerate(self.road_condition_label):
                self.all_recognitions['label'][-1].append(label)
                self.all_recognitions['score'][-1].append(recognition[j])

    def append_all_detections_dict(self, timestamp, class_id, score, bbox, num_of_object_in_scene,
                                   objects_in_scene_count):
        """ Append information to all_detections dict
        """

        # Save frame information
        self.all_detections['timestamp'].append(timestamp)

        # Save detection information
        self.all_detections['class_id'].append(class_id)
        self.all_detections['score'].append(score)
        self.all_detections['bbox'].append(bbox)
        self.all_detections['num_of_object_in_scene'].append(num_of_object_in_scene)
        self.all_detections['objects_in_scene_count'].append(objects_in_scene_count)

    def append_per_class_dict(self, timestamp, class_ids, scores):
        """ Append information to per_class_detections dict
        """

        # Iterate classes
        for c_id in set(class_ids):
            # Get detections with c_id
            det_idx = np.where(np.array(class_ids) == c_id)

            # Get highscore
            class_scores = np.array(scores)[det_idx]
            highscore = class_scores[np.argmax(class_scores)]

            # Save frame information
            self.per_class_detections[c_id]['timestamp'].append(timestamp)

            # Save detection information
            self.per_class_detections[c_id]['score'].append(highscore)
            self.per_class_detections[c_id]['objects_in_scene_count'].append(len(det_idx))

    def save_detections(self, video_meta):
        """ Saves detections to disk
        """

        # Rename output folder
        video_meta['filename'] = video_meta['filename'].replace('.', '_')

        # Save detections
        output_dir = os.path.join('./output', video_meta['filename'])
        create_dir(output_dir)

        # Save all_recognitions dict
        logger.info('Saving recognitions per class')

        # Interpolate and save recognitions
        self.interpolate_recognition_dict(output_dir, video_meta)

        # Save all_detections dict
        logger.info('Saving all detections in one file')

        output_path = os.path.join(output_dir, 'detect_00_all.csv')

        all_detections_df = pd.DataFrame(data=self.all_detections)
        all_detections_df.to_csv(output_path, index=None)

        # Save per_class_detections dict
        logger.info('Saving detections per class')

        # Fill list with all args to run through save_dicts_for_classes()
        arg_list = []
        for class_id in self.per_class_detections:
            output_path = os.path.join(output_dir, '{}_detect_{}.csv'.format(video_meta['filename'], class_id))
            arg_list.append(((class_id, video_meta), output_path))

        with Pool(processes=self.config['settings']['num_workers']) as pool:
            pool.starmap(self.save_dicts_for_classes, arg_list)

        # Clear dicts for next video file
        self.prepare_dicts()

    def save_label_maps(self, output_dir):
        pass

    def save_dicts_for_classes(self, args, output_path):
        """ Run the interpolation function with the args; saves result to disk
        """

        result = self.interpolate_detection_dict(*args)
        result.to_csv(output_path, index=None)

    def interpolate_detection_dict(self, class_id, video_meta):
        """ Interpolates the dict for a single class

        :param class_id: Class to interpolate dict for
        :param video_meta: Video meta data dict includes keys: 'filename', 'duration', 'fps'
        :return: {'filename': [],  'timestamp': [], 'score': [], 'objects_in_scene_count': []}
        """

        logger.info('Processing {} - {}'.format(class_id, self.id2cat[class_id]))

        # Extract relevant video meta data for readability
        duration = video_meta['duration']
        fps = video_meta['fps']

        # Extract relevant data for readability
        data = self.per_class_detections[class_id]
        # Get timestamps in seconds as np.array for use with np.where
        timestamps = np.array(data['timestamp']) / 1000

        # Calc number of necessary data points
        num_data_points = round(duration * self.config['data']['target_sample_rate'])
        # Get time interval between data points from sample rate
        sample_period = 1 / self.config['data']['target_sample_rate']

        '''Special case - No object of class class_id found'''
        if len(timestamps) == 0:
            return pd.DataFrame(data={
                # 'filename': [filename] * num_data_points,
                'timestamp': [i * sample_period for i in range(num_data_points)],
                'score': np.zeros(num_data_points),
                'objects_in_scene_count': np.zeros(num_data_points)
            })

        '''Normal case - Found at least one object of class class_id'''
        # Iterating over all possible timestamps of the data points twice
        # First iteration: Check if an object was detected near time t (+- max_t_delta)
        # Second iteration: Write all data points and interpolate between if necessary

        # Calc max time delta between detections
        max_t_delta = self.config['data']['fps_step'] / fps

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
                if t_delta <= self.config['data']['relevant_future']:
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

    def interpolate_recognition_dict(self, output_dir, video_meta):
        """ Interpolates recognitions for each class

        :param output_dir: Dir to save result in
        :param video_meta: Video meta data dict includes keys: 'filename', 'duration', 'fps'
        """

        # Calc number of necessary data points
        num_data_points = round(video_meta['duration'] * self.config['data']['target_sample_rate'])
        # Get time interval between data points from sample rate
        sample_period = 1 / self.config['data']['target_sample_rate']

        for class_id, class_label in enumerate(self.road_condition_label):
            # Prepare result dict
            result = {}

            # Get evaluated timestamps in seconds
            timestamp = np.array(self.all_recognitions['timestamp']) / 1000
            # Get scores for current class
            score = np.array(self.all_recognitions['score'])[:, class_id]

            # Get target timestamps
            result['timestamp'] = [i * sample_period for i in range(num_data_points)]
            # Create interpolation function and interpolate scores
            interpolation = interp1d(timestamp, score, fill_value='extrapolate')
            result['score'] = np.array([interpolation(ts) for ts in result['timestamp']])

            # Apply threshold
            result['score'][result['score'] < self.config['road_condition']['thresh']] = 0

            # Get output path for csv
            output_path = os.path.join(output_dir, '{}_recog_{}.csv'.format(video_meta['filename'], class_id))
            # Save to disk
            pd.DataFrame(data=result).to_csv(output_path, index=None)

            # Clear result dict
            del result
