import datetime
import os
import logging
import cv2
import numpy as np
import yaml

# Create class logger
logger = logging.getLogger('Util')

epsilon = 0.00001


def create_dir(path):
    """ Create dir, if it doesnt exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def load_config():
    """ Loads config file
    :return: config dict
    """

    with open("./config.yaml", 'r') as f:
        return yaml.safe_load(f)


def time_from_s(s):
    """ Converts Seconds to Hours, Minutes, Seconds, Milliseconds
    """

    m, s = divmod(s, 60 + epsilon)
    h, m = divmod(m, 60 + epsilon)
    return h, m, s


def time_from_ms(ms):
    """ Converts Milliseconds to Hours, Minutes, Seconds, Milliseconds
    """

    s, ms = divmod(ms, 1000 + epsilon)
    m, s = divmod(s, 60 + epsilon)
    h, m = divmod(m, 60 + epsilon)
    return h, m, s, ms


def preprocess(frames, image_width=None, image_height=None, normalize=False):
    """ Preprocess batch of images
    """

    preprocessed_frames = []

    # Iterate each frame in frames
    for frame in frames:

        # resize frame
        if image_width is not None and image_height is not None:
            frame = cv2.resize(frame, (image_width, image_height))

        # normalize frame
        if normalize:
            frame = np.divide(np.asarray(frame, np.float32), 255)

        # add result
        preprocessed_frames.append(frame)

    return np.asarray(preprocessed_frames)


class Timer:
    def __init__(self):
        # store the start time, end time
        self._start = None
        self._end = None

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()


class FPS(Timer):
    def __init__(self):
        super().__init__()

        # store total number of frames
        # that were examined between the start and end intervals
        self._numFrames = 0

    def update(self, increment=1):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += increment

    def frames_seen(self):
        # return num of seen frames
        return self._numFrames

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class VideoTimer(Timer):
    def __init__(self):
        super().__init__()

        self._detection_time = 0
        self._detection_timer = Timer()

        self._video_time = 0
        self._video_timer = Timer()

        self._videos_seen = 0
        self._video_duration_processed = 0

    def start_detection(self):
        self._detection_timer.start()

    def stop_detection(self):
        self._detection_timer.stop()

        self._detection_time += self._detection_timer.elapsed()

        self._detection_timer._start = None
        self._detection_timer._end = None

    def start_video(self):
        self._video_timer.start()

    def stop_video(self, video_duration):
        self._video_timer.stop()
        self._video_time += self._video_timer.elapsed()

        self._video_timer._start = None
        self._video_timer._end = None

        self._videos_seen += 1
        self._video_duration_processed += video_duration

    def print_times(self):
        processing_time_per_vid = (self._video_time - self._detection_time) / self._videos_seen
        processing_time_per_vid_second = (self._video_time - self._detection_time) / self._video_duration_processed

        processing_str = 'Pre- and post processing took\n' \
                         '\t{:02.0f}:{:02.0f} (m:s) per video\n' \
                         '\t{:.2f}s per video second\n'.format(*divmod(processing_time_per_vid, 60),
                                                               processing_time_per_vid_second)

        detection_time_per_vid = self._detection_time / self._videos_seen
        detection_time_per_vid_second = self._detection_time / self._video_duration_processed

        detection_str = 'Running inference on networks took\n' \
                        '\t{:02.0f}:{:02.0f} (m:s) per video\n' \
                        '\t{:.2f}s per video second\n'.format(*divmod(detection_time_per_vid, 60),
                                                              detection_time_per_vid_second)

        elapsed_time = self.elapsed()
        time_per_vid = elapsed_time / self._videos_seen
        time_per_vid_second = elapsed_time / self._video_duration_processed

        time_str = 'Whole process took\n' \
                   '\t{:02.0f}:{:02.0f} (m:s) per video\n' \
                   '\t{:.2f}s per video second\n' \
                   '\t{:.0f}:{:02.0f}:{:02.0f} (h:m:s) for {} videos\n' \
                   '\t\twith a total duration of {:.0f}:{:02.0f}:{:02.0f} (h:m:s)\n'.format(
                    *divmod(time_per_vid, 60),
                    time_per_vid_second,
                    *time_from_s(elapsed_time),
                    self._videos_seen,
                    *time_from_s(self._video_duration_processed))

        logger.info('\n=== Times spend on the different tasks ===\n'
                    + processing_str + detection_str + time_str)
