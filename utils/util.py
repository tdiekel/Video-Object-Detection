import datetime
import os

import cv2
import numpy as np
import yaml


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


def time_from_ms(ms):
    """ Converts Milliseconds to Hours, Minutes, Seconds, Milliseconds
    """

    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
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


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self, increment=1):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += increment

    def frames_seen(self):
        # return num of seen frames
        return self._numFrames

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
