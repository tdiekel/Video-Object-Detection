import json
import os
import subprocess
import sys
from contextlib import contextmanager

import cv2


def create_dir(path):
    """ Create dir, if it doesnt exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def get_video_meta(filename, meta='duration'):
    """ Get metadata from video file

    :param filename: Path to video file
    :param meta: metadata: duration, avg_frame_rate
    :return: metadata
    """

    if meta == 'fps':
        meta = 'avg_frame_rate'

    if meta not in ['duration', 'avg_frame_rate', 'width', 'height']:
        sys.exit('Video metadata \'{}\' isnt supported. '
                 'Please choose \'{}\' or \'{}\''.format(meta, 'duration', 'avg_frame_rate'))

    command = ["ffprobe",
               "-loglevel", "quiet",
               "-print_format", "json",
               "-show_format",
               "-show_streams",
               filename
               ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    _json = json.loads(out)

    if 'format' in _json:
        if meta in _json['format']:
            return _json['format'][meta]

    if 'streams' in _json:
        # commonly stream 0 is the video
        for s in _json['streams']:
            if meta in s:
                return s[meta]

    return None

@contextmanager
def video_capture(*args, **kwargs):
    """Start video capture session"""

    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images(filepath, frame_step_size=1, batch_size=1):
    """Returns images from video
    
    :param filepath: Path to video file.
    :param frame_step_size: Returning every <frame_step_size> frame from video.
    :param batch_size: Number of frames to return at once.
    :return: [frames], [timestamps]
    """

    # capture video
    with video_capture(filepath) as cap:

        # Check if frames should be converted to RGB
        convert_to_rgb = cap.get(cv2.CAP_PROP_CONVERT_RGB)

        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get idx of frames to analyze
        frames_to_analyze = [x for x in range(0, frame_count, frame_step_size)]
        frames_to_analyze.reverse()

        while True:
            frames = []
            timestamps = []

            # get video frames
            for i in range(batch_size):
                # Set frame idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_analyze.pop())

                # Read frame
                ret, frame = cap.read()

                # Check boolean flags indicating whether images should be converted to RGB.
                if convert_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not ret:
                    raise RuntimeError("Failed to capture image")

                frames.append(frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            yield frames, timestamps


def time_from_ms(ms):
    """ Converts Milliseconds to Hours, Minutes, Seconds, Milliseconds
    """

    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s, ms
