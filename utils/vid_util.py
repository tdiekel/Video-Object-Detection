import json
import os
import subprocess
import sys
from contextlib import contextmanager
import time
import cv2
import logging
from utils.setup_logger import logger

# Create class logger
from utils.util import FPS, time_from_ms

logger = logging.getLogger('VideoUtil')


def get_video_meta(filename, meta='duration'):
    """ Get metadata from video file

    :param filename: Path to video file
    :param meta: metadata: duration, avg_frame_rate
    :return: metadata
    """

    if meta == 'fps':
        meta = 'avg_frame_rate'

    if meta not in ['duration', 'nb_frames', 'avg_frame_rate', 'width', 'height']:
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


def get_video_meta_data(filepath):
    """ Get video metadata

    :param filepath: Path to video file
    :return: {'filename': str, 'duration': float, 'fps': float, 'im_height': int, 'im_width': int}
    """

    filename = os.path.basename(filepath)
    duration = float(get_video_meta(filepath, meta='duration'))
    nb_frames = int(get_video_meta(filepath, meta='nb_frames'))
    fps = float(eval(get_video_meta(filepath, meta='fps')))
    im_height = get_video_meta(filepath, meta='height')
    im_width = get_video_meta(filepath, meta='width')

    return {'filename': filename,
            'duration': duration,
            'nb_frames': nb_frames,
            'fps': fps,
            'im_height': im_height,
            'im_width': im_width}


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

        while not len(frames_to_analyze) == 0:
            frames = []
            timestamps = []

            # get video frames
            for i in range(batch_size):
                # Set frame idx
                if not len(frames_to_analyze) == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_analyze.pop())
                else:
                    continue

                # Read frame
                ret, frame = cap.read()

                # Check boolean flags indicating whether images should be converted to RGB.
                if convert_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not ret:
                    logger.warn("Failed to capture frame")
                    continue

                frames.append(frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            yield frames, timestamps


def video_worker(filepath, video_meta, e_vid, queues, frame_step_size=1, batch_size=1):
    frame_gen = yield_images(filepath, frame_step_size, batch_size)

    fps = FPS().start()

    while True:

        full = any(q.full() for q in queues)

        if not full:
            # Read next frames
            try:
                frames, timestamps = next(frame_gen)
                num_frames = len(frames)

                fps.update(len(frames))

                # Skip if no frames where retrived
                if num_frames == 0:
                    continue

            except StopIteration:
                # Inform main thread the all frames were read
                e_vid.set()

                logger.info("Finished loading video, waiting for detection processes")

                # Stop fps measurement
                fps.stop()
                h, m, s, _ = time_from_ms(fps.elapsed() * 1000)
                logger.info('Processed video {} with {:.2f} frames per second in {:.0f}:{:.0f}:{:.0f}'.format(
                    os.path.basename(filepath), fps.fps(), h, m, s))
                break

            for q in queues:
                q.put((frames, timestamps, video_meta))

            logger.info('Put frame {}/{} in queues'.format(
                fps.frames_seen(), video_meta['nb_frames'] // frame_step_size))

        else:
            time.sleep(0.1)
