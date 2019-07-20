import argparse
import os
import sys
from glob import glob

import yaml

from Detector import Detector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Range(object):
    """ Enables argparse to check ranges
        Source: https://stackoverflow.com/a/12117089/9943279
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return '{0}-{1}'.format(self.start, self.end)


def assert_folder(paths):
    """ Asserts every path in paths is a folder.
    :param paths: str or list
    """
    if isinstance(paths, list):
        for path in paths:
            assert os.path.isdir(path), 'Folder not found. \'{}\''.format(path)
    else:
        assert os.path.isdir(paths), 'Folder not found. \'{}\''.format(paths)


def assert_file(file_paths):
    """ Asserts every path in file_paths is a file.
        :param file_paths: str or list
        """
    if isinstance(file_paths, list):
        for file_path in file_paths:
            assert os.path.isfile(file_path), 'File not found. \'{}\''.format(file_path)
    else:
        assert os.path.isfile(file_paths), 'File not found. \'{}\''.format(file_paths)


def check_config(config):
    """ Check settings from config.yaml
    """

    # check settings
    assert isinstance(config['settings']['queue_size'], int), \
        'Queue size must be integer.'
    assert 0 < config['settings']['queue_size'], \
        'Queue size must be larger than 0.'

    assert isinstance(config['settings']['num_workers'], int), \
        'FPS step size must be integer.'
    assert 0 < config['settings']['num_workers'], \
        'FPS step size must be larger than 0.'

    assert isinstance(config['settings']['batch_size'], int), \
        'Batch size must be integer.'
    assert 0 < config['settings']['batch_size'], \
        'Batch size must be larger than 0.'

    # check data
    assert isinstance(config['data']['fps_step'], int), \
        'FPS step size must be integer.'
    assert 0 < config['data']['fps_step'], \
        'FPS step size must be larger than 0.'

    assert_file(config['data']['label_map'])

    assert isinstance(config['data']['target_sample_rate'], int), \
        'Target sample rate must be integer.'
    assert 0 < config['data']['target_sample_rate'], \
        'Target sample rate must be larger than 0.'

    assert isinstance(config['data']['relevant_future'], int), \
        'Relevant future must be integer.'
    assert 0 < config['data']['relevant_future'], \
        'Relevant future must be larger than 0.'

    # check road_condition
    assert_file(config['road_condition']['graph_path'])

    assert isinstance(config['road_condition']['thresh'], float), \
        'Threshold must be float.'
    assert 0.0 < config['road_condition']['thresh'] <= 1.0, \
        'Threshold must be between 0.0 and 1.0.'

    assert_file(config['road_condition']['label_map'])

    assert isinstance(config['road_condition']['num_classes'], int), \
        'Number of classes must be integer.'
    assert 0 < config['road_condition']['num_classes'], \
        'Number of classes must be larger than 0.'

    # check object_detection
    # TODO check for multiple frameworks

    # check tensorflow
    assert_file(config['object_detection']['tensorflow']['graph_path'])

    assert config['object_detection']['tensorflow']['input_type'] in ['image_tensor', 'tf_example'], \
        'Input type must be \'image_tensor\' or \'tf_example\'.'

    assert_file(config['object_detection']['tensorflow']['label_map'])

    assert isinstance(config['object_detection']['tensorflow']['thresh'], float), \
        'Threshold must be float.'
    assert 0.0 < config['object_detection']['tensorflow']['thresh'] <= 1.0, \
        'Threshold must be between 0.0 and 1.0.'

    assert isinstance(config['object_detection']['tensorflow']['max_class_id'], int), \
        'Max number of classes must be integer.'
    assert 0 < config['object_detection']['tensorflow']['max_class_id'], \
        'Max number of classes must be larger than 0.'

    # CPU / GPU settings
    possible_device_types = ['cpu', 'gpu']

    for system in ['road_condition', 'object_detection']:
        config[system]['device_type'] = config[system]['device_type'].lower()

        assert config[system]['device_type'] in possible_device_types, \
            'Device type must be either \'cpu\' or \'gpu\' ({}).'.format(system)

        if config[system]['device_type'] == 'cpu':
            assert isinstance(config[system]['device_id'], int), \
                'Device ID must be integer ({}).'.format(system)
            assert 0 <= config[system]['device_id'], \
                'Device ID must be 0 or greater ({}).'.format(system)

        if config[system]['device_type'] == 'gpu':
            assert isinstance(config[system]['device_id'], int) or isinstance(config[system]['device_id'], str), \
                'Device ID must be a single integer or multiple comma separated values ({}).'.format(system)
            if isinstance(config[system]['device_id'], int):
                assert 0 <= config[system]['device_id'], \
                    'Device ID must be 0 or greater ({}).'.format(system)
            if isinstance(config[system]['device_id'], str):
                device_ids = []
                for device_id in config[system]['device_id'].split(','):
                    try:
                        device_id = int(device_id)
                    except ValueError:
                        assert False, \
                            'Device ID must be integer, \'{}\' is not an integer ({}).'.format(device_id, system)
                    assert 0 <= device_id, \
                        'Device ID must be 0 or greater ({}).'.format(system)
                    assert device_id not in device_ids, \
                        'Device IDs must be unique ({}).'.format(system)
                    device_ids.append(device_id)


def check_args(args):
    """
    Checks if args are valid.
    :return: true if args vaild, else false
    """

    '''Argument groups'''
    # Visualization group
    if args.visualize:
        assert args.thresh_vis <= 0 or args.thresh_vis >= 1, 'Threshold must be between 0 and 1.'

    # Dataset group
    assert_folder(args.data_path)

    if args.videos == 'one':
        assert_file(os.path.join(args.data_path, args.fname))
    elif args.videos == 'list':
        assert_file([os.path.join(args.data_path, fname) for fname in args.list])
    elif args.videos == 'all':
        video_files = glob(os.path.join(args.data_path, '*' + args.file_type))
        assert_file(video_files)

    return args


def parse_args(args):
    """ Parse the arguments.
    """

    # Create parser, subparser and groups
    parser = argparse.ArgumentParser(prog='Object Detection Evaluator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Parser group for visualizations
    vis_parser = parser.add_argument_group('Visualize settings')
    vis_parser.add_argument('--visualize',
                            help='Show detections in video.',
                            action='store_const',
                            const=True,
                            default=False)
    vis_parser.add_argument('--thresh_vis',
                            help='Minimum score threshold for a box to be visualized.',
                            type=float,
                            default=0.6,
                            choices=[Range(0.0, 1.0)])

    # Parser group for dataset
    ds_parser = parser.add_argument_group('Data settings')
    ds_parser.add_argument('--data-path',
                           help='Path of the data folder.',
                           type=str,
                           required=True)
    ds_parser.add_argument('--videos',
                           help='Define which video(s) should be evaluated.\n'
                                '\tFor \'one\' set \'--fname\'.\n'
                                '\tFor \'list\' set \'--list\'.\n'
                                '\tFor \'all\' uses all videos in given folder with \'--file-type\'.',
                           type=str,
                           choices=['one', 'list', 'all'],
                           default='one')
    ds_parser.add_argument('--fname',
                           help='Filename of the video.',
                           type=str)
    ds_parser.add_argument('--list',
                           help='List of filenames of videos.',
                           type=str,
                           nargs='*')
    ds_parser.add_argument('--file-type',
                           help='File type of the videos.',
                           type=str,
                           choices=['avi', 'mp4'],
                           default='avi')

    return check_args(parser.parse_args(args))


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Assert config file exists
    assert os.path.isfile("./config.yaml"), 'Config file not found. Please do not rename or move!'

    # Check config
    with open("./config.yaml", 'r') as f:
        check_config(yaml.safe_load(f))

    detector = Detector(args)
    detector.run()


if __name__ == '__main__':
    main()
