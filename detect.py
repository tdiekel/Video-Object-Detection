import argparse
import os
import sys
from glob import glob

from detectors.TensorFlowDetector import TensorFlowDetector
# import faulthandler
# faulthandler.enable()

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
        assert_file([os.path.join(args.data_path, fname) for fname in video_files])

    if '/' not in args.label_map:
        args.label_map = os.path.join(args.dataset_path, args.label_map)
    assert_file(args.label_map)

    '''Subparsers'''
    # Check tensorflow args
    assert_folder(args.graph_path)

    args.frozen_inference_graph = os.path.join(args.graph_path, 'frozen_inference_graph.pb')
    assert_file(args.frozen_inference_graph)

    if '/' not in args.tf_label_map:
        args.tf_label_map = os.path.join(args.graph_path, args.tf_label_map)
    assert_file(args.tf_label_map)

    return args


def parse_args(args):
    """ Parse the arguments.
    """

    # Create parser, subparser and groups
    parser = argparse.ArgumentParser(prog='Object Detection Evaluator', add_help=False)

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
                            default=0.6)

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
                           help='File type of the videos. (Default: avi)',
                           type=str,
                           choices=['avi', 'mp4'],
                           default='avi')
    ds_parser.add_argument('--fps-step',
                           help='Step size for frames when analyzing (e.g. fps=2 analyzes every second frame).',
                           type=int,
                           default=1)
    ds_parser.add_argument('--label-map',
                           help='Name of json label map file. Can also be a path when not in \'--data-path\'.',
                           type=str,
                           default='label_map.json')

    # Create subparsers for frameworks
    subparsers = parser.add_subparsers(description='Frameworks specific arguments', dest='framework')

    # Tensorflow parser
    tf_parser = subparsers.add_parser(name='tensorflow', help='TensorFlow settings', add_help=False)
    tf_req = tf_parser.add_argument_group('Required arguments')
    tf_opt = tf_parser.add_argument_group('Optional arguments')
    tf_req.add_argument('-g', '--graph-path',
                        help='Root path of the graph. Folder must contain the \'frozen_inference_graph.pb\' file.',
                        type=str,
                        required=True)
    tf_opt.add_argument('--input-type',
                        help='Input type the inference graph expects.',
                        nargs='?',
                        choices=['image_tensor'],  # , 'tf_example'],
                        default='image_tensor')
    tf_req.add_argument('--thresh',
                        help='Minimum score threshold for detection.',
                        type=float,
                        default=0.5)
    tf_req.add_argument('--tf-label-map',
                        help='Name of pbtxt label map file. Can also be a path when not in \'--graph-path\'.',
                        type=str,
                        default='label_map.pbtxt')
    tf_req.add_argument('--max-class-id',
                        help='Class id with the highest number in the dataset. ',
                        type=int,
                        required=True)
    tf_req.add_argument('--tf-batch-size',
                        help='Number of images or frames to run at once through the network. Reduce when OOM.',
                        type=int,
                        default=1)
    tf_opt.add_argument('-h', '--help',
                        action='help',
                        help='Show this help message and exit.')

    # Add help at last
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-h', '--help',
                          action='help',
                          help='Show this help message and exit.')

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.framework == 'tensorflow':
        detector = TensorFlowDetector(args)
    else:
        sys.exit('Framework \'{}\' currently not supported.'.format(args.framework))

    detector.run()


if __name__ == '__main__':
    main()
