import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from test_engine import run_inference
import argparse
def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--num_classes', dest='num_classes',
        help='Number of classes in your custom dataset',
        default=None, type=int)
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
        args = parse_args()
        run_inference(
                args=args,
                ind_range=None,
                multi_gpu_testing=False,  # args.multi is original
                check_expected_results=False)  # True is original
