import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--inception-gpu', type=int, default=1)
    parser.add_argument('--n-z', type=int, default=100)
    parser.add_argument('--g-res', type=int, default=-1)
    parser.add_argument('--d-res', type=int, default=-1)
    parser.add_argument('--g-weight-decay', type=float, default=0.00001)
    parser.add_argument('--d-weight-decay', type=float, default=0.00001)
    parser.add_argument('--mbd', action='store_true', default=False)
    parser.add_argument('--inception-score-model-path', type=str, default='./lib/inception_score/inception_score.model')
    parser.add_argument('--inception-score-n-samples', type=int, default=5000)
    return parser.parse_args()


def setup_dirs(base):
    path_templates = ['./{base}/', './{base}/logs/', './{base}/images/']
    for path_template in path_templates:
        path = path_template.format(base=base)
        if not os.path.exists(path):
            print('Creating directory', path)
            os.makedirs(path)
