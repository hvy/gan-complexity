import argparse
import collections
import json

import numpy
import matplotlib
import matplotlib.pyplot as plt

# Use Latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', type=list, default=[
        'result_0051_std/logs/inception_score',
        'result_0056_res_n1/logs/inception_score',
        'result_0056_res_n2/logs/inception_score',
        'result_0056_res_n3/logs/inception_score',
        'result_0056_res_n5/logs/inception_score',
        #'result_0054_mbd_std/logs/inception_score',
        #'result_0055_mbd_res_n0/logs/inception_score',
        #'result_0055_mbd_res_n2/logs/inception_score',
        #'result_0055_mbd_res_n3/logs/inception_score',
        #'result_0055_mbd_res_n5/logs/inception_score',
        ])
    parser.add_argument('--out', type=str, default='inception_score_comparison.pdf')
    parser.add_argument('--keys', nargs='+', type=str, default=['inception_score_mean', 'inception_score_std'])
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def load_log(filename, keys):

    """Parse a JSON file and return a dictionary with the given keys. Each
    key maps to a list of corresponding data measurements in the file."""

    log = collections.defaultdict(list)
    with open(filename) as f:
        for data in json.load(f):
            for key in keys:
                log[key].append(data[key])
    return log


def plot_logs(filename, keys, labels, logs, limit=None):

    # Hard-coded figure size
    plt.figure(figsize=(8, 4))

    for i, (label, log) in enumerate(zip(labels, logs)):
        means = log[keys[0]]
        stds = log[keys[1]]

        if limit is not None:
            means = means[:limit]
            stds = stds[:limit]

        print('Log:', label)

        highest_i = numpy.argmax(means)
        highest_mean = means[highest_i]
        highest_mean_std = stds[highest_i]

        print(highest_mean)
        print(highest_mean_std)

        assert(len(means) == len(stds))

        if 'no_mbd' in label or 'mbd' not in label:
            linestyle = '-'
            linewidth = 1
        else:
            linestyle = '-'
            linewidth = 1

        if 'vanilla' in label or 'std' in label:
            color = 'C1'
        elif 'n0' in label or 'n1' in label:
            color = 'C2'
        elif 'n2' in label:
            color = 'C3'
        elif 'n3' in label:
            color = 'C4'
        elif 'n5' in label:
            color = 'C5'

        label = label.split('/')[0]
        if 'no_mbd' not in label and 'mbd' in label:
            uses_mbd = True
        else:
            uses_mbd = False

        if 'std' in label:
            label = 'Standard'
        elif 'n0' in label or 'n1' in label:
            label = 'Residual, N=1'
        elif 'n2' in label:
            label = 'Residual, N=2'
        elif 'n3' in label:
            label = 'Residual, N=3'
        elif 'n5' in label:
            label = 'Residual, N=5'

        if uses_mbd:
            label += ', MBD'

        # plt.errorbar(range(len(means)), means, yerr=stds, fmt='-o', markersize=2, label=label)
        plt.plot(range(len(means)), means, linestyle, linewidth=linewidth, label=label, color=color)

    ax = plt.gca()
    ax.yaxis.grid(clip_on=False, linestyle=':')
    ax.legend(loc=4)
    ax.set_ylim([1.5, 8])
    if limit is not None:
        ax.set_xlim([0, limit])
    plt.ylabel('Inception Score')
    plt.xlabel('Epochs')
    plt.xlim(xmin=0)

    print('Saving plot {}'.format(filename))

    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close()


def main(args):
    logs = [load_log(log, args.keys) for log in args.logs]
    plot_logs(args.out, args.keys, args.logs, logs, args.limit)


if __name__ == '__main__':
    args = parse_args()
    main(args)
