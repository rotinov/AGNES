import os
from matplotlib import pyplot as plt
import matplotlib

import pandas
import numpy

_colors = [
    {"R": 0, "G": 122, "B": 255},
    {"R": 52, "G": 199, "B": 89},
    {"R": 88, "G": 86, "B": 214},
    {"R": 255, "G": 149, "B": 0},
    {"R": 255, "G": 45, "B": 85},
    {"R": 175, "G": 82, "B": 222},
    {"R": 255, "G": 59, "B": 48},
    {"R": 90, "G": 200, "B": 250},
    {"R": 255, "G": 204, "B": 0}
]


def _one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0  # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = numpy.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = numpy.exp(- 1. / decay_steps)
    sum_ys = numpy.zeros_like(xnews)
    count_ys = numpy.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = numpy.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = numpy.nan

    return xnews, ys, count_ys


def _symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    xs, ys1, count_ys1 = _one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _, ys2, count_ys2 = _one_sided_ema(-xolds[::-1], yolds[::-1],
                                       -high if high is not None else None,
                                       -low if low is not None else None,
                                       n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = numpy.nan
    return xs, ys, count_ys


def _mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def _plot_one(axis, root_dir, color=(0.1, 0.7, 0.9, 1.0), shaded_err=True, shaded_std=True, print_out=True):
    all_sub_dirs = next(os.walk(root_dir))[1]
    if print_out:
        print(" " * 4, "Seeds:", ", ".join(all_sub_dirs))

    all_rews = []
    all_steps = []
    for sub_dir in all_sub_dirs:
        fname = os.path.join(root_dir, sub_dir, "progress.csv")
        if not os.path.exists(fname):
            continue
        try:
            dataframe = pandas.read_csv(fname, index_col=None, comment='#')
        except:
            return

        item_steps = numpy.asarray(dataframe['misc/serial_timesteps'])
        item_rewards = numpy.asarray(dataframe['eprewmean'])

        all_rews.append(item_rewards)
        all_steps.append(item_steps)

    low = max(steps[0] for steps in all_steps)
    high = min(steps[-1] for steps in all_steps)

    if low == high:
        return

    steps_new = numpy.linspace(low, high, 512)

    smoothed_rews = []
    for steps, rews in zip(all_steps, all_rews):
        smoothed_rews.append(
            _symmetric_ema(steps, rews, low, high)[1]
        )

    av_rews = numpy.mean(smoothed_rews, axis=0)
    std = numpy.std(smoothed_rews, axis=0)
    stderr = std / numpy.sqrt(len(av_rews))

    plt.plot(steps_new, av_rews, color=color)

    if shaded_err:
        axis.fill_between(steps_new, av_rews - stderr, av_rews + stderr, color=[color], alpha=.4)
    if shaded_std:
        axis.fill_between(steps_new, av_rews - std, av_rews + std, color=[color], alpha=.2)


def draw_plot(path, shaded_err: bool = True, shaded_std: bool = True, redraw: bool = True):
    title = path[path.rfind("/") + 1:path.rfind("_")]

    if redraw:
        plt.ion()

    fig, axes = plt.subplots(1, 1, num=title)
    fig.canvas.set_window_title('Performance')
    plt.show(block=False)

    if redraw:
        date = -1

        while True:
            date_new = -1
            for root, dirs, files in os.walk(path):
                for name in files:
                    date_new = max(os.stat(os.path.join(root, name)).st_mtime, date_new)

            if date == date_new:
                _mypause(1)
                continue

            date = date_new

            plt.cla()
            axis = plt.gca()

            all_sub_dirs = next(os.walk(path))[1]

            for i in range(len(all_sub_dirs)):
                _plot_one(
                    axis,
                    os.path.join(path, all_sub_dirs[i]),
                    color=tuple([float(item) / 255. for item in _colors[i % len(_colors)].values()]),
                    shaded_err=shaded_err, shaded_std=shaded_std, print_out=False
                )

            axis.legend(all_sub_dirs)

            plt.title(path[path.rfind("/") + 1:path.rfind("_")])
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            fig.tight_layout()

            _mypause(1)
    else:
        axis = plt.gca()

        all_sub_dirs = next(os.walk(path))[1]

        for i in range(len(all_sub_dirs)):
            print("Plotting:", all_sub_dirs[i])
            _plot_one(
                axis,
                os.path.join(path, all_sub_dirs[i]),
                color=tuple([float(item) / 255. for item in _colors[i % len(_colors)].values()]),
                shaded_err=shaded_err, shaded_std=shaded_std
            )

        axis.legend(all_sub_dirs)

        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        fig.tight_layout()

        plt.show()
