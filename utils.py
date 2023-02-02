import os
import numpy as np
import time
import random
from collections import OrderedDict
import gc
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def replace(string_in, replace_from, replace_to='_'):
    if not isinstance(replace_from, list):
        replace_from = [replace_from]
    string_out = string_in
    for replace_entry in replace_from:
        string_out = string_out.replace(replace_entry, replace_to)
    return string_out


class BaseStat():
    """
    Basic statistic from which all other statistic types inherit
    """
    def __init__(self, name):
        self.name = name
        self.ep_idx = 0
        self.stat_collector = None

    def collect(self, value):
        pass

    def get_data(self):
        return {}

    def next_step(self):
        pass

    def next_ep(self):
        self.ep_idx += 1

    def next_batch(self):
        pass

    def compute_mean(self, mean, value, counter):
        return (counter * mean + value) / (counter + 1)

    def compute_ma(self, ma, value, ma_weight):
        return (1 - ma_weight) * ma + ma_weight * value


class AvgStat(BaseStat):
    """
    Standard average statistic (can track total means, moving averages,
    exponential moving averages etcetera)
    """
    def __init__(self, name, coll_freq='ep', ma_weight=0.1):
        super(AvgStat, self).__init__(name=name)
        self.counter = 0
        self.mean = 0.0
        self.ma = 0.0
        self.last = None
        self.means = []
        self.mas = []
        self.values = []
        self.times = []
        self.coll_freq = coll_freq
        self.ma_weight = ma_weight

    def collect(self, value, delta_counter=1):
        self.counter += delta_counter

        self.values.append(value)
        self.times.append(self.counter)
        self.mean = self.compute_mean(self.mean, value, len(self.means))
        self.means.append(self.mean)
        if self.counter < 10:
            # Want the ma to be more stable early on
            self.ma = self.mean
        else:
            self.ma = self.compute_ma(self.ma, value, self.ma_weight)
        self.mas.append(self.ma)
        self.last = value

    def get_data(self):
        return {'times': self.times, 'means': self.means, 'mas': self.mas, 'values': self.values}

    def print(self, timestamp=None):
        if self.counter <= 0:
            return
        self._print_helper()

    def _print_helper(self, mean=None, ma=None, last=None):

        # Set defaults
        if mean is None:
            mean = self.mean
        if ma is None:
            ma = self.ma
        if last is None:
            last = self.last

        if isinstance(mean, float):
            print('Mean %-35s tot: %10.5f, ma: %10.5f, last: %10.5f' %
                  (self.name, mean, ma, last))
        else:
            print('Mean %-35s tot:  (%.5f' % (self.name, mean[0]), end='')
            for i in range(1, mean.size - 1):
                print(', %.5f' % mean[i], end='')
            print(', %.5f)' % mean[-1])
            print('%-40s ma:   (%.5f' % ('', ma[0]), end='')
            for i in range(1, ma.size - 1):
                print(', %.5f' % ma[i], end='')
            print(', %.5f)' % ma[-1])
            print('%-40s last: (%.5f' % ('', last[0]), end='')
            for i in range(1, last.size - 1):
                print(', %.5f' % last[i], end='')
            print(', %.5f)' % last[-1])

    def save(self, save_dir):
        file_name = replace(self.name, [' ', '(', ')', '/'], '-')
        file_name = replace(file_name, ['<', '>'], '')
        file_name += '.npz'
        np.savez(os.path.join(save_dir, file_name),
                 values=np.asarray(self.values), means=np.asarray(self.means),
                 mas=np.asarray(self.mas), times=np.asarray(self.times))

    def plot(self, times=None, values=None, means=None, mas=None, save_dir=None):
        # Set defaults
        if times is None:
            times = self.times
        if values is None:
            values = self.values
        if means is None:
            means = self.means
        if mas is None:
            mas = self.mas
        if save_dir is None:
            save_dir_given = None
            save_dir = os.path.join(self.log_dir, 'stats', 'data')
        else:
            save_dir_given = save_dir

        # Define x-label
        if self.coll_freq == 'ep':
            xlabel = 'episode'
        elif self.coll_freq == 'step':
            xlabel = 'step'

        if np.asarray(values).ndim > 1:
            # Plot all values
            self._plot(times, values, self.name + ' all', xlabel, 'y', None,
                       save_dir_given)

            # Plot total means
            self._plot(times, means, self.name + ' total mean', xlabel, 'y', None,
                       save_dir_given)

            # Plot moving averages
            self._plot(times, mas, self.name + ' total exp ma', xlabel, 'y', None,
                       save_dir_given)
        else:
            self._plot_in_same(times, [values, means, mas],
                               self.name, xlabel, 'y',
                               ['all-data', 'mean', 'ma'],
                               [None, '-.', '-'], [0.25, 1.0, 1.0],
                               save_dir_given)

        # Also save current data to file
        if save_dir_given is None:
            file_name = replace(self.name, [' ', '(', ')', '/'], '-')
            file_name = replace(file_name, ['<', '>'], '')
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, file_name), values)

    def _plot(self, x, y, title='plot', xlabel='x', ylabel='y', legend=None,
              log_dir=None):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()

    def _plot_in_same(self, x, ys, title='plot', xlabel='x', ylabel='y',
                      legend=None, line_styles=None, alphas=None,
                      log_dir=None):
        if alphas is None:
            alphas = [1.0 for _ in range(len(ys))]
        plt.figure()
        for i in range(len(ys)):
            if line_styles[i] is not None:
                plt.plot(x, ys[i],
                         linestyle=line_styles[i], alpha=alphas[i])
            else:
                plt.plot(x, ys[i], 'yo', alpha=alphas[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()


class StatCollector():
    """
    Statistics collector class
    """
    def __init__(self, log_dir, tot_nbr_steps, print_iter):
        self.stats = OrderedDict()
        self.log_dir = log_dir
        self.ep_idx = 0
        self.step_idx = 0
        self.epoch_idx = 0
        self.print_iter = print_iter
        self.tot_nbr_steps = tot_nbr_steps

    def has_stat(self, name):
        return name in self.stats

    def register(self, name, stat_info):
        if self.has_stat(name):
            sys.exit("Stat already exists")

        if stat_info['type'] == 'avg':
            stat_obj = AvgStat(name, stat_info['freq'])
        else:
            sys.exit("Stat type not supported")

        stat = {'obj': stat_obj, 'name': name, 'type': stat_info['type']}
        self.stats[name] = stat

    def s(self, name):
        return self.stats[name]['obj']

    def next_step(self):
        self.step_idx += 1

    def next_ep(self):
        self.ep_idx += 1
        for stat_name, stat in self.stats.items():
            stat['obj'].next_ep()
        if self.ep_idx % self.print_iter == 0:
            self.print()
            self._plot_to_hdock()

    def print(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].print()

    def plot(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].plot(save_dir=self.log_dir)

    def save(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].save(save_dir=self.log_dir)

def get_random_crop(H, W, crop_size, coords=None, min_frame=0, H_MIN=0, W_MIN=0):
    coords_inside = np.zeros((0, 2))
    for attempt in range(250):
        start_h = random.randint(H_MIN, H - crop_size - 1)
        end_h = start_h + crop_size
        start_w = random.randint(W_MIN, W - crop_size - 1)
        end_w = start_w + crop_size
        if coords is None:
            return [start_h, end_h, start_w, end_w], None, None
        else:
            crop_found = False
            idxs_inside = []
            for i in range(coords.shape[0]):
                coord_h = coords[i, 0]
                coord_w = coords[i, 1]
                if start_h + min_frame <= coord_h and end_h - min_frame > coord_h+1 and start_w + min_frame <= coord_w and end_w - min_frame > coord_w+1:
                    crop_found = True
                    coords_inside = np.concatenate((coords_inside, coords[i, :][np.newaxis, :]), axis=0)
                    idxs_inside.append(i)
            if crop_found:
                return [start_h, end_h, start_w, end_w], coords_inside, np.array(idxs_inside)
    return None, None, None
