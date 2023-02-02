import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Global vars
LOG_PATH_DATE = ''  # Set to the path of the model to plot result curves for
MA_SMOOTH = 1*.6 * 0.00025 # Note: If MA_TYPE = 'win', then window size is 1 / MA_SMOOTH
MA_TYPE = 'exp'  # 'exp' (exponential) or 'win' (fix-sized window).
STEP_SIZE_DISPLAY = 25
SCALE_FACTOR = 1
X_AXIS_END = None
MIN_Y = 0
USE_SQRT = False  # True --> sqrt(MSE) shown instead of MSE
if 0:
    STAT_NAME = 'CE_loss'
    MAX_Y = 0.1
elif 0:
    STAT_NAME = 'mIoU'
    MAX_Y = 1.0
elif 1:
    STAT_NAME = ['mIoU', 'IoU_0', 'IoU_1']
    #STAT_NAME = ['Recall', 'Recall_0', 'Recall_1', 'Recall_2', 'Recall_3', 'Recall_4', 'Recall_5']
    #STAT_NAME = ['Precision', 'Precision_0', 'Precision_1', 'Precision_2', 'Precision_3', 'Precision_4', 'Precision_5']
    #STAT_NAME = ['mIoU', 'IoU_0', 'IoU_1', 'IoU_2', 'IoU_3', 'IoU_4', 'IoU_5']
    MAX_Y = 1.0
def _custom_ma(data, ma_smooth=MA_SMOOTH):
    for idx, val in enumerate(data['values']):
        if idx < 25:
            data['mas_custom'][idx] = data['means'][idx]
        else:
            # Filter out any possible nan-entries in the data
            i = 0
            while np.isnan(data['values'][idx - i]):
                i += 1
            data['values'][idx] = data['values'][idx - i]
            if MA_TYPE == 'exp':
                data['mas_custom'][idx] = (1 - ma_smooth) * data['mas_custom'][idx - 1] + ma_smooth * data['values'][idx]
            else:
                # Fix-sized moving average window
                win_size = min(idx, int(round(1 / MA_SMOOTH)))
                data['mas_custom'][idx] = np.mean(data['values'][idx - win_size : idx + 1])

def _plot(datas, title='plot', xlabel='# training batches', ylabel=STAT_NAME,
          start_it=0, max_x=None, max_y=None, min_y=None, force_axis=False, fig=None):
    if fig is None:
        show_plot = False
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
    else:
        show_plot = True
        ax = fig.add_subplot(1, 2, 2)
    show_plot = False
    max_nbr_data = 0
    for data in datas:
        max_nbr_data = max(max_nbr_data, len(data[0]['mas_custom']))
    for data in datas:
        x = data[0]['times']
        y = SCALE_FACTOR*data[0]['mas_custom']
        if USE_SQRT:
            y = np.sqrt(y)
        x = range(0,max_nbr_data)
        y = np.concatenate([y, np.nan * np.ones(max_nbr_data - len(y))])
        plt.plot(x[start_it:], y[start_it:])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(STAT_NAME)
    ax = plt.gca()
    if USE_SQRT:
        max_y = np.sqrt(max_y)
        min_y = np.sqrt(min_y)
    if max_x is None:
        max_x = x[-1]
    if force_axis:
        ax.set_xlim([0, max_x])
        if min_y is not None and max_y is not None:
            ax.set_ylim([min_y, max_y])
        elif min_y is not None:
            ax.set_ylim([min_y, 100])
        elif max_y is not None:
            ax.set_ylim([0, max_y])
    else:
        ax.set_xlim([0, min(max_x, x[-1])])
        if max_y is not None:
            ax.set_ylim([0, min(max_y, max(np.max(y['means'][start_it:]), np.max(y['mas'][start_it:])))])
    ax.set_aspect(max_x / (max_y - min_y))
    if show_plot:
        plt.show()
    return fig

# Read data from log path
if not isinstance(LOG_PATH_DATE, list):
    LOG_PATH_DATE = [LOG_PATH_DATE]
if not isinstance(STAT_NAME, list):
    STAT_NAME = [STAT_NAME]
L2_losses_all = []
L2_losses_all_val = []
for stat_name in STAT_NAME:
    L2_losses = {'means': 0, 'mas': 0, 'values': 0, 'times': 0}
    L2_losses_val = {'means': 0, 'mas': 0, 'values': 0, 'times': 0}
    min_x_range = np.inf
    min_x_range_val = np.inf
    for log_path_date in LOG_PATH_DATE:
        log_path = os.path.join('../log', log_path_date, 'train_stats', stat_name + '.npz')
        tmp = np.load(log_path)
        for key in L2_losses:
            min_x_range = min(min_x_range, len(tmp[key]))
            if not isinstance(L2_losses[key], int) and len(L2_losses[key]) > min_x_range:
                L2_losses[key] = L2_losses[key][:min_x_range]
            L2_losses[key] += tmp[key][:min_x_range] / len(LOG_PATH_DATE)
        log_path = os.path.join('../log', log_path_date, 'train_stats', stat_name + '_val.npz')
        tmp = np.load(log_path)
        for key in L2_losses_val:
            min_x_range_val = min(min_x_range_val, len(tmp[key]))
            if not isinstance(L2_losses_val[key], int) and len(L2_losses_val[key]) > min_x_range_val:
                L2_losses_val[key] = L2_losses_val[key][:min_x_range_val]
            if key == 'times':
                L2_losses_val[key] += np.array([STEP_SIZE_DISPLAY * vv for vv in tmp[key]]) / len(LOG_PATH_DATE)
            else:
                L2_losses_val[key] += tmp[key][:min_x_range_val] / len(LOG_PATH_DATE)

    # Create MA-smoothing of raw data
    L2_losses['mas_custom'] = np.zeros_like(L2_losses['mas'])
    L2_losses_val['mas_custom'] = np.zeros_like(L2_losses_val['mas'])
    _custom_ma(L2_losses)
    _custom_ma(L2_losses_val, ma_smooth=10*MA_SMOOTH)

    # Append to list of all losses
    L2_losses_all.append([L2_losses])
    L2_losses_all_val.append([L2_losses_val])

# Plot results
fig_out = _plot(L2_losses_all, max_y=MAX_Y, min_y=MIN_Y, force_axis=True)
_plot(L2_losses_all_val, max_y=MAX_Y, min_y=MIN_Y, force_axis=True, fig=fig_out)
fig_out.savefig('result_plot.png')
plt.cla()
plt.clf()
plt.close('all')
print("Saved result plot!")
