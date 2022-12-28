import os
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from display_utils import DynamicConsoleTable
import math
import time
import os.path
import glob
import textgrid
import datetime
from PIL import Image
from tqdm import tqdm
import csv
import json

import transforms

opj = os.path.join

#### Preprocessing
def transform_data(sequence_groups, sample_rate=250):

    #### Apply DC offset and drift correction
    drift_low_freq = 0.5 #0.5
    sequence_groups = transforms.subtract_initial(sequence_groups)
    sequence_groups = transforms.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    # sequence_groups = transforms.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq, for the power line noise
    notch_freq = 60
    num_times = 3 #pretty much just the filter order
    freqs = np.round(np.arange(1, sample_rate/(2. * notch_freq)) * notch_freq).astype(np.int32)
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = transforms.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    # sequence_groups = transforms.normalize_std(sequence_groups)

    def normalize_kernel(kernel, subtract_mean=False):
        if subtract_mean:
            kernel = np.array(kernel, np.float32) - np.mean(kernel)
        return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
    def ricker_function(t, sigma):
        return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
    def ricker_wavelet(n, sigma):
        f = lambda x: ricker_function(x, sigma)
        return np.array([f(i) for i in range(-n//2, n//2+1)])

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = transforms.correlate(sequence_groups, ricker_kernel)
    ricker_subtraction_multiplier = 2.0

    sequence_groups = np.array(sequence_groups) - np.array(ricker_subtraction_multiplier) * ricker_convolved

    low_freq = 0.5 #0.5
    high_freq = 4 #8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = transforms.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)
    return sequence_groups


def visualize_per_task(subject_id, task, csv_folder, plot_folder, sample_rate, surrounding):
    channels = range(0, 8)
    
    files = glob.glob(opj(csv_folder, '*.csv'))
    files.sort()

    words = [i.split('/')[-1].replace('.csv', '') for i in files]
    words = [w.split('_')[0] for w in words]
    words = set(words)

    electrode_placement_img = Image.open('electrode_placement.png')

    for w in words:
        print(w)
        word_files = [f for f in files if w == os.path.basename(f).split('_')[0]]
        all_instances = []
        for f in word_files:
            header_skipped = False
            with open(f, 'r') as file:
                csvreader = csv.reader(file)
                data = []
                for row in csvreader:
                    if header_skipped == False:
                        header_skipped = True
                        continue
                    float_row = [float(i.replace(' ', '')) for i in row]
                    data.append(float_row)
                    
            if len(data) > 0:
                all_instances.append(data)

        fig, ax = plt.subplots(2, 2)
        fig.figimage(electrode_placement_img, 2200, 1500)
        suptitle = "Subject: "+subject_id+", Task: "+task+", Word: "+w
        plt.suptitle(suptitle)

        # Set common labels
        fig.text(0.5, 0.03, 'Sample', ha='center', va='center')
        fig.text(0.03, 0.5, u'Signal (\u03bcV)', ha='center', va='center', rotation='vertical')

        for i in range(len(all_instances)): # word instances

            sequence = np.array(all_instances[i]).T # change to (channels, values)
            sequence_len = len(sequence)
            sequence = transform_data(sequence, sample_rate)

            x = i//2
            y = i%2
            colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']
            for j in range(len(channels)): # channel
                ax[x, y].plot(sequence[j], c=colors[channels[j]], alpha=1.0, lw=1.0, label=str(j))

            x_lims = ax[x, y].get_xlim()
            ax[x, y].axvline(x=x_lims[0]+surrounding, linestyle=':', c='black')
            # ax[x, y].axvline(x=x_lims[1]-surrounding, linestyle=':', c='black')

            ax[x, y].set_title('Anticipatory phase (1s) : Spoken phase', size=6)
            ax[x, y].tick_params(axis='both', labelsize=6)

        if len(all_instances)<2:
            ax[0, 1].text(0.3, 0.5, 'No Signal', fontsize=15)
        if len(all_instances)<3:
            ax[1, 0].text(0.3, 0.5, 'No Signal', fontsize=15)
        if len(all_instances)<4:
            ax[1, 1].text(0.3, 0.5, 'No Signal', fontsize=15)

            # plt.legend(loc='upper right')
        plt.savefig(opj(plot_folder, '{}.png'.format(w)), dpi=400)
        plt.clf()


def visualize(args):
    sample_rate = 250
    # surrounding = 0
    surrounding = int(sample_rate*args.surrounding)

    metadata_file = open(args.metadata, 'r')
    metadata = json.load(metadata_file)['subjects'][args.subject_id]
    tasks_metadata = metadata['tasks']

    for task in tasks_metadata:
        task_name = task['name']
        csv_folder = opj(args.analysis_folder, args.subject_id, 'csv', task_name)
        plot_folder = opj(args.analysis_folder, args.subject_id, 'plots', task_name)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        print(task_name)
        visualize_per_task(args.subject_id, task_name, csv_folder, plot_folder, sample_rate, surrounding)
        print('-'*10)