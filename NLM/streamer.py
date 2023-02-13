import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import glob
import textgrid
from datetime import datetime, timedelta
import csv
from tqdm import tqdm
import json
from PIL import Image
import streamlit as st
import pandas as pd
import playsound
from scipy.io import wavfile

import transforms

opj = os.path.join

def transform_data(sequence_groups, sample_rate=250, initial_values=None):

    #### Apply DC offset and drift correction
    drift_low_freq = 0.5 #0.5
    if initial_values is None:
        sequence_groups = transforms.subtract_initial(sequence_groups)
    else:
        sequence_groups = sequence_groups - initial_values

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


def parse_beep_textgrid(beep_text_grid_file_mac, 
                        beep_text_grid_file_surface, 
                        meta_file,
                        audio_name,
                        experiment_start_times_surface,
                        surface_calibration_task='DDK 1'
                        ):
    # Check difference in time using beep time sync
    # initialized because of UnboundLocalError from python interpreter
    initial_time_stamp_surface = datetime.now()
    initial_time_stamp_mac = datetime.now()

    calibration_missing_subjects = ['ASD005', 'ASD006', 'ASD004']
    for s in calibration_missing_subjects:
        if s in meta_file:
            if s == 'ASD005':
                time_delta_mac_surface = timedelta(hours=3, minutes=5, seconds=40, microseconds=382775)
            if s == 'ASD006':
                time_delta_mac_surface = timedelta(hours=3, minutes=6, seconds=20, microseconds=382775)
            if s in ['ASD004']:
                time_delta_mac_surface = timedelta(hours=3, minutes=15, seconds=20, microseconds=382775)
            print("Using default time difference:",time_delta_mac_surface)
            return time_delta_mac_surface

    with open(experiment_start_times_surface, 'r') as f:
        content = f.read().split('\n')
        for c in content:
            if surface_calibration_task+' Start' in c:
                initial_time_stamp_surface = c.split(': ')[-1]
                initial_time_stamp_surface =  datetime.strptime(initial_time_stamp_surface, '%m%d%Y %H:%M:%S.%f')

    tg_beep_surface = textgrid.TextGrid.fromFile(beep_text_grid_file_surface)
    beep_time_stamp_surface = initial_time_stamp_surface + timedelta(seconds=tg_beep_surface[0][0].time)

    with open(meta_file, 'r') as f:
        content = f.read().split('\n')
        for c in content:
            if audio_name in c:
                initial_time_stamp_mac = c.split(', ')[-1]
                initial_time_stamp_mac = datetime.strptime(initial_time_stamp_mac, '%Y-%m-%d %H:%M:%S.%f')

    tg_beep_mac = textgrid.TextGrid.fromFile(beep_text_grid_file_mac)
    beep_time_stamp_mac = initial_time_stamp_mac + timedelta(seconds=tg_beep_mac[0][0].time)

    time_delta_mac_surface = beep_time_stamp_mac - beep_time_stamp_surface

    print('Beep time Surface : ', beep_time_stamp_surface)
    print('Beep time Mac : ', beep_time_stamp_mac)
    print('Difference: ', time_delta_mac_surface)

    return time_delta_mac_surface


def get_experiment_start_time_stamp_mac(time_delta_mac_surface, 
                        experiment_start_times_surface,
                        task,
                        attempt):
        # Check difference in time using beep time sync

    # initialized because of UnboundLocalError from python interpreter
    full_task = task+' '+attempt
    experiment_start_time_stamp_surface = datetime.now()

    with open(experiment_start_times_surface, 'r') as f:
        content = f.read().split('\n')
        for c in content:
            if full_task in c.lower() and 'start' in c.lower():
                experiment_start_time_stamp_surface = c.split(': ')[-1]
                experiment_start_time_stamp_surface = datetime.strptime(experiment_start_time_stamp_surface, '%m%d%Y %H:%M:%S.%f')

    experiment_start_time_stamp_mac = experiment_start_time_stamp_surface + time_delta_mac_surface
    print('Experiment start Surface : ', experiment_start_time_stamp_surface)
    print('Experiment start Mac : ', experiment_start_time_stamp_mac)

    return experiment_start_time_stamp_mac, experiment_start_time_stamp_surface



def parse_annotation_textgrid(textgrid_path, experiment_start_time_stamp_mac):

    # Read a IntervalTier object.
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    print("------- IntervalTier Example -------")
    print(tg[0])
    print(len(tg[0]))

    labels = []
    start_times = []
    end_times = []
    durations = []

    for i in tg[0]:
        timedelta_ = timedelta(seconds=i.minTime)
        start_time = experiment_start_time_stamp_mac + timedelta_

        timedelta_ = timedelta(seconds=i.maxTime)
        end_time = experiment_start_time_stamp_mac + timedelta_

        duration = (end_time - start_time).total_seconds()

        if len(i.mark.replace(' ', '')) > 0:
            print('Label : ', i.mark)
            print('Start time : ', i.minTime)
            print('End time : ', i.maxTime)
            print('Duration :', duration)

            labels.append(i.mark)
            start_times.append(i.minTime)
            end_times.append(i.maxTime)
            durations.append(duration)
            print('-'*10)

    label_names, counts = np.unique(labels, return_counts=True)
    print(*zip(label_names, counts), sep="\n")

    print('Average duration : {:.3f} seconds'.format(np.mean(durations)))

    data = np.stack((labels, start_times, end_times, durations), axis=-1)

    return data, label_names


def stream_data(subject_id, task, emg_recordings_filename, experiment_start_time_stamp_mac, experiment_start_time_stamp_windows, annotation_textgrid, label_names, attempt):
    
    with open(emg_recordings_filename, 'r') as f:
	    emg_recording = f.read().split('\n')

    electrode_placement_img = Image.open('electrode_placement.png').resize((200, 100))
    colors = ['gray', 'purple', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']

    time_window = 125
    steps = 25
    sample_rate = 250

    # # stream data
    # plt.ion()

    # start_time = datetime.strptime(emg_recording[7].split(', ')[-1],  '%H:%M:%S.%f')

    # change to default date in python for comparison
    start_time_with_date = experiment_start_time_stamp_mac.strftime("%m/%d/%Y, %H:%M:%S")
    start_time = datetime.strptime(start_time_with_date.split(',')[-1].strip(), '%H:%M:%S')

    st.image(electrode_placement_img, caption='Electrode Reference')

    # fig, ax = plt.subplots(2, figsize=(8, 6))
    # fig.figimage(electrode_placement_img, 900, 700)

    audio_file = '/Users/wazeer/mas/nlm/data/{}/audio/naming_1/audio_data.wav'.format(subject_id)
    # audio_file = '/Users/wazeer/Dropbox (MIT)/CPPG & SFARI data (1)/CPPG-ASD-004/202210210807_MVData_Naming_1_subj_CPPG-ASD-004/audio_data.wav'
    # playsound.playsound(audio_file, False)
    audio_samplerate, audio_data = wavfile.read(audio_file)

    # get normalization values using the initial values as reference. Averaging over initial 10 samples for robustness
    # initial_values = np.expand_dims(np.array([float(i) for i in emg_recording[7].split(', ')[1:9]]), axis=-1)
    initial_values = [emg_recording[k].split(', ')[1:9] for k in range(7, 17)]
    initial_values = [list(map(float, sample)) for sample in initial_values]
    initial_values = np.expand_dims(np.mean(np.array(initial_values), axis=0), axis=-1)

    processed_channel_data = [emg_recording[k].split(',')[1:9] for k in range(7,len(emg_recording)-1)]
    processed_channel_data = np.array(processed_channel_data).T.astype(np.float32)
    processed_channel_data = transform_data(processed_channel_data, sample_rate, initial_values)
    processed_channel_data = np.array(processed_channel_data)

    min_val, max_val = np.percentile(processed_channel_data, 0.5), np.percentile(processed_channel_data, 99.5)
    audio_min_val, audio_max_val = np.percentile(audio_data, 0.5), np.percentile(audio_data, 99.5)

    processed_time_stamps = [emg_recording[k].split(', ')[-1] for k in range(7, len(emg_recording)-1)]
    processed_time_stamps = [i.split(' ')[-1] for i in processed_time_stamps]
    # processed_time_stamps =[datetime.strptime(time_stamp, '%H:%M:%S.%f') for time_stamp in processed_time_stamps]
    processed_time_stamps =[datetime.strptime(time_stamp, '%H:%M:%S.%f') for time_stamp in processed_time_stamps]
    processed_time_stamps = [(time_stamp-start_time).total_seconds() for time_stamp in processed_time_stamps]

    # print(annotation_textgrid)
    # fig, ax = plt.subplots(2, figsize=(40, 20))

    # # plt.figure(figsize=(40, 20), dpi=300)
    # ax[0].plot(audio_data[::44100//250])

    # for j in range(8):
    #     ax[1].plot(processed_channel_data[j, 7:len(processed_channel_data[j])], c=colors[j], alpha=1.0, lw=1.0, label=str(j))
    # ax[1].set_ylim([-400, 400])
    # ax[0].set_ylim([-2000, 2000])
    # # plt.show()
    # plt.title("Subject: "+subject_id+", Task: "+task+", Attempt: "+attempt)
    # plt.ylabel("Signal")
    # plt.xlabel("Seconds")
    # plt.xticks(range(0, len(processed_channel_data[0]), 1000),
    #     [i/250 for i in range(0, len(processed_channel_data[0]), 1000)])
    # plt.savefig('emg.png')
    # print("Saved")
    # exit()

    for i in range(6+time_window, len(emg_recording)-time_window, steps):

        # datapoint_timestamp = datetime.strptime(emg_recording[i].split(', ')[-1], '%H:%M:%S.%f').time()
        # if datapoint_timestamp < experiment_start_time_stamp_mac.time():
        #     audio_idx = time_window
        #     continue

        channel_data = processed_channel_data[:, i-time_window:i+time_window]
        time_stamps = processed_time_stamps[i-time_window:i+time_window]
        if time_stamps[0] <0:
            continue

        approx_timestamps = np.linspace(
            time_stamps[0],
            time_stamps[-1],
            2*time_window) # for graph smoothing

        # Set common labels
        suptitle = "Subject: "+subject_id+", Task: "+task+", Attempt: "+attempt
        fig.suptitle(suptitle)
        fig.text(0.5, 0.03, 'Time since start (s)', ha='center', va='center')
        fig.text(0.03, 0.5, u'Signal (\u03bcV)', ha='center', va='center', rotation='vertical')

        for j in range(len(channel_data)): # channel
            ax[0].plot(approx_timestamps, channel_data[j], c=colors[j], alpha=1.0, lw=1.0, label=str(j))

        # plot the audio data
        audio_time_window_start = int(audio_samplerate/sample_rate*(i-time_window))
        audio_time_window_end = int(audio_samplerate/sample_rate*(i+time_window))
        audio_samples =  audio_data[audio_time_window_start:audio_time_window_end]
        ax[1].plot(audio_samples)
        # audio_idx += steps


        ax[0].set_ylim([min_val, max_val])
        ax[1].set_ylim([audio_min_val, audio_max_val])

        ax[0].set_title("EMG data")
        ax[1].set_title("Audio data")
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
        # time.sleep(steps/sample_rate)
        ax[0].cla()
        ax[1].cla()

def extract_data_per_task(subject_id, task_metadata, emg_folder, annotation_folder,  time_delta_mac_surface, experiment_start_times_surface):

    task = task_metadata['name']
    emg_files = task_metadata['emg']
    text_grid_files = task_metadata['textgrids']
            
    for attempt in range(len(emg_files)):
        emg_file = opj(emg_folder, emg_files[attempt])
        text_grid_file = opj(annotation_folder, text_grid_files[attempt])
        attempt_str = str(attempt+1)

        experiment_start_time_stamp_mac, experiment_start_time_stamp_windows = get_experiment_start_time_stamp_mac(
                            time_delta_mac_surface, 
                            experiment_start_times_surface,
                            task, 
                            attempt_str)

        annotation_textgrid, label_names = parse_annotation_textgrid(text_grid_file, experiment_start_time_stamp_mac)
        stream_data(subject_id, task, emg_file, experiment_start_time_stamp_mac, experiment_start_time_stamp_windows,
                annotation_textgrid, label_names, attempt_str)


def stream(args):
    print("Extracting data for the subject", args.subject_id)

    metadata_file = open(args.metadata, 'r')
    metadata = json.load(metadata_file)['subjects'][args.subject_id]

    calibration_metadata = metadata['calibration']
    tasks_metadata = metadata['tasks']

    sample_rate = 250
    channels = range(0, 8)
    surrounding = 0
    # surrounding = int(sample_rate*args.surrounding)

    subject_folder = opj(args.data_folder, args.subject_id)
    emg_folder = opj(subject_folder, 'alterego', 'emg')
    calibration_folder = opj(subject_folder, 'alterego', 'calibration_audio')
    annotation_folder = opj(subject_folder, 'textgrids')


    beep_text_grid_file_mac = opj(annotation_folder, calibration_metadata['beep_text_grid_file_mac'])
    beep_text_grid_file_surface =  opj(annotation_folder, calibration_metadata['beep_text_grid_file_surface'])
    meta_file = opj(calibration_folder, calibration_metadata['meta_file'])
    experiment_start_times_surface = opj(subject_folder, calibration_metadata['experiment_start_times_surface'])

    time_delta_mac_surface = parse_beep_textgrid(
                    beep_text_grid_file_mac, 
                    beep_text_grid_file_surface, 
                    meta_file,
                    calibration_metadata['audio_name'],
                    experiment_start_times_surface)

    for task_metadata in tasks_metadata:
        if task_metadata['name'] == args.task:
            extract_data_per_task(args.subject_id, task_metadata, emg_folder, annotation_folder, time_delta_mac_surface, experiment_start_times_surface)


