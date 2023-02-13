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

opj = os.path.join

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

    calibration_missing_subjects = ['ASD005', 'ASD006']
    for s in calibration_missing_subjects:
        if s in meta_file:
            if s == 'ASD005':
                # time_delta_mac_surface = timedelta(hours=3, minutes=5, seconds=40, microseconds=382775) # a candidate
                time_delta_mac_surface = timedelta(hours=3, minutes=5, seconds=41, microseconds=382775) # a candidate
            if s == 'ASD006':
                # time_delta_mac_surface = timedelta(hours=3, minutes=6, seconds=20, microseconds=382775) # a candidate
                time_delta_mac_surface = timedelta(hours=3, minutes=6, seconds=20, microseconds=382775) # a candidate
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

    return experiment_start_time_stamp_mac



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
            start_times.append(start_time)
            end_times.append(end_time)
            durations.append(duration)
            print('-'*10)

    label_names, counts = np.unique(labels, return_counts=True)
    print(*zip(label_names, counts), sep="\n")

    print('Average duration : {:.3f} seconds'.format(np.mean(durations)))

    data = np.stack((labels, start_times, end_times, durations), axis=-1)

    return data, label_names


def extract_emg_annotations(csv_folder, emg_recordings_filename, annotation_textgrid, label_names, attempt, surrounding):
    
    with open(emg_recordings_filename, 'r') as f:
	    emg_recording = f.read().split('\n')


    for i, label_name in enumerate(label_names):
        print('Label ', label_name)
        csv_file = opj(csv_folder, "{}_{}.csv".format(label_name, attempt))
        emg_recording_label = emg_recording.copy()

        block_counts = []
        block_count = 0

        selection_started = False
        
        try:
            # open the file in the write mode
            with open(csv_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Channel_{}'.format(i) for i in range(1,9)])

                for d in tqdm(annotation_textgrid):
                    if d[0] == label_name:
                        row = label_name+"_{}".format(attempt)
                        channel_data = []
                        start_time = d[1].time()
                        end_time = d[2].time()
                        for j in range(6, len(emg_recording_label)):
                            time_stamp = emg_recording_label[j].split(', ')[-1]
                            if len(time_stamp) == 0:
                                continue
                            time_stamp = datetime.strptime(time_stamp, '%H:%M:%S.%f').time()
                            if not selection_started and time_stamp > start_time:
                                # found the first data point for the word
                                selection_started = True
                                # get the anticipation phase
                                channel_data = channel_data + [d.split(',')[1:9] for d in
                                    emg_recording_label[j-surrounding:j]]
                            
                            elif selection_started and time_stamp > start_time and time_stamp < end_time:
                                # selection for the word is ongoing

                                emg_recording_label[j] = emg_recording_label[j].replace(
                                    '0, 1, 1, 0, 1', '0, 1, 1, 1, 1')
                                block_count += 1
                                channel_data.append(emg_recording_label[j].split(',')[1:9])

                            elif selection_started and time_stamp > end_time:
                                # selection for the word is done
                                block_counts.append(block_count)
                                block_count = 0

                                # channel_data = channel_data + [d.split(',')[1:9] for d in
                                # emg_recording_label[j:j+surrounding]]

                                writer.writerows(channel_data)
                                break
                                
        except Exception as e:
            print(e)
            return   

        # validity checker 
        
        count = 0
        for a in range(len(emg_recording_label)-1):
            if '0, 1, 1, 1, 1' in emg_recording_label[a]:
                if '0, 1, 1, 0, 1' in emg_recording_label[a+1]:
                    count+=1

        print(count)
        print(block_counts)


def extract_data_per_task(task_metadata, emg_folder, annotation_folder, csv_folder, time_delta_mac_surface, experiment_start_times_surface, surrounding):

    task = task_metadata['name']
    emg_files = task_metadata['emg']
    text_grid_files = task_metadata['textgrids']
            
    csv_folder = opj(csv_folder, task)

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    for attempt in range(len(emg_files)):
        emg_file = opj(emg_folder, emg_files[attempt])
        text_grid_file = opj(annotation_folder, text_grid_files[attempt])
        attempt_str = str(attempt+1)

        experiment_start_time_stamp_mac = get_experiment_start_time_stamp_mac(
                            time_delta_mac_surface, 
                            experiment_start_times_surface,
                            task, 
                            attempt_str)

        annotation_textgrid, label_names = parse_annotation_textgrid(text_grid_file, experiment_start_time_stamp_mac)
        extract_emg_annotations(csv_folder, emg_file, annotation_textgrid, label_names, attempt_str, surrounding)


def extract_data(args):
    print("Extracting data for the subject", args.subject_id)

    metadata_file = open(args.metadata, 'r')
    metadata = json.load(metadata_file)['subjects'][args.subject_id]

    calibration_metadata = metadata['calibration']
    tasks_metadata = metadata['tasks']

    sample_rate = 250
    channels = range(0, 8)
    # surrounding = 0
    surrounding = int(sample_rate*args.surrounding)

    subject_folder = opj(args.data_folder, args.subject_id)
    emg_folder = opj(subject_folder, 'alterego', 'emg')
    calibration_folder = opj(subject_folder, 'alterego', 'calibration_audio')
    annotation_folder = opj(subject_folder, 'textgrids')

    if not os.path.exists(args.analysis_folder):
        os.makedirs(args.analysis_folder)

    csv_folder = opj(args.analysis_folder, args.subject_id, 'csv')

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
        extract_data_per_task(task_metadata, emg_folder, annotation_folder, csv_folder, time_delta_mac_surface, experiment_start_times_surface, surrounding)


