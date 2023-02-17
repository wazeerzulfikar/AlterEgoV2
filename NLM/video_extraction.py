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
import subprocess

opj = os.path.join

def extract_videos_per_task(task_metadata, input_video_folder, video_folder, annotation_folder, surrounding):

    task = task_metadata['name']
    emg_files = task_metadata['emg']
    text_grid_files = task_metadata['textgrids']
            
    video_folder = opj(video_folder, task)
    input_video_folder = '/Users/wazeer/mas/nlm/videos/CPPG-ASD-005/'

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    for attempt in range(len(emg_files)):
        text_grid_file = opj(annotation_folder, text_grid_files[attempt])
        attempt_str = str(attempt+1)

        video_task_folder = "{}_{}".format(task.capitalize(), attempt_str)
        actual_video_task_folder = [x[0] for x in os.walk(input_video_folder) if video_task_folder in x[0]][0]
        input_file = opj(video_folder, actual_video_task_folder, "ffmpeg_results_vid.mp4")

        tg = textgrid.TextGrid.fromFile(text_grid_file)
        print("------- IntervalTier Example -------")
        print(tg[0])
        print(len(tg[0]))

        labels = []
        start_times = []
        end_times = []
        durations = []

        for i in tg[0]:
            if len(i.mark.replace(' ', '')) <= 0:
                continue


            start_time = i.minTime
            end_time = i.maxTime
            label = i.mark
            minute = int(start_time/60)
            sec = max(int(start_time%60)-1, 0)
            millisec = int((start_time-int(start_time))*100)
            start_time_str = "00:{:02d}:{:02d}.{:02d}".format(minute, sec, millisec)

            minute = int(end_time/60)
            sec = int(end_time%60)+1
            millisec = int((end_time-int(end_time))*100) #buffer to finish
            end_time_str = "00:{:02d}:{:02d}.{:02d}".format(minute, sec, millisec)

            output_file = opj(video_folder, '{}_{}.mp4'.format(label, attempt_str))
            print(input_file)
            print(output_file)
            print(start_time_str)
            print(end_time_str)

            try:
                subprocess.check_call(["ffmpeg", "-y", "-i", 
                    input_file,
                    "-loglevel", "quiet",
                    "-ss", start_time_str, "-to", end_time_str, 
                    "-acodec",
                    "copy", output_file])
            except Exception as e:
                print("ERROR")
                print(e)
                continue
            print('-'*10)
            



def extract_videos(args):
    print("Extracting videos for the subject", args.subject_id)

    metadata_file = open(args.metadata, 'r')
    metadata = json.load(metadata_file)['subjects'][args.subject_id]

    calibration_metadata = metadata['calibration']
    tasks_metadata = metadata['tasks']

    sample_rate = 250
    channels = range(0, 8)
    # surrounding = 0
    surrounding = int(sample_rate*args.surrounding)

    subject_folder = opj(args.data_folder, args.subject_id)
    annotation_folder = opj(subject_folder, 'textgrids')

    if not os.path.exists(args.analysis_folder):
        os.makedirs(args.analysis_folder)

    video_folder = opj(args.analysis_folder, args.subject_id, 'videos')

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    for task_metadata in tasks_metadata:
        input_video_folder = ""
        extract_videos_per_task(task_metadata, input_video_folder, video_folder, annotation_folder, surrounding)


