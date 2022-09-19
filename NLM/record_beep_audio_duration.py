'''
Records from the device mic for the specified duraton and saves it to specified path.
Needed to record the beep from the experiment to time sync the multiple devices together.
'''
import datetime
import argparse
import os

import sounddevice as sd
from scipy.io.wavfile import write

def start_record(args):
	if not os.path.exists(args.data_folder):
	    os.makedirs(args.data_folder)

	fs = 44100
	seconds = args.duration

	print('Started recording for a duration of {} seconds'.format(args.duration))
	myrecording = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
	start_timestamp = datetime.datetime.now()
	sd.wait()
	write(os.path.join(args.data_folder, '{}_{}.wav'.format(args.subject_id, args.run)), fs, myrecording)

	with open(os.path.join(args.data_folder, 'meta.txt'), 'a') as f:
	    f.write('{}_{}'.format(args.subject_id, args.run)+', '+str(start_timestamp)+'\n')

	print("Saving audio to {}".format(os.path.join(args.data_folder, '{}_{}.wav'.format(args.subject_id, args.run))))
	print("Saving timestamp to {}".format(os.path.join(args.data_folder, 'meta.txt')))