'''
Records from the device mic when the `cmd` key is pressed and saves it to specified path.
Needed to record the beep from the experiment to time sync the multiple devices together.

brew install portaudio to succesfully install pyaudio (using pip)
'''

import argparse
import os
import queue
import datetime
import time

from pynput import keyboard
import pyaudio # brew install portaudio first 
import wave

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str)
parser.add_argument('--subject_id', type=str)
args = parser.parse_args()

if not os.path.exists(args.data_folder):
    os.makedirs(args.data_folder)

CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

p = pyaudio.PyAudio()
frames = []

class MyListener(keyboard.Listener):
    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.key_pressed = None

        self.stream = p.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=CHUNK,
                             stream_callback = self.callback)


    def on_press(self, key):
        if key == keyboard.Key.cmd_l:
            self.key_pressed = True

    def on_release(self, key):
        if key == keyboard.Key.cmd_l:
            self.key_pressed = False

    def callback(self, in_data, frame_count, time_info, status):
        if self.key_pressed == True:
            frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        elif self.key_pressed == False:
            frames.append(in_data)
            return (in_data, pyaudio.paComplete)

        else:
            return (in_data,pyaudio.paContinue)


listener = MyListener()
listener.start()
started = False

print("Press cmd to start recording, keep it held for whole duration..")
while True:
    time.sleep(0.05)
    if listener.key_pressed == True and started == False:
        started = True
        listener.stream.start_stream()
        start_timestamp = datetime.datetime.now()
        print ("Started Recording ...")

    elif listener.key_pressed == False and started == True:
        print ("Saving audio to {}".format(os.path.join(args.data_folder, '{}.wav'.format(args.subject_id))))
        print ("Saving timestamp to {}".format(os.path.join(args.data_folder, 'meta.txt')))
        listener.stream.stop_stream()
        listener.stream.close()
        p.terminate()

        with wave.open(os.path.join(args.data_folder, '{}.wav'.format(args.subject_id)), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        with open(os.path.join(args.data_folder, 'meta.txt'), 'a') as f:
            f.write(args.subject_id+', '+str(start_timestamp)+'\n')

        break
