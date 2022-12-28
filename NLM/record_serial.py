import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os

# local packages
import data_collection
import transforms

# channels = [1, 3, 4, 6] # Must be same as trained model if test_model==True
#channels = [1, 3, 4] # DO NOT CHANGE
channels = range(0,8)

def transform_data(sequence_groups, sample_rate=250):
    #### Apply DC offset and drift correction
    drift_low_freq = 0.5 #0.5
    sequence_groups = transforms.subtract_initial(sequence_groups)
    sequence_groups = transforms.highpass_filter(sequence_groups, drift_low_freq, sample_rate)
    sequence_groups = transforms.subtract_mean(sequence_groups)

    #### Apply notch filters at multiples of notch_freq
    notch_freq = 60
    num_times = 3 #pretty much just the filter order
    freqs = [int(round(x * notch_freq)) for x in np.arange(1, sample_rate/(2. * notch_freq))]
    for _ in range(num_times):
        for f in reversed(freqs):
            sequence_groups = transforms.notch_filter(sequence_groups, f, sample_rate)

    #### Apply standard deviation normalization
    #sequence_groups = transforms.normalize_std(sequence_groups)

    def normalize_kernel(kernel, subtract_mean=False):
        if subtract_mean:
            kernel = np.array(kernel, np.float32) - np.mean(kernel)
        return np.array(kernel, np.float32) / np.sum(np.abs(kernel))
    def ricker_function(t, sigma):
        return 2./(np.sqrt(3*sigma)*np.pi**0.25)*(1.-(float(t)/sigma)**2)*np.exp(-(float(t)**2)/(2*sigma**2))
    def ricker_wavelet(n, sigma):
        return np.array([ricker_function(x, sigma) for x in range(-n//2, n//2+1)])

    #### Apply ricker wavelet subtraction
    ricker_width = 35 * sample_rate // 250
    ricker_sigma = 4.0 * sample_rate / 250
    ricker_kernel = normalize_kernel(ricker_wavelet(ricker_width, ricker_sigma))
    ricker_convolved = transforms.correlate(sequence_groups, ricker_kernel)
    ricker_subtraction_multiplier = 2.0
    sequence_groups = sequence_groups - ricker_subtraction_multiplier * ricker_convolved

    #### Apply sine wavelet kernel
#    period = int(sample_rate)
#    sin_kernel = normalize_kernel(np.sin(np.arange(period)/float(period) * 1*np.pi), subtract_mean=True)
#    sequence_groups = transforms.correlate(sequence_groups, sin_kernel)

    low_freq = 0.5 #0.5
    high_freq = 8 #8
    order = 1

    #### Apply soft bandpassing
    sequence_groups = transforms.bandpass_filter(sequence_groups, low_freq, high_freq, sample_rate, order=order)

    #### Apply hard bandpassing
#    sequence_groups = transforms.fft(sequence_groups)
#    sequence_groups = transforms.fft_frequency_cutoff(sequence_groups, low_freq, high_freq, sample_rate)
#    sequence_groups = np.real(transforms.ifft(sequence_groups))
    sequence_groups = np.ones_like(sequence_groups)

    return sequence_groups

#February Data Collection word_map of 25 phrases to control an AI assistant
# word_map = ['turn up the heat by two', 'increase the volume of the music', 'change the lights to blue', 'Date and time please', 'unlock the front door', 'set a five minute timer', 'schedule a doctor\'s appointment', 'send mom a text message', 'restart the home wifi', 'What is the weather outside', 'tell me the headline news stories', 'set an alarm for twelve o\'clock', 'cancel tomorrow\'s meeting', 'give me a motivational quote', 'multiply eighty four and six', 'show me pictures of dogs', 'fast forward the show', 'play the next episode', 'search for concert tickets', 'navigate to the convenience store', 'remind me to call brother', 'create a calendar event', 'Hang up the incoming call', 'listen to my voicemail', 'Open my top playlist', 'Finished']

# word_map = ['baby', 'book', 'cup', 'dog', 'ten', 'pig', 'table', 'apple', 'paper', 'potato', 'chocolate', 'hi', 'bye', 'bottle', 'ball', 'kitten', 'duck', 'girl', 'Finished']
word_map = ['baby', 'book', 'cup', 'dog', 'ten', 'pig', 'table', 'apple', 'paper', 'potato', 'chocolate', 'hi', 'bye', 'bottle', 'ball', 'kitten', 'duck', 'girl']


labels = np.arange(len(word_map))
np.random.shuffle(labels)
labels = list(labels) #Change array before each time executing program
labels = labels + [-1]

recorded_length = None
last_recorded_count = -1
def on_data(history, trigger_history, index_history, count, samples_per_update, recorded_count):
    global last_recorded_count
    global recorded_length
    # if recorded_count > last_recorded_count:
    #     os.system('say "' + word_map[labels[recorded_count]] + '" &')
    # last_recorded_count = recorded_count
    # print
    # print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i+1) for i in range(8)])
    # print str('{:.1f}'.format(count/250.)) + 's\t\t' + '\t'.join(
    #     map(lambda (i, x): '{:f}'.format(x) if i in channels else '--\t', enumerate(history[-1])))
    # print
    # if recorded_count > 0:
    #     start, end = None, None
    #     for i in range(len(trigger_history))[::-1]:
    #         if trigger_history[i] and end is None:
    #             end = i
    #         elif not trigger_history[i] and end:
    #             start = i
    #             break
    #     if start and end:
    #         recorded_length = end - start
    #     # print 'WPM:', 60.0 / (float(recorded_length) / 250 / len(word_map[labels[recorded_count-(1 if end < len(trigger_history)-1 else 0)]].split(' ')))
    #     # print 'WPM:', 60.0 / (float(recorded_length) / 250)
    # print
    # print 'Sample #' + str(recorded_count+1)+'/'+str(len(labels)-1), '\tNext:', word_map[labels[recorded_count]]
    # print

# def on_data(history, trigger_history, index_history, count, samples_per_update, recorded_count):
#     print 'SPU: ' + str(samples_per_update) + '\t\t' + '\t'.join(['Channel ' + str(i+1) for i in range(8)])
#     print str('{:.1f}'.format(count/250.)) + 's\t\t' + '\t'.join(
#         map(lambda (i, x): '{:f}'.format(x) if i in channels else '--\t', enumerate(history[-1])))
#     print
#     print 'Sample #' + str(recorded_count + 1) + '/' + str(len(labels) - 1), '\t', word_map[labels[recorded_count]]
#     print

# data.serial.start('/dev/tty.usbserial-DM01HUN9', #change serial number if device not connecting
#                   on_data, channels=channels, transform_fn=transform_data,
#                   history_size=1500, shown_size=1200, override_step=40)#35

def start_record(args):
    # /dev/tty.usbserial-DM01HQ99 for the openbci (no button)
    # /dev/tty.usbserial-DM00Q882 for the openbci (lurie center)
    data_collection.start('/dev/tty.usbserial-DM00Q882', #change serial number if No such file or directory: '/dev/tty.usbserial-*
                      on_data, channels=channels, transform_fn=transform_data,
                      history_size=1500, shown_size=1200, override_step=40,
                      subject_id=args.subject_id, task_name=args.task, run_number=args.run,
                      plot=True)#35
