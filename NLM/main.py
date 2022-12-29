import argparse

import record_serial
import record_beep_audio_duration
import data_extraction
import visualization
import streamer

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='sub-command help', dest='command')
parser_a = subparsers.add_parser('record_data', help='record data from the openBCI')
parser_b = subparsers.add_parser('record_beep', help='record the calibration beep for a fixed duration (15s)')
parser_c = subparsers.add_parser('extract_data', help='extract the data collected into labelled csv')
parser_d = subparsers.add_parser('visualize', help='visualize the data collected')
parser_e = subparsers.add_parser('stream', help='stream the data collected')


parser_a.add_argument('--subject_id', type=str, required=True)
parser_a.add_argument('--task', type=str, required=True)
parser_a.add_argument('--run', type=int, default=1)

parser_b.add_argument('--data_folder', type=str, default='serial_data/calibration_audio')
parser_b.add_argument('--subject_id', type=str, required=True)
parser_b.add_argument('--run', type=int, default=1)
parser_b.add_argument('--duration', type=int, default=15)

parser_c.add_argument('--metadata', type=str, required=True)
parser_c.add_argument('--data_folder', type=str, required=True)
parser_c.add_argument('--analysis_folder', type=str, required=True)
parser_c.add_argument('--subject_id', type=str)
parser_c.add_argument('--surrounding', type=float, default=1, help="Surrounding phases of production in seconds")

parser_d.add_argument('--metadata', type=str, required=True)
parser_d.add_argument('--analysis_folder', type=str, required=True)
parser_d.add_argument('--subject_id', type=str)
parser_d.add_argument('--surrounding', type=float, default=1, help="Surrounding phases of production in seconds")

parser_e.add_argument('--metadata', type=str, required=True)
parser_e.add_argument('--data_folder', type=str, required=True)
parser_e.add_argument('--subject_id', type=str)
parser_e.add_argument('--task', type=str)

args = parser.parse_args()

if args.command == 'record_data':
	record_serial.start_record(args)
if args.command == 'record_beep':
	record_beep_audio_duration.start_record(args)
if args.command == 'extract_data':
	data_extraction.extract_data(args)
if args.command == 'visualize':
	visualization.visualize(args)
if args.command == 'stream':
	streamer.extract_data(args)
