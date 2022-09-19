import argparse

import record_serial
import record_beep_audio_duration

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='sub-command help', dest='command')
parser_a = subparsers.add_parser('record_data', help='record_data')
parser_b = subparsers.add_parser('record_beep', help='record_beep')


parser_a.add_argument('--subject_id', type=str, required=True)
parser_a.add_argument('--task', type=str, required=True)
parser_a.add_argument('--run', type=int, default=1)

parser_b.add_argument('--data_folder', type=str, default='serial_data/calibration_audio')
parser_b.add_argument('--subject_id', type=str, required=True)
parser_b.add_argument('--run', type=int, default=1)
parser_b.add_argument('--duration', type=int, default=30)


args = parser.parse_args()

if args.command == 'record_data':
	record_serial.start_record(args)
if args.command == 'record_beep':
	record_beep_audio_duration.start_record(args)
