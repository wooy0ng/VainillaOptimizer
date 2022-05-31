from mode import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=False, default='train')

args = parser.parse_args()

if args.mode == 'train':
    train(args)
    test(args)
elif args.mode == 'test':
    test(args)