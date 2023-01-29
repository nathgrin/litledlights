#!/usr/bin/env python3
import argparse
import sys

import board
import neopixel

import time

from utils import get_strip,clear



def run_off(args):
    print('Fuck you, shit is lit')


def run_clear(args):
    strip = get_strip()
    strip.fill( (0,0,0) )
    strip.show()


def run_tst(args):
    print("There is no test :(")


def main(argv):
    parser = argparse.ArgumentParser(prog="lll")
    subparsers = parser.add_subparsers()

    off_parser = subparsers.add_parser('off')
    off_parser.set_defaults(func=run_off)

    clear_parser = subparsers.add_parser('clear')
    clear_parser.set_defaults(func=run_clear)

    tst_parser = subparsers.add_parser('tst')
    tst_parser.set_defaults(func=run_tst)

    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])



