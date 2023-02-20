#!/usr/bin/env python3
import argparse
import sys

import board
import neopixel

import numpy as np
import time

from utils import get_strip,clear
import color

import leds


def run_off(args):
    print('Fuck you, shit is lit')


def run_clear(args):
    strip = get_strip()
    strip.fill( (0,0,0) )
    strip.show()


def run_tst(args):
    # print("There is no test :(")
    
    color = ( 75,75,75 )
    
    strip = get_strip()
        
    with strip:
        strip.fill( (0,0,0) )
        
        print("Planes:")
        print("  z-direction up")
        nml = np.array([0,0,1])
        leds.moving_plane(nml,strip=strip, color=color)
        print("  z-direction down")
        nml = np.array([0,0,-1])
        leds.moving_plane(nml,strip=strip, color=color)
        print("  x-direction")
        nml = np.array([1,0,0])
        leds.moving_plane(nml,strip=strip, color=color)
        print("  y-direction")
        nml = np.array([0,1,0])
        leds.moving_plane(nml,strip=strip, color=color)
        print("  xz-direction")
        nml = np.array([1,0,1])
        leds.moving_plane(nml,strip=strip, color=color)
        
        print("Coords Nans:")
        strip.fill( (0,0,0) )
        ind = np.isnan(strip.x)
        strip[ind] = color
        strip.show()
        input("enter to continue")
        
        print("Coords cmap:")
        print("  z-direction")
        nml = np.array([0,0,1])
        leds.colormap_coords(nml)
        input("enter to continue")
        print("  x-direction")
        nml = np.array([1,0,0])
        leds.colormap_coords(nml)
        input("enter to continue")
        print("  y-direction")
        nml = np.array([0,1,0])
        leds.colormap_coords(nml)
        input("enter to continue")
    
    
def run_huphollandhup(args):
    strip = get_strip()
    with strip:
        leds.huphollandhup(strip=strip)

def run_piemel(args):
    strip = get_strip()
    with strip:
        leds.piemel(strip=strip)
        
        input("Done looking at it?")

def run_one(args):
    color_on = ( 125,125,125 )
    
    # print(color_on)
    strip = get_strip()
    
    ind = args.ind[0]
    
    strip[ind] = color_on
    strip.show()
    
    
def run_fill(args):
    color_on = color.orange#( 125,125,125 ) # args!
    
    # print(color_on)
    strip = get_strip()
    
    
    strip.fill( color_on )
    strip.show()



def main(argv):
    # print(argv)
    parser = argparse.ArgumentParser(prog="lll")
    subparsers = parser.add_subparsers()

    off_parser = subparsers.add_parser('off')
    off_parser.set_defaults(func=run_off)

    clear_parser = subparsers.add_parser('clear')
    clear_parser.set_defaults(func=run_clear)

    tst_parser = subparsers.add_parser('tst')
    tst_parser.set_defaults(func=run_tst)
    
    tst_parser = subparsers.add_parser('huphollandhup')
    tst_parser.set_defaults(func=run_huphollandhup)
    
    tst_parser = subparsers.add_parser('piemel')
    tst_parser.set_defaults(func=run_piemel)

    one_parser = subparsers.add_parser('one')
    one_parser.set_defaults(func=run_one)
    one_parser.add_argument('ind',type=int,nargs=1,help="index of the led")
    
    one_parser = subparsers.add_parser('fill')
    one_parser.set_defaults(func=run_fill)
    one_parser.add_argument('color',type=str,nargs=1,help="fillcolor")

    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])



