import os
import config
import numpy as np
import ledstrip

try:
    import neopixel
    import board
except:
    print("utils: imports failed.. set connect_ledlights=False")
    config.connect_ledlights = False

def clear(strip: ledstrip.ledstrip=None):
    if strip is None:
        with get_strip() as strip:
            pass
    else:
        strip.fill( (0,0,0) )

    
def get_strip(output_pin: str='D18', # getattr(board,str)
        nleds: int=config.nleds,
        brightness: float=1.,
        pixel_order: str='RGB', # getattr(neopixel,str)
        auto_write: bool=False,
        
        coords_fname: str=config.coords3d_fname,
        
        ) -> ledstrip.ledstrip:
    
    if config.connect_ledlights:
        output_pin = getattr(board,output_pin)
        pixel_order = getattr(neopixel,pixel_order)
    else:
        output_pin = None
        pixel_order = pixel_order # Now suddenly it is string = bad
    
    strip = ledstrip.ledstrip(output_pin, nleds,brightness=brightness,
            auto_write=auto_write,pixel_order=pixel_order)
    
    if coords_fname is not None:
        if os.path.exists(coords_fname):
            from coords import get_coords
            coords3d = get_coords(coords_fname)
            strip.set_coords3d(coords3d)
    
    return strip
