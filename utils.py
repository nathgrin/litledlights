import os
import config
import numpy as np
import ledstrip

try:
    import neopixel
    import board
except:
    print("ultils: imports failed..")
    config.dbg = True

def clear(strip: ledstrip.ledstrip=None):
    if strip is None:
        with get_strip() as strip:
            pass
    else:
        strip.fill( (0,0,0) )

    
def get_strip(output_pin: 'board.pin'=board.D18,
        num_pixels: int=config.nleds,
        brightness: float=1.,
        pixel_order: 'neopixel.RGB'=neopixel.RGB,
        auto_write: bool=False,
        
        coords_fname: str=config.coords3d_fname,
        
        ) -> ledstrip.ledstrip:
    strip = ledstrip.ledstrip(output_pin, num_pixels,brightness=brightness,
            auto_write=auto_write,pixel_order=pixel_order)
    
    if coords_fname is not None:
        if os.path.exists(coords_fname):
            from coords import get_coords
            coords3d = get_coords(coords_fname)
            strip.set_coords3d(coords3d)
    
    
    return strip
