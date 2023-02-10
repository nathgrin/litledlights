import os
try:
    import ledstrip
    import neopixel
    import board
    rpimode = True
except:
    print("ultils: imports failed..")
    rpimode = False

def clear(strip: neopixel.NeoPixel=None):
    if strip is None:
        with get_strip() as strip:
            pass
    else:
        strip.fill( (0,0,0) )


def get_strip(output_pin: board.pin=board.D18,
        num_pixels: int=200,
        brightness: float=1.,
        pixel_order: neopixel.RGB=neopixel.RGB,
        auto_write: bool=False,
        
        coords_fname: str="coords.txt",
        
        ) -> ledstrip.ledstrip:
    strip = ledstrip.ledstrip(output_pin, num_pixels,brightness=brightness,
            auto_write=auto_write,pixel_order=pixel_order)
    
    if coords_fname is not None:
        if os.path.exists(coords_fname):
            import coords
            coords3d = coords.get_coords(coords_fname)
            strip.set_xyz(coords3d)
    
    
    return strip
