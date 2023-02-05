import neopixel
import board

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
        auto_write: bool=False
        ) -> neopixel.NeoPixel:
    strip = neopixel.NeoPixel(output_pin, num_pixels,brightness=brightness,
            auto_write=auto_write,pixel_order=pixel_order)
    return strip
