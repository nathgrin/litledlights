import os
import ledstrip
import config
import numpy as np

try:
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

def rotationmtx(axis:np.ndarray,angle:float) -> np.array:
    #https://en.wikipedia.org/wiki/Rotation_matrix
    axis = axis/np.linalg.norm(axis)
    x,y,z = axis[0],axis[1],axis[2]
    costheta = np.cos(angle)
    sintheta = np.sin(angle)
    onemincos = 1.-costheta
    R = np.array( [
        [ costheta + x*x*onemincos , x*y*onemincos-z*sintheta , x*z*onemincos+y*sintheta ],
        [ y*x*onemincos + z*sintheta , costheta+y*y*onemincos , y*z*onemincos - x*sintheta ],
        [ z*x*onemincos - y*sintheta , z*y*onemincos+x*sintheta , costheta+z*z*onemincos ]
    ] )
    return R

def npunit(index:int,size=3):
    arr = np.zeros(size)
    arr[index] = 1.
    return arr
    

def xyz_to_rthetaphi(xyz): # https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    
    xy = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    phi = np.arctan2(xyz[1], xyz[0])
    return np.array([r,theta,phi])
    
def rthetaphi_to_xyz(rthetaphi):
    x = rthetaphi[0]*np.sin(rthetaphi[1])*np.cos(rthetaphi[2])
    y = rthetaphi[0]*np.sin(rthetaphi[1])*np.sin(rthetaphi[2])
    z = rthetaphi[0]*np.cos(rthetaphi[1])
    return np.array([x,y,z])


    
def get_strip(output_pin: board.pin=board.D18,
        num_pixels: int=config.nleds,
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
