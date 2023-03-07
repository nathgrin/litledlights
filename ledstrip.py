
import numpy as np
import config
import coords

try:
    import neopixel
except:
    print("ledstrip: imports failed..")
    config.dbg = True



class ledstrip(neopixel.NeoPixel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.coords3d = None
        
    def __setitem__(self, key,value):
        if type(key) is np.ndarray:
            if key.dtype == bool: # np array bools, single value
                for i,k in enumerate(key):
                    if k:
                        super().__setitem__(i,value)
        else:
            return super().__setitem__(key,value)
        
    def set_multiple(self,keys,values):
        for i,(key,value) in enumerate(zip(keys,values)):
            if key.dtype == bool:
                if key:
                    super().__setitem__(i,value)
            else:
                super().__setitem__(i,value)
        
        
    def set_coords3d(self,coords3d: coords.Coords3d) -> None:
        self.coords3d = coords3d
        
        
    
        
        
