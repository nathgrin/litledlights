import neopixel
import numpy as np

class ledstrip(neopixel.NeoPixel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.xyz = None
        
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
        
        
    def set_xyz(self,xyz: np.array) -> None:
        self.xyz = xyz
        
        self.x = xyz.transpose()[0]
        self.y = xyz.transpose()[1]
        self.z = xyz.transpose()[2]
        
    
        
        
