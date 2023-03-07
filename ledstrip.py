
import numpy as np
import config
import coords

parentclasses = []
try:
    import neopixel
    # it is also possible to make use of `importlib` to go through a
    # list of possible imports rather than doing this one-by-one in
    # this verbose manner
    parentclasses.append(neopixel.NeoPixel)
except:
    print("ledstrip: imports failed..")
    config.connect_ledlights = False



class ledstrip(*parentclasses):
    def __init__(self,*args,**kwargs):
        self.connect_ledlights = connect_ledlights
        if self.connect_ledlights:
            super().__init__(*args,**kwargs)
        
        self.coords3d = None
        
    def __setitem__(self, key,value):
        if type(key) is np.ndarray:
            if key.dtype == bool: # np array bools, single value
                for i,k in enumerate(key):
                    if k:
                        self.__setitem__one(i,value)
        else:
            self.__setitem__one(key,value)
        
    def set_multiple(self,keys,values):
        for i,(key,value) in enumerate(zip(keys,values)):
            if key.dtype == bool:
                if key:
                    self.__setitem__one(i,value)
            else:
                self.__setitem__one(i,value)
    
    def __setitem__one(self, key,value):
        if self.connect_ledlights:
            super().__setitem__(key,value)
        
    def set_coords3d(self,coords3d: coords.Coords3d) -> None:
        self.coords3d = coords3d
        
        
    
        
        
