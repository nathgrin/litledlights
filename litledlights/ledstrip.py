
import numpy as np
import config
import colors
import coords

parentclasses = []
try:
    import neopixel
    # it is also possible to make use of `importlib` to go through a
    # list of possible imports rather than doing this one-by-one in
    # this verbose manner
    parentclasses.append(neopixel.NeoPixel)
except:
    print("ledstrip: imports failed.. set connect_ledlights=False")
    config.connect_ledlights = False



class ledstrip(*parentclasses):
    def __init__(self,*args,**kwargs):
        self.connect_ledlights = config.connect_ledlights
        if self.connect_ledlights:
            super().__init__(*args,**kwargs)
        
        self.coords3d = None
        
        
    def __enter__(self):
        if self.connect_ledlights:
            return super().__enter__()
        else: # this is stupid, this is waht super().__enter does
            return self
    
    def __exit__(self,*args,**kwargs):
        if self.connect_ledlights:
            super().__exit__(*args,**kwargs) # Super does deinit(), which turns all lights off (this makes with work)
            

        
    def __setitem__one(self, key: int,value: 'colors.Color'):
        if self.connect_ledlights:
            if isinstance(value,colors.Color):
                val = value['rgb']
            else:
                val = value
            super().__setitem__(key,val)
        
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
    def set_all(self,values):
        for i,value in enumerate(values):
            self.__setitem__one(i,value)
        
    
    def show(self):
        if self.connect_ledlights:
            super().show()
    
    def set_coords3d(self,coords3d: coords.Coords3d) -> None:
        self.coords3d = coords3d
        
    def __repr__(self):
        return "<{0} nleds: {1} havecoords: {2}>".format(type(self).__name__,self.n,self.coords3d is not None)
        
    
        
        
