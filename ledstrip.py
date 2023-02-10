import neopixel
import numpy as np

class ledstrip(neopixel.Neopixel):
    def __init__(self,*args,**kwargs):
        super.__init__(*args,**kwargs)
        
        self.xyz = None
        
        
    def set_xyz(self,xyz: np.array) -> None:
        self.xyz = xyz
        
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
        
    
        
        