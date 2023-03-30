import numpy as np
import os
import config


class Coords3d(object):
    xyz = None
    cil = None
    sph = None
    
    
    def __init__(self):
        pass
    
    def from_file(self, fname: str=config.coords3d_fname):
        self.xyz = load_coords_file(fname)
        
        self.x = self.xyz[:,0]
        self.y = self.xyz[:,1]
        self.z = self.xyz[:,2]

    def __getitem__(self, key: int):
        return self.xyz[key]
    
def get_coords(fname: str=config.coords3d_fname):
    if os.path.isfile(fname):
        coords3d = Coords3d()
        coords3d.from_file(fname)
    else:
        print("Warning: Could not find file {0}, return None".format(fname))
        coords3d = None
    
    return coords3d

def load_coords_file(fname:str=config.coords3d_fname):
    return np.loadtxt(fname)
