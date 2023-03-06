import numpy as np



class Coords3d(object):
    xyz = None
    cil = None
    sph = None
    
    
    def __init__(self):
        pass
    
    def from_file(self, fname: str="coords.txt"):
        self.xyz = load_coords_file(fname)
        transp = self.xyz.transpose()
        self.x = transp[0]
        self.y = transp[1]
        self.z = transp[2]

    def __getitem__(self, key: int):
        return self.xyz[key]
    
def get_coords(fname: str="coords.txt"):
    coords3d = Coords3d()
    coords3d.from_file(fname)
    return coords3d

def load_coords_file(fname:str="coords.txt"):
    return np.loadtxt(fname)