from utils import get_strip
import time
import numpy as np


import utils

def colormap_coords(
        direction: np.array, # direction vector
        strip=None,
        cmap_name: str=None,
        color_off:tuple[int]=(0,0,0),
        vmin: float=None, vmax: float=None
        ):
    
    strip = get_strip() if strip is None else strip
    
    if strip.xyz is not None:
            
        vals = np.dot(strip.xyz , direction)
        
        import matplotlib as mpl
        cmap_name = "viridis" if cmap_name is None else cmap_name
        cmap = mpl.cm.get_cmap(cmap_name)
        vmin = np.nanmin(vals) if vmin is None else vmin
        vmax = np.nanmax(vals) if vmax is None else vmax
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        
        colors = cmap(norm(vals))
        colors = [ (int(c[0]*255),int(c[1]*255),int(c[2]*255)) for c in colors ]
        # print(colors) #rgbw?
        ind = vals < np.inf
        strip.set_multiple( ind,colors )
        # for i,c in enumerate(colors):
            # strip[i] = (int(c[0]*255),int(c[1]*255),int(c[2]*255))
        
        strip.show()
        
            
        
    else:
        print("Plane warning: No xyz")
    
    


def moving_plane(
        normal: np.ndarray, # normal vector of the plane (also defines direction!)
        strip=None,
        
        speed: float = 1.,
        p0: float = -3., # p0 and max in terms of normal veector where the plane starst
        pmax: float = 3., # im pretty sure it travels in the normal direction ,but does it?
        
        color:tuple[int]=(115,115,115),
        dt: float=0.1, loop: bool=False,
        color_off:tuple[int]=(0,0,0),
        thickness: float=0.1,
        ):
    
    
    strip = get_strip() if strip is None else strip
    # print("in a plane:",strip)
    # print(strip.x)
    # print(strip.xyz)
    # with strip:
    if strip.xyz is not None:
        
        t = 0
        while True:
            t += dt
            pos = speed*t+p0
            
            strip.fill(color_off)
            
            
            ind1 = np.dot(strip.xyz , normal) < pos +thickness
            ind2 = np.dot(strip.xyz , normal) > pos -thickness
            ind = np.logical_and(ind1,ind2)
            strip[ind] = color
            
            strip.show()
            time.sleep(dt)
            
            if pos > pmax:
                break
            
        strip.fill( color_off )
        strip.show()
        
    else:
        print("Plane warning: No xyz")
        
        
        
def rotating_plane(
        normal: np.ndarray, # normal vector of the plane (also defines direction!)
        rotation_axis: np.ndarray, # nml vector rotates around this
        strip=None,
        
        pos: np.ndarray = np.array([0,0,1]),
        
        period: float = 5.,
        thickness: float=0.1,
        
        color:tuple[int]=(115,115,115),
        color_off:tuple[int]=(0,0,0),
        
        mode: str="fill", # bar for bad with thickness, fill for 2sided fill
        
        dt: float=0.1,
        loop: bool=False,
        tmax: float= 100.,
        ):
    
    
    strip = get_strip() if strip is None else strip
    # print("in a plane:",strip)
    # print(strip.x)
    # print(strip.xyz)
    # with strip:
    if strip.xyz is not None:
        
        t = 0
        while True:
            # pos = speed*t+p0
            normal = np.dot( utils.rotationmtx( rotation_axis, 2*np.pi*dt/period ) , normal )
            
            strip.fill(color_off)
            
            
            if mode == "fill":
                ind = np.dot(strip.xyz -pos , normal) <=  0
            else: # bar
                ind1 = np.dot(strip.xyz -pos , normal) <  thickness
                ind2 = np.dot(strip.xyz -pos , normal) > -thickness
                ind = np.logical_and(ind1,ind2)
            
            strip[ind] = color
            
            strip.show()
            time.sleep(dt)
            
            if t > tmax and not loop:
                break
            t += dt
            
            
        strip.fill( color_off )
        strip.show()
        
    else:
        print("Plane warning: No xyz")
      
def huphollandhup(strip=None,
        
        dt: float=0.1, loop: bool=False,
        color_off:tuple[int]=(0,0,0),
        ):
    from color import namedcolors,hsv_to_rgb
    orange = namedcolors['orange']
    red = namedcolors['red']
    white = namedcolors['white']
    blue = namedcolors['blue']
    
    strip = get_strip() if strip is None else strip  
    
    if strip.xyz is not None:
        z = strip.z
        zmin,zmax = np.nanmin(z),np.nanmax(z)
        dz = zmax-zmin
        
        # [0] + [1]*sin( [2]*x - [3]*t + [4] )
        # cnst_red  = [1.8*dz/3.,dz/30.,2.*np.pi/1.,2.*np.pi/2.,0]
        # cnst_blue = [0.8*dz/3.,dz/30.,2.*np.pi/1.,2.*np.pi/1.3,0]
        cnst_red  = [2.*dz/3.,0,0,0,0]
        cnst_blue = [1.*dz/3.,0,0,0,0]
        
        cnst_val_red   = [0.5,0.2,2.*np.pi/1.,2.*np.pi/2.,0]
        cnst_val_white = [0.5,0.2,2.*np.pi/1.,2.*np.pi/2.,0]
        cnst_val_blue  = [0.5,0.2,2.*np.pi/1.,2.*np.pi/2.,0]
        
        
        def sinfunc(x,t,cnst):
            return cnst[0] + cnst[1]*np.sin(cnst[2]*x-cnst[3]*t+cnst[4])
        
        t = 0
        while True:
            t += dt
            
            white_ind = z < np.inf # all
            
            # Red
            ind = z >= zmin+sinfunc(strip.x,t,cnst_red)
            val = sinfunc(strip.x,t,cnst_val_red)
            h,s = 0.,1.
            color = [ hsv_to_rgb( h,s,v ) if not np.isnan(v) else (0,0,0) for v in val ]
            strip.set_multiple(ind,color)
            
            white_ind = np.logical_and(white_ind,~ind) # filter redz
            
            # Blue
            ind = z <= zmin+sinfunc(strip.x,t,cnst_blue)
            val = sinfunc(strip.x,t,cnst_val_blue)
            h,s = 240./360.,1.
            color = [ hsv_to_rgb( h,s,v ) if not np.isnan(v) else (0,0,0) for v in val ]
            strip.set_multiple(ind,color)
            
            white_ind = np.logical_and(white_ind,~ind) # filter bluez
            
            # White
            ind = white_ind # otherz
            val = sinfunc(strip.x,t,cnst_val_white)
            h,s = 0.,0.
            color = [ hsv_to_rgb( h,s,v ) if not np.isnan(v) else (0,0,0) for v in val ]
            strip.set_multiple(ind,color)
            
            
            strip.show()
            time.sleep(dt)
            
            if t > 100:
                break
        
    strip.fill(orange)
    strip.show()
    input()
            
def piemel(strip=None,
        color:tuple[int]=(155,0,10),
        dt: float=1., loop: bool=False,
        color_off:tuple[int]=(0,0,0)):
        
        strip = get_strip() if strip is None else strip
    
        radius = 0.2
        ball1 = (0,0,0)
        ball2 = (-0.5,0,0)
        
        ind = np.square(strip.x-ball1[0])+np.square(strip.y-ball1[1])+np.square(strip.z-ball1[2]) <= radius*radius
        strip[ind] = color
        
        ind = np.square(strip.x-ball2[0])+np.square(strip.y-ball2[1])+np.square(strip.z-ball2[2]) <= radius*radius
        strip[ind] = color
        
        pillar = (-0.25,0,0)
        ind1 = np.square(strip.x-pillar[0])+np.square(strip.y-pillar[1]) <= 0.75*0.75*radius*radius
        ind2 = strip.z > 0
        ind = np.logical_and( ind1, ind2 )
        strip[ind] = color
        
        
        strip.show()
        
                
        
        
        
        

def cycle_sequential(strip=None,
        color:tuple[int]=(255,255,255),
        dt: float=1., loop: bool=False,
        color_off:tuple[int]=(0,0,0)):
    
    strip = get_strip() if strip is None else strip
    
    with strip:
        strip.fill( color_off )
        strip.show()
        while True:

            for l in range(len(strip)):
                strip[l-1] = color_off
                strip[l]   = color
                strip.show()
                time.sleep(dt)
            if not loop:
                break


def blink_binary(inlist: list[int],nbits: int=None,
        strip=None,
        color:tuple[int]=(255,255,255),
        dt: float=1., loop: int=False,
        color_off:tuple[int]=(0,0,0)):
    
    strip = get_strip() if strip is None else strip
    
    
    bins = [format(n, 'b') for n in inlist]
    lens = [ len(n) for n in bins ]
    nbits = max(lens) if nbits is None else max(max(lens),nbits) # Watch out, theres no warning here!
    bins = [ x.zfill(nbits) for x in bins ]
    # ~ print(bins)
    # ~ print(lens)
    
    with strip:
        while True:
            strip.fill( color_off )
            strip.show()
            
            for i in range(nbits):
                for l in range(len(strip)):
                    strip[l] = color if int(bins[l][i]) else color_off
                
                strip.show()
                time.sleep(dt)
                
            if not loop:
                break
        
        
        
        

    
def main():
    
    with get_strip() as strip:
        # inlist = range(len(strip))
        # blink_binary(inlist,strip=strip)
        rotating_plane(utils.npunit(0),utils.npunit(1),strip=strip)
    

if __name__ == "__main__":
    main()
    
