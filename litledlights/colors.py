import numpy as np

class Color(object):
    ctype = "rgb"
    val   = (0,0,0)
    
    _types = ['rgb','hsv','hsl']
    
    def __init__(self,color=(0,0,0),ctype: str="rgb"):
        self.__setitem__(ctype,color)
        

    def __getitem__(self, ctype: str) -> tuple[int,int,int]:
        if ctype not in self._types:
            raise KeyError(self._KeyErrorMessage(ctype))
        if ctype == self.ctype:
            return self.val
        else:
            val = colorfromto[self.ctype][ctype](self.val) # This is probably bad syntax idea
            return val
    
    def get(self,*args):
        if len(args) == 1:
            return self.__getitem__(args[0])
        elif len(args) == 2:
            try:
                return self.__getitem__(args[0])
            except:
                return args[2]
        else:
            raise ValueError("{0}.get() only accepts 1 or 2 args..".format(type(self).__name__))

    def __setitem__(self, ctype: str, color: tuple[int,int,int]):
        # Input validation
        if ctype not in self._types:
            raise ValueError(self._KeyErrorMessage(ctype))
        self.ctype = ctype
        self.val   = color

    def __repr__(self) -> str:
        return "<{0} {1}:{2}>".format(type(self).__name__, self.ctype,self.val)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def _KeyErrorMessage(self,ctype: str) -> str:
        return "{1} not one of valid ctypes: {0}".format(self._types,ctype)

# maybe this should be a json file
red = Color((155,0,0)) # bad, more like pink
blue = Color((0,0,155))
white = Color((155,155,155))
green = Color((0,155,0))
pink = Color((227,28,121))
orange = Color((255,117,0))
gold = Color((255, 215, 0))
namedcolors = {
        'orange': orange,
        'red': red, 
        'pink': pink,
        'blue': blue,
        'green': green,
        'white': white
        }
namedcolors['w'] = white
namedcolors['b'] = blue
namedcolors['r'] = red
namedcolors['g'] = green

def combine_colors(*args): # combine any number of color objects
    # print(args)
    return Color(combine_hsl( *[c['hsl'] for c in args] ),ctype='hsl')
    
    
    
def combine_hsl(*args):
    # print(args)
    
    # for l we take some weird sum that never quite reaches one.
    ls = sorted([a[2] for a in args])[::-1]
    l = ls[0]
    for li in ls[1:]:
        l += (1-l)*li
    
    # for hue, saturation we take the average of the vectors as if they are polar coordinates
    xys = np.array([ (a[1]*np.cos(np.pi*a[0]),a[1]*np.sin(np.pi*a[0]))  for a in args])
    avg = np.mean(xys,axis=0)
    
    s = np.sqrt(np.square(avg[0])+np.square(avg[1]))
    h = np.arctan2(avg[1],avg[0])/(2.*np.pi)
    
    return h,s,l

def rgb_to_hsv(args):
    if len(args) == 1:
        r,g,b = args[0]
    elif len(args) == 3:
        r,g,b = args
    else:
        raise ValueError("Input (r,g,b) as tuple or triple argument, not: %s"%(str(args)))
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    # normalize stupidly
    h = h/360.
    v = v/100.
    s = s/100.
    return h, s, v

def hsv_to_rgb(*args):
    """
    from Tcll https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
    h,s,v in (0,1)
    """
    if len(args) == 1:
        h, s, v = args[0]
    elif len(args) == 3:
        h, s, v = args
    else:
        raise ValueError("Input (h,s,v) as tuple or triple argument, not: %s"%(str(args)))
    if h<0 or h>1 or s<0 or s>1 or v<0 or v>1:
        errormsg = "I accept 0 <= h,s,v <= 1"
        if h>1:
            errormsg += ", try h={0}".format(h/360)
        raise ValueError(errormsg)
    
    if s == 0.0: v*=255; return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    # if i == 0: return (v, t, p)
    # if i == 1: return (q, v, p)
    # if i == 2: return (p, v, t)
    # if i == 3: return (p, q, v)
    # if i == 4: return (t, p, v)
    # if i == 5: return (v, p, q)
    if i == 0: return (int(v), int(t), int(p))
    if i == 1: return (int(q), int(v), int(p))
    if i == 2: return (int(p), int(v), int(t))
    if i == 3: return (int(p), int(q), int(v))
    if i == 4: return (int(t), int(p), int(v))
    if i == 5: return (int(v), int(p), int(q))


# FOllowing are from Lywx at https://gist.github.com/mathebox/e0805f72e7db3269ec22
def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def saturate(value):
    return clamp(value, 0.0, 1.0)

def hue_to_rgb(h: float) -> tuple[float,float,float]:
    """_summary_

    Args:
        h (float): _description_

    Returns:
        tuple[float,float,float]: rgb in (0,1)
    """
    r = abs(h * 6.0 - 3.0) - 1.0
    g = 2.0 - abs(h * 6.0 - 2.0)
    b = 2.0 - abs(h * 6.0 - 4.0)
    return saturate(r), saturate(g), saturate(b)

def hsl_to_rgb(*args):
    if len(args) == 1:
        h, s, l = args[0]
    elif len(args) == 3:
        h, s, l = args
    else:
        raise ValueError("Input (h,s,l) as tuple or triple argument, not: %s"%(str(args)))
    
    r, g, b = hue_to_rgb(h)
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    r = (r - 0.5) * c + l
    g = (g - 0.5) * c + l
    b = (b - 0.5) * c + l
    return int(r*255), int(g*255), int(b*255)

def rgb_to_hsl(*args):
    if len(args) == 1:
        r, g, b = args[0]
    elif len(args) == 3:
        r, g, b = args
    else:
        raise ValueError("Input (r,b,g) as tuple or triple argument, not: %s"%(str(args)))
    
    rgb_p = ( r/255.,g/255.,b/255. )
    
    Cmax = max(rgb_p)
    Cmin = min(rgb_p)
    
    delta = Cmax-Cmin
    
    if delta == 0.:
        H = 0.
    elif Cmax == rgb_p[0]:
        H = 60.* ( ((rgb_p[1]-rgb_p[2])/delta) % 6 )
    elif Cmax == rgb_p[1]:
        H = 60.* ( ((rgb_p[2]-rgb_p[0])/delta) +2. )
    elif Cmax == rgb_p[2]:
        H = 60.* ( ((rgb_p[0]-rgb_p[1])/delta) +4. )
    
    L = (Cmax+Cmin)/2.
    
    if Cmax == 0.:
        S = 0.
    else:
        S = delta/(1.-abs(2.*L-1.))
    
    
    return (H/360.,S,L)

def hsv_to_hsl(*args): # This will introduce rounding err
    if len(args) == 1:
        h, s, v = args[0]
    elif len(args) == 3:
        h, s, v = args
    else:
        raise ValueError("Input (h,s,v) as tuple or triple argument, not: %s"%(str(args)))
    
    l = v * (1 - s/2)
    s = 0 if l in (0, 1) else (v - l)/min(l, 1-l)
    return h, s, l

    
def hsl_to_hsv(*args): # This will introduce rounding err
    if len(args) == 1:
        h, s, l = args[0]
    elif len(args) == 3:
        h, s, l = args
    else:
        raise ValueError("Input (h,s,l) as tuple or triple argument, not: %s"%(str(args)))

    v = l + s * min(l, 1-l)
    s = 0 if v == 0 else 2*(1 - l/v)
    return h, s, v

colorfromto = {'rgb':{'hsv':rgb_to_hsv,'hsl':rgb_to_hsl},
               'hsv':{'rgb':hsv_to_rgb,'hsl':hsv_to_hsl},
               'hsl':{'rgb':hsl_to_rgb,'hsv':hsl_to_hsv},
               }



def random_color():
    ctup = (np.random.uniform(),np.random.uniform(0.5,1.),np.random.uniform(0.5,1.))
    color_on = Color(ctup,ctype='hsv')
    return color_on

def main():
    print(rgb_to_hsl(150,0,0))
    print(rgb_to_hsl(150,150,0))
    print(rgb_to_hsl(150,0,150))
    print( hsl_to_rgb(rgb_to_hsl(150,0,0)))
    print( hsl_to_rgb(rgb_to_hsl(150,150,0)))
    print( hsl_to_rgb(rgb_to_hsl(150,0,150)))
    
if __name__ == "__main__":
    main()
