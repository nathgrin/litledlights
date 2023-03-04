class Color(dict):
    
    def __init__(self,color=(0,0,0),ctype: str="rgb"):
        # if ctype == "rgb":
        #     rgb = color
        #     self.update(rgb=rgb)
        # elif ctype == "hsv":
        #     hsv = color
        #     rgb = hsv_to_rgb(hsv)
        #     self.update(rgb=rgb,hsv=hsv)
        
        self.update({ctype:color})
        

    def __getitem__(self, key):
        # print('GET', key)
        try: # Find requested Key
            val = dict.__getitem__(self, key)
        except KeyError as err:
            # Or look for another to convert
            for ctype in [ 'rgb','hsv','hsl']:
                if key == ctype:
                    continue
                val = self.get(ctype,None)
                if val is not None:
                    out = colorfromto[ctype][key](val) # This is probably bad syntax idea
                    return out
            raise err
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
            raise ValueError(".get() only accepts 1 or 2 args..")

    def __setitem__(self, key, val):
        # print('SET', key, val)
        dict.__setitem__(self, key, val)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)
        
    def update(self, *args, **kwargs):
        # print('update', args, kwargs)
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
    
    def __str__(self):
        return str(self['rgb'])
    

# maybe this should be a json file
red = (155,0,0) # bad, more like pink
blue = (0,0,155)
white = (155,155,155)
green = (0,155,0)
pink = (227,28,121)
orange = (255,117,0)
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
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)


# FOllowing are from Lywx at https://gist.github.com/mathebox/e0805f72e7db3269ec22
def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def saturate(value):
    return clamp(value, 0.0, 1.0)

def hue_to_rgb(h):
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
    return int(r), int(g), int(b)


colorfromto = {'rgb':{'hsv':rgb_to_hsv,'hsl':None},
               'hsv':{'rgb':hsv_to_rgb,'hsl':None},
               'hsl':{'rgb':hsl_to_rgb,'hsv':None},
               }