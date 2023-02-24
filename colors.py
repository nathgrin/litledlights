class Color(object):
    def _init__(self):
        self.x = 0

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


def hsv_to_rgb(h, s, v):
    """
    from Tcll https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
    h,s,v in (0,1)
    """
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

def hsl_to_rgb(h, s, l):
    r, g, b = hue_to_rgb(h)
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    r = (r - 0.5) * c + l
    g = (g - 0.5) * c + l
    b = (b - 0.5) * c + l
    return int(r), int(g), int(b)