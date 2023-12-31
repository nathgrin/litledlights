import cv2 as cv2
import numpy as np
"""Stolen from opencv readthedocs tutorial (full of mistakes, but well it helped) 
https://opencv-tutorial.readthedocs.io/en/latest/app/app.html
https://github.com/rasql/opencv-tutorial/blob/master/cvlib.py
"""
# !BGR!
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)


class App:
    wins = [] # List of windows
    win = None # Current window
    
    options = dict( win_color=BLACK, obj_color=GREEN, sel_color=RED)
    
    win_id = 0
    
    def __init__(self):
        
        #  Keys           key : (function,help msg)
        self.hotkeys = {    'h': (self.help,"help"),
                            'i': (self.inspect,"inspect"),
                            'w': (Window,"new window"),
                            'o': (Rectangle,"new Object"),
                            }
        self.win_id = 0
        
        # Lets go
        winname = 'window0'
        cv2.namedWindow(winname)
        
        Window(win=winname)
        
        xy = [100,40]
        dy = 35
        Text(xy,"Test",anchor='lt')
        xy[1] += dy
        Rectangle(xy,color=BLUE)
        xy[1] += dy
        Circle(xy,anchor='lb')
        xy[1] += dy
        CrossedRectangle(xy,anchor='lb')
        xy[1] += dy
        CrossedCircle(xy,anchor='lb')
        xy[1] += dy
        

        
    def help(self):
        def print_keyval(k,val):
            print("  {0}: {1}".format(repr(k)[1:-1],val)) # Strip 'quotes
            
        print("---- HELP --- ")
        print("> Not implemented yet")
        print_keyval('q',"quit")
        for k,val in self.hotkeys.items():
            print_keyval(k,val[1])
            
        if self.win is not None:
            for k,val in self.win.hotkeys.items():
                print_keyval(k,val[1])
            if self.win.obj is not None:
                print("> Selected object has keys:")
                for k,val in self.win.obj.hotkeys.items():
                    print_keyval(k,val[1])
                print_keyval('alt+L-click', "Move object to cursor (hold to drag)")
    
    def inspect(self):
        print('--- INSPECT ---')
        print('App.wins', App.wins)
        print('App.win', App.win)
        print('App.win.objs', App.win.objs)
        print('App.win.obj', App.win.obj)
        
    def run(self):
        key = ''
        while key != 'q':
            k = cv2.waitKey(0)
            
            key = chr(k)
            
            print("KeyPress",k, key,repr(key))
            
            ret = self.key(key)
            if not ret:
                print("Did not recognize key:",k,key,repr(key))
                self.help()
            

        cv2.destroyAllWindows()
        
    

    def key(self, k):
        """Keypress handler"""

        if self.win is not None:
            ret = self.win.key(k)
            if ret:
                return True
        if k in self.hotkeys:
            self.hotkeys[k][0]()
            return True
        
        return False

        
        
class Window:
    """Create a window."""
    
    obj_options = dict(id=0,
                       color=App.options['obj_color'],
                       anchor=None,
                       )
    
    
    def __init__(self, win=None, img=None, size=[200, 600]):
        App.wins.append(self)
        App.win = self
        
        self.objs = []
        self.obj = None # Currently selected
        
        if img is None:
            img = np.zeros((size[0],size[1], 3), np.uint8)
            img[:,:] = App.options['win_color']

        if win is None:
            win = 'window' + str(App.win_id)
            App.win_id += 1

        self.win = win
        
        self.set_img(img)
        
        self.obj_options = Window.obj_options.copy()
        
        
        
        self.hotkeys = {
                    '\t': (self.select_next_obj,"(tab) Select next object"),
                    chr(27): (self.unselect_obj,"(esc) Deselect object"),
                    'r': (self.delete_obj,"(r) Remove object"),
                    }
        
        cv2.imshow(win, img)
        
        cv2.setMouseCallback(win, self.mouse)
        

    def set_img(self,img):
        
        self.img = img
        
        self.size = img.shape[:2][::-1] # shape of img is (y,x) (and #channels)
        
        self.img0 = img.copy()
        
    

    def mouse(self, event, x, y, flags, param):
        """
        Events:
            EVENT_LBUTTONDBLCLK 7   EVENT_LBUTTONDOWN    1   EVENT_LBUTTONUP    4
            EVENT_MBUTTONDBLCLK 9   EVENT_MBUTTONDOWN    3   EVENT_MBUTTONUP    6
            EVENT_MOUSEHWHEEL  11   EVENT_MOUSEWHEEL    10   EVENT_MOUSEMOVE    0
            EVENT_RBUTTONDBLCLK 8   EVENT_RBUTTONDOWN    2   EVENT_RBUTTONUP    5
        Flags:
            EVENT_FLAG_LBUTTON  1   EVENT_FLAG_MBUTTON   4   EVENT_FLAG_RBUTTON 2
            EVENT_FLAG_CTRLKEY  8   EVENT_FLAG_SHIFTKEY 16   EVENT_FLAG_ALTKEY 32
        """
        # text = 'mouse event {} at ({}, {}) with flags {}'.format(event, x, y, flags)
        # cv2.displayStatusBar(self.win, text, 1000)
        # cv2.displayOverlay(self.win, text, 1000)
        # print(text)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags == cv2.EVENT_FLAG_LBUTTON:
                App.win = self # set self as current active window
                
                self.obj = None
                for obj in self.objs:
                    obj.selected = False
                    if self.obj is None and obj.is_inside(x, y):
                        obj.selected = True
                        self.obj = obj
                
                self.draw() # redraw
        # print(event)  
        
        if event == cv2.EVENT_MOUSEMOVE:
            if flags == cv2.EVENT_FLAG_ALTKEY:
                if self.obj is not None:
                    self.obj.set_pos( [x,y],{'event':event,'flags':flags,'param':param} )
                    self.draw() # redraw

    def key(self,k):
        """Keypress handler"""
        if self.obj is not None:
            ret = self.obj.key(k)
            if ret:
                self.draw()
                return True
        if k in self.hotkeys:
            self.hotkeys[k][0]()
            self.draw()
            return True
        
        return False

        
    def draw(self):
        self.img[:] = self.img0[:] # Restore img

        for obj in self.objs: # Draw objects
            obj.draw()

        cv2.imshow(self.win, self.img)
        
        
    def select_next_obj(self):
        """Select the next object, or the first in none is selected."""
        if self.obj is None:
            i = -1
        else:
            i = self.objs.index(self.obj)
            self.objs[i].selected = False
        
        i = (i+1) % len(self.objs)
        self.objs[i].selected = True
        self.obj = self.objs[i]
    
    def unselect_obj(self):
        if self.obj != None:
            self.obj.selected = False
            self.obj = None
            
    def delete_obj(self):
        if self.obj != None:
            
            i = self.objs.index(self.obj)
            
            obj = self.objs.pop(i)
            obj.selected = False
            del(obj)
            
            self.obj = None
            
            
    
class Object:
    """Add an object to the current window."""
    
    def __init__(self, pos, **kwargs):
        
        pos = np.array(pos)
        
        App.win.objs.append(self)
        # App.win.obj = self # set self as current obj
        self.img = App.win.img
        
        self.selected = False
        
        # kwargs # Watch out, this overwrites the defaults!! sounds very bad
        defaults = App.win.obj_options
        defaults['id'] += 1
        self.id = defaults['id']
        
        self.color = kwargs.get('color',defaults['color'])
        
        anchor = kwargs.get('anchor',defaults['anchor'])
        self.set_anchor( anchor, {} )
            
        self.set_pos(pos, {}) # empty options
        
        self.hotkeys = {}
        
        App.win.draw()

    def __str__(self):
        return '<Object {} at ({}, {})>'.format(self.id, *self.pos)
    
    
    def set_pos(self, xy, options: dict):
        if self.anchor is not None:
            xy = self.center_from_anchor(xy, options)
        self.pos = xy
    
    
    def set_anchor(self, anchor: str, options: dict) -> None:
        helpstr = """
        anchor: str of length 2
        concatenate:
        hor:  (l)eft (c)enter (r)ight
        vert: (t)op (m)iddle (b)ottom
        e.g., cm is center-middle
        exact position depends on shape (see e.g., rectangle)
        """
        err = False
        if anchor is None:
            err = False
        elif len(anchor) != 2:
            err = True
        elif anchor[0] not in 'lcr':
            err = True
        elif anchor[1] not in 'tmb':
            err = True
        
        if err:
            raise ValueError("Invalid Anchor: {}, {} ".format(anchor,helpstr))
        
        self.anchor = anchor
    def center_from_anchor(self, xy, options):
        # see e.g., rectangle
        return xy
    
    def draw(self):
        # see e.g., rectangle
        return False

    def is_inside(self, x, y):
        # see e.g., rectangle
        return False
    
    def mouse(self, event, x, y, flags, param):
        """ See also window mouse method

        Args:
            event (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            flags (_type_): _description_
            param (_type_): _description_
        """
        # print(event)
        if event == cv2.EVENT_LBUTTONDOWN:
            print(self)
        
        
    def key(self, k):
        """Keypress handler"""
        if k in self.hotkeys:
            self.hotkeys[k][0]()
            self.draw()
            return True
        
        return False

class Rectangle(Object):
    def __init__(self, pos: np.ndarray,
                 size: list=[100,40],
                 border_thickness: int=1,
                 **kwargs):
        
        self.size = w, h = size
        self.border_thickness = border_thickness
        
        super().__init__(pos,**kwargs)
    
    def center_from_anchor(self, xy, options):
        w,h = self.size
        # Adjust for anchor
        if self.anchor[0] == 'l':
            xy[0] += w//2
        elif self.anchor[0] == 'r':
            xy[0] += -w//2
        if self.anchor[1] == 't':
            xy[1] += h//2
        elif self.anchor[1] == 'b':
            xy[1] += -h//2
        return xy
    
    def is_inside(self,x,y):
        x0, y0 = self.pos
        w, h = self.size
        return x0 <= x+w//2 <= x0+w and y0 <= y+h//2 <= y0+h
    
    def draw(self):
        x, y = self.pos
        w, h = self.size
        color = self.color if not self.selected else App.options['sel_color']
        cv2.rectangle(self.img, (x-w//2, y-h//2, w, h), color, self.border_thickness)
        

class Text(Rectangle):
    """Add a text object to the current window."""
    
    text_options = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1,
                   thickness=2,
                   lineType=cv2.LINE_8,
                   # color=GREEN, # Color is an object prop
                   bottomLeftOrigin=False)
    
    
    
    def __init__(self, pos: np.ndarray, text: str, draw_box: bool=True,
                 **kwargs):
        
        pos = np.array(pos)
        
        self.text_options = self.text_options.copy()
        
        for k, v in kwargs.items():
            if k in Text.text_options:
                self.text_options[k] = kwargs.pop(k)

        
        self.text = text
        
        
        self.draw_box = draw_box
        
        super().__init__(pos,**kwargs)
        
        self.size = self.get_size()
        
        text_hotkeys = {}
        self.hotkeys.update(text_hotkeys)
        
        
        
    def set_text(self,text: str):
        self.text = text
        
        
    def get_size(self):
        """Returns the text size and baseline under the forme (w, h), b."""
        d = self.text_options
        return cv2.getTextSize(self.text, d['fontFace'], d['fontScale'],d['thickness'])
    
    def draw(self, pos=np.array((0, 0))):
        """_summary_

        Args:
            pos (_type_, optional): offset for text. Defaults to np.array((0, 0)).
        """
        self.size,b = (w, h), b = self.get_size()
        x, y = pos + self.pos # for offset
        
        opt = self.text_options.copy() # tmp copy
        color = self.color
        if self.selected:
            color = App.options['sel_color']
        cv2.putText(self.img, self.text, (x-w//2, y+h//2), color=color,**opt)
        
        if self.draw_box:
            super().draw()
        
    def toggle_drawbox(self):
        self.draw_box = not self.draw_box
    
    

class CrossedRectangle(Rectangle):
    def __init__(self, *args,
                 **kwargs):
        
        super().__init__(*args,**kwargs)
    
    def draw(self):
        x, y = self.pos
        w, h = self.size
        xmin,xmax = x-w//2,x+w//2
        ymin,ymax = y-h//2,y+h//2
        color = self.color if not self.selected else App.options['sel_color']
        cv2.rectangle(self.img, (xmin, ymin, w, h), color, self.border_thickness)
        
        cv2.line( self.img, (xmin,ymin),(xmax,ymax), color, self.border_thickness )
        cv2.line( self.img, (xmin,ymax),(xmax,ymin), color, self.border_thickness )
        
class Circle(Object):
    def __init__(self, pos: np.ndarray,
                 size: float=20, # Size is radius
                 border_thickness: int=1,
                 **kwargs):
        
        self.size = r = size
        self.border_thickness = border_thickness
        
        super().__init__(pos,**kwargs)
    
    
    def center_from_anchor(self, xy, options):
        r = self.size
        # Adjust for anchor
        if self.anchor[0] == 'l':
            xy[0] += r
        elif self.anchor[0] == 'r':
            xy[0] += -r
        if self.anchor[1] == 't':
            xy[1] += r
        elif self.anchor[1] == 'b':
            xy[1] += -r
        return xy
    
    def is_inside(self,x,y):
        x0, y0 = self.pos
        r = self.size
        return (x-x0)**2+(y-y0)**2 <= r**2
    
    def draw(self):
        x, y = self.pos
        r    = self.size
        color = self.color if not self.selected else App.options['sel_color']
        cv2.circle(self.img,(x,y),r,color,self.border_thickness)
    

class CrossedCircle(Circle):
    def __init__(self, *args,
                 **kwargs):
        """_summary_
        kwargs:
        orientation: diagonal or upright
        """
        self.orientation = kwargs.pop('orientation','diagonal')
        
        super().__init__(*args,**kwargs)
    
    def draw(self):
        x, y = self.pos
        r    = self.size
        color = self.color if not self.selected else App.options['sel_color']
        cv2.circle(self.img,(x,y),r,color,self.border_thickness)
        
        if self.orientation == 'diagonal':
            rsin45 = r*0.70710678118 # sin45deg
            xmin,xmax = int(x-rsin45),int(x+rsin45)
            ymin,ymax = int(y-rsin45),int(y+rsin45)
            cv2.line( self.img, (xmin,ymin),(xmax,ymax), color, self.border_thickness )
            cv2.line( self.img, (xmin,ymax),(xmax,ymin), color, self.border_thickness )
        elif self.orientation == 'upright':
            cv2.line( self.img, (x-r,y),(x+r,y), color, self.border_thickness )
            cv2.line( self.img, (x,y-r),(x,y+r), color, self.border_thickness )
    
def main():
    
    print("makecoords2d")
    
    App().run()
    
    
if __name__ == "__main__":
    main()