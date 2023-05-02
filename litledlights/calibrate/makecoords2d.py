import cv2 as cv
import numpy as np
"""Stolen from opencv readthedocs tutorial (not recommended) 
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
        cv.namedWindow('window0')
        #  Keys           key : (function,help msg)
        self.shortcuts = { 'h': (self.help,"help"),
                           'i': (self.inspect,"inspect"),
                           'w': (Window,"new window"),
                           'o': (Object,"new Object"),
                           'n': (Node,"new Node"),
                           }
        
        self.win_id = 0
        
        Window()
        
    def help(self):
        print("---- HELP --- ")
        print("> Not implemented yet")
        print("q: quit")
        for k,val in self.shortcuts.items():
            print("{0}: {1}".format(k,val[1]))
    
    def inspect(self):
        print('--- INSPECT ---')
        print('App.wins', App.wins)
        print('App.win', App.win)
        print('App.win.objs', App.win.objs)
        print('App.win.obj', App.win.obj)
        print('App.win.children', App.win.children)
        print('App.win.node', App.win.node)
        
    def run(self):
        key = ''
        while key != 'q':
            k = cv.waitKey(0)
            
            key = chr(k)
            
            print(k, key)
            self.key(key)
            

        cv.destroyAllWindows()
        
    

    def key(self, k):
        "Keypress handler"

        if self.win is not None:
            ret = self.win.key(k)
            if ret:
                return True
        if k in self.shortcuts:
            self.shortcuts[k][0]()
            return True
        else:
            self.help()
        return False

        
        
class Window:
    """Create a window."""
    
    obj_options = dict(pos=(100, 40), size=(100, 30), id=0)
    
    node_options = dict(pos=np.array((20, 20)),
                        size=np.array((100, 20)),
                        gap=np.array((10, 10)),
                        dir=np.array((0, 1)),
                        )
    
    def __init__(self, win=None, img=None):
        App.wins.append(self)
        App.win = self
        
        self.objs = []
        self.obj = None # Currently selected
        # For some reason we treat nodes sepeartely from other objects?
        self.children = []
        self.stack = [self] # parent stack
        self.node = None # Currently selected
        self.current_parent = None

        if img is None:
            img = np.zeros((200, 600, 3), np.uint8)
            img[:,:] = App.options['win_color']

        if win is None:
            win = 'window' + str(App.win_id)
            App.win_id += 1

        self.win = win
        self.img = img


        self.img0 = img.copy()
        
        self.obj_options = Window.obj_options.copy()
        
        self.upper = False
        
        self.shortcuts = {  '\t': (self.select_next_obj,"Select next object"),
                    chr(27): (self.unselect_obj,"Deselect object"),
                    chr(0): (self.toggle_case,"Toggle case"), }
        
        cv.imshow(win, img)
        
        cv.setMouseCallback(win, self.mouse)
        
        # Reset node options
        Node.options = Window.node_options.copy()

    

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
        text = 'mouse event {} at ({}, {}) with flags {}'.format(event, x, y, flags)
        # cv.displayStatusBar(self.win, text, 1000)
        # cv.displayOverlay(self.win, text, 1000)
        # print(text)
        
        if event == cv.EVENT_LBUTTONDOWN:
            App.win = self # set self as current active window
            
            self.draw() # redraw
            
            
            self.obj = None
            for obj in self.objs:
                obj.selected = False
                if obj.is_inside(x, y):
                    obj.selected = True
                    self.obj = obj
        # print(event)
        if event == cv.EVENT_MOUSEMOVE:
            if flags == cv.EVENT_FLAG_ALTKEY:
                if self.obj is not None:
                    self.obj.pos = x-self.obj.size[0]//2, y-self.obj.size[1]//2

    def key(self,k):
        """Keypress handler"""
        if self.obj is not None:
            ret = self.obj.key(k)
            if ret:
                self.draw()
                return True
        if k in self.shortcuts:
            self.shortcuts[k][0]()
            self.draw()
            return True
        
        return False

        
    def draw(self):
        self.img[:] = self.img0[:] # Restore img

        for obj in self.objs: # Draw objects
            obj.draw()

        cv.imshow(self.win, self.img)
        
        
    def select_next_obj(self):
        """Select the next object, or the first in none is selected."""
        try:
            i = self.objs.index(self.obj)
        except ValueError:
            i = -1
        self.objs[i].selected = False
        i = (i+1) % len(self.objs)
        self.objs[i].selected = True
        self.obj = self.objs[i]
    
    def unselect_obj(self):
        if self.obj != None:
            self.obj.selected = False
            self.obj = None
    
    def toggle_case(self):
        
        # elif k == chr(0):  # alt, ctrl, shift
        self.upper = not self.upper
        if self.upper:
            print("UPPER case")
        else:
            print("LOWER CASE")
        # if self.upper:
        #     cv.displayStatusBar(self.win, 'UPPER case', 1000)
        # else:
        #     cv.displayStatusBar(self.win, 'LOWER case', 1000)
        return True

class Object:
    """Add an object to the current window."""
    def __init__(self, **options):
        App.win.objs.append(self)
        App.win.obj = self
        self.img = App.win.img
        
        self.selected = False
        
        # options 
        d = App.win.obj_options
        d.update(options)
        self.id = d['id']
        self.pos = x, y = d['pos']
        self.size = w, h = d['size']
        
        d['id'] += 1 # increment id
        d['pos'] = x,y+h+5 # position of next object
        
        
        self.shortcuts = {  }
        
        App.win.draw()

    def __str__(self):
        return '<Object {} at ({}, {})>'.format(self.id, *self.pos)
    
    def draw(self):
        x, y = self.pos
        w, h = self.size
        color = App.options['obj_color'] if not self.selected else App.options['sel_color']
        cv.rectangle(self.img, (x, y, w, h), color, 1)

    def is_inside(self, x, y):
        x0, y0 = self.pos
        w, h = self.size
        return x0 <= x <= x0+w and y0 <= y <= y0+h
    
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
        if event == cv.EVENT_LBUTTONDOWN:
            print(self)
            
        
    def key(self, k):
        """Keypress handler"""
        if k in self.shortcuts:
            self.shortcuts[k][0]()
            self.draw()
            return True
        
        return False
        

class Text(Object):
    """Add a text object to the current window."""
    options = dict( fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=BLUE,
                    thickness=1,
                    lineType=cv.LINE_8, )
    
    def __init__(self, text='Text', **options):
        
        self.text_options = Text.options.copy()
        
        for k, v in options.items():
            if k in Text.options:
                self.text_options[k] = v

        self.text = text
        
    def get_size(self):
        """Returns the text size and baseline under the forme (w, h), b."""
        d = self.text_options
        return cv.getTextSize(self.text, d['fontFace'], d['fontScale'],d['thickness'])

class Node:
    
    def __init__(self,level=0, **options):
        # This is very nasty, default node options get updated and saved at creation of new notes
        # update node options from constructor options
        for k, v in options.items():
            if k in Node.options:
                if isinstance(v, tuple):
                    v = np.array(v)
                Node.options[k] = v
            
        # create instance attributes
        self.pos = None
        self.size = None
        self.gap = None
        self.dir = None
        self.level = None
        self.parent = None
        self.win = None
        # update instance attributes from node options
        self.__dict__.update(Node.options)
        
        # Next node position
        pos = self.pos + (self.size+self.gap)*self.dir
        Node.options['pos'] = pos
        
        if self.win is None:
            self.win = App.win
        
        if self.parent is None:
            self.parent = self.win.node
        self.win.current_parent = self.parent
        
        # if level was negative
        for i in range(-self.level):
            self.win.current_parent.enclose_children()
            self.parent = self.win.current_parent.parent
            self.win.current_parent = self.parent
        
        
        
    def draw(self, pos=np.array((0, 0))):
        x, y = pos + self.pos
        w, h =  self.size
        cv.rectangle(self.img, (x, y, w, h), RED, 1)
        if self.selected:
            cv.rectangle(self.img, (x-2, y-2, w+4, h+4), GREEN, 1)

        for child in self.children:
            child.draw(self.pos)
            
    def is_inside(self, pos):
        """Check if the point (x, y) is inside the object."""
        pos = np.array(pos)
        return all(self.pos < pos) and all(pos < self.pos+self.size)
        
    def enclose_children(self):
        p = np.array((0, 0))
        for node in self.children:
            p = np.maximum(p, node.pos+node.size)
        self.size = p - self.pos # thiongy doesnt say self.pos?

class Demo(App):
    def __init__(self):
        super().__init__()

        Node()
        Node(level=1)
        Node()
        Node()
        Node(level=-1, dir=(1, 0))
        Node()
        Node()
        Node()

def main():
    
    print("makecoords2d")
    
    # App().run()
    Demo().run()
    
    
if __name__ == "__main__":
    main()