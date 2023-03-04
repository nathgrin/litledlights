import numpy as np
import config
import colors
try:
    import utils
except:
    print("import utils failed")
    
import misc_func
import time

class AnimationInstruction(dict):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AnimationStrip(list):
    
    def __init__(self,nleds: int=None,initial_state: tuple[int]=(0,0,0),leds_xyz=None):
        self.nleds = nleds if nleds is not None else config.nleds
        
        for i in range(self.nleds):
            if leds_xyz is not None:
                xyz = leds_xyz[i]
            self.append(AnimationLed(state=initial_state,xyz=xyz))

    

    def render(self,t: float):
        
        return [ led.render(t) for led in self ]

class AnimationLed(list):
    
    def __init__(self,state: tuple[int]=colors.Color((0,0,0)),xyz=None,
                 off_state: colors.Color=colors.Color((0,0,0))):
        self.state = state
        self.xyz = xyz
        self._off_state = off_state
    def __repr__(self):
        return str(self.state)
    
    def instruct(self,instruction: AnimationInstruction):
        self.append(instruction)
    
    def render(self,t: float):
        # Something with instruction
        
        if len(self) == 0:
            return self.state
        # if any has prio then we should deal with that here
        states = []
        
        # Loop the instructions
        for i,instr in enumerate(self):
             
            state = self.one_instr(instr,t)
            
            if state is None: # DELETE the instruction
                self.pop(i)
                i += -1
            else:
                states.append(state)
                
        # Handle combine states
        print("STATES",states)
        if len(states) == 1:
            self.state = states[0]
        return self.state

    def one_instr(self,instr, t:float,):
        
        state = None # return val
        which = instr.get('which',None)
        
        if which is None:
            return state
        which = which.lower()
        
        if which == "fade":
            
            t0 = instr.get('t0')
            duration = instr.get('duration')
            
            
            delta = t-t0
            if delta >= duration:
                self.turn_off()
                return state
            
            print("ONE INSTRUCTION")
            print(instr)
            c = instr['color']['hsv']
            
            print('color',instr['color'])
            print(c)
            state = colors.Color( (c[0], c[1], c[2]*(1-(t-t0)/duration)) ,ctype='hsv')
            
            
        elif which == "blink":
            
            t0 = instr.get('t0')
            duration = instr.get('duration')
            c = instr.get('color')
            
            delta = t-t0
            if delta > duration:
                self.turn_off()
                return state
            
            state = c
        
        elif which == "stay":
            
            c = instr.get('color')
            
            self._off_state = c
            self.turn_off()
            return state
            
            
        elif which == "off":
            self.turn_off()
            return state
            
        return state
        
    def turn_off(self):
        self.state = self._off_state
    

class AnimationObject():
    type="object"
    id = ""
    def __init__(self,x: np.ndarray(3),v: np.ndarray(3),a: np.ndarray(3),id=None):
        self.x = x
        self.v = v
        self.a = a
        
        if id is None:
            self.id = str(time.time())
        
        
    def update(self,dt):
        dv = self.a*dt
        dx = (self.v+0.5*dv)*dt
        self.v = self.v + dv
        self.x = self.x + dx
        
        if self.x[0] > 15:
            return 'KILL'
        return True
    
    def __repr__(self):
        msg = "x: {0}, v: {1}, a: {2}".format(self.x,self.v,self.a)
        return '%s(%s)' % (type(self).__name__, msg)
    
class AnimationBall(AnimationObject):
    type = "ball"
    
    

def main():
    print("animate")
    print("nleds",config.nleds)
    
    
    color_on = colors.Color((115,0,0))
    
    
    anistrip = AnimationStrip(leds_xyz=config.coords3d)
    
    duration = 5.
    t0 = 0.
    instr = AnimationInstruction(which="Fade",mode="linear",color=color_on,t0=t0,duration=duration)
    print(instr)
    
    ind = 0
    anistrip[ind].instruct(instr)
    
    # Settings
    t = 0
    dt = 1.
    
    # Define objects
    objects = [ AnimationBall(np.zeros(3),misc_func.npunit(0),np.zeros(3)) ]
    
    
    # Game Loop
    while True:
        t += dt
        
        # Update position
        for i,obj in enumerate(objects):
            flag = obj.update(dt)
            print(flag)
            if flag == 'KILL':
                objects.pop(i)
                i += -1
            
            print(objects)
        
        
        # Instruct lights
        
        
        # Render
        anistrip.render(t)
        
        
        print(anistrip[ind])
        
        input("wait "+str(t))
    
    
if __name__ == "__main__":
    main()