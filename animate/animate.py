import numpy as np
import config
import colors
import coords
try:
    import utils
except:
    print("animate: import utils failed")
    
import misc_func
import time

class AnimationInstruction(dict):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AnimationStrip(list):
    
    def __init__(self,nleds: int=None,initial_state: tuple[int]=(0,0,0),coords3d: coords.Coords3d=None):
        self.nleds = nleds if nleds is not None else config.nleds
        
        if coords3d is None:
            coords3d = coords.get_coords()
        self.coords3d = coords3d
        
        for i in range(self.nleds):
            if coords3d is not None:
                xyz = coords3d[i]
            self.append(AnimationLed(state=initial_state,xyz=xyz))

    def instruct(self,ind,instruction):
        for i in ind:
            self[i].instruct(instruction)

    def render(self,t: float):
        
        return [ led.render(t) for led in self ]

class AnimationLed(list):
    
    def __init__(self,state: colors.Color=colors.Color((0,0,0)),xyz=None,
                 off_state:  colors.Color=colors.Color((0,0,0))):
        self.state = state
        self.xyz = xyz
        self._off_state = off_state
    def __repr__(self):
        return "<{0} {1}>".format(type(self).__name__,str(self.state))
    
    def instruct(self,instruction: AnimationInstruction):
        id = instruction.get('id',None)
        
        if id is not None: # Only allow single instruction per ID
            if id in [instr['id'] for instr in self]:
                return False
        self.append(instruction)
        return True
    
    def render(self,t: float):
        # Something with instruction
        
        if len(self) == 0:
            return self.state
        
        # if any has prio then we should deal with that here
        
        
        # Loop the instructions
        states = []
        for i,instr in enumerate(self):
             
            state = self.one_instr(instr,t)
            
            if state is None: # DELETE the instruction
                self.pop(i)
                i += -1
            else:
                states.append(state)
                
        # Handle combine states
        if len(states) == 1:
            self.state = states[0]
        elif len(states) > 1:
            self.state = colors.combine_colors(*states)
        
        return self.state

    def one_instr(self,instr: AnimationInstruction, t:float,):
        
        state = None # return val, if None, instr will be deleted
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
            
            # print("ONE INSTRUCTION")
            # print(instr)
            c = instr['color']['hsv']
            
            # print('color',instr['color'])
            # print(c)
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
    def __init__(self,x: np.ndarray(3),v: np.ndarray(3),a: np.ndarray(3),force: np.ndarray(3)=np.zeros(3),m: float=1.,id: str=None, instr: AnimationInstruction=None):
        self.x = x
        self.v = v
        self.a = a
        
        self.m = m
        self.force = force
        
        if id is None:
            id = str(time.time())
            print("Generating ID for {0}: {1}".format(type(self).__name__, id))
        self.id = id
        
        self.set_instruction(instr)
    
    def set_instruction(self,instr: AnimationInstruction = None):
        if instr is None:
            instr = AnimationInstruction(which="off")
        
        instr['id'] = self.id
        self.instruction = instr.copy()
        
    def update(self,dt:float,force:np.ndarray(3)) -> str:
        self.force = force
        self.a = force/self.m
        dv = self.a*dt
        dx = (self.v+0.5*dv)*dt
        self.v = self.v + dv
        self.x = self.x + dx
        
        return 'ok'
    
    def evaluate_termination(self,t: float):
        if abs(self.x[0]) > 5:
            return True
        return False
    
    
    def select_lights(self,coords3d: coords.Coords3d):
        ind = np.where(~np.isnan(coords3d.x))
        return ind
    
    def __repr__(self):
        # return "<{0} id:{1} x:{2} v:{3} a:{4}>".format(type(self).__name__, self.id,self.x,self.v,self.a)
        return "<{0} id:{1} x:{2} v:{3}>".format(type(self).__name__, self.id,self.x,self.v)
    
class AnimationBall(AnimationObject):
    type = "ball"
    
    
    
    def __init__(self,*args,**kwargs):
        # print("ballinit")
        
        self.radius = kwargs.pop('radius',1.)
        self.rsqr = self.radius*self.radius
        
        super().__init__(*args,**kwargs)
    
    def select_lights(self,coords3d: coords.Coords3d):
        dists = np.sum( np.power( coords3d.xyz - self.x ,2) ,axis=1)
        ind = np.where( dists < self.rsqr )[0]
        return ind
    
def animationloop(strip: 'ledstrip.ledstrip', anistrip: AnimationStrip, objects: list[type[AnimationObject]],
                  t0: float=0., dt: float=1.):
    
    force = np.zeros(3)
    
    t = t0
    
    # Game Loop
    while True:
        
        print(objects)
        
        # Instruct lights
        for i,obj in enumerate(objects):
            ind = obj.select_lights(anistrip.coords3d)
            anistrip.instruct(ind,obj.instruction)
        
        
        # Render
        states = anistrip.render(t)
        
        strip.set_all(states)
        strip.show()
        
        # print(ind)
        # for i in ind:
        #     print(anistrip[i])
        
        
        input("wait "+str(t))
        
        # Update position
        for i,obj in enumerate(objects):
            flag = obj.update(dt,force)
            
            kill = obj.evaluate_termination(t)
            if kill:
                objects.pop(i)
                i += -1
                
        t += dt
        time.sleep(dt)
    
def main():
    print("animate")
    print("nleds",config.nleds)
    
    
    coords3d = coords.get_coords()
    
    anistrip = AnimationStrip(coords3d=coords3d)
    
    # Prepare instructon
    color_on = colors.Color((115,0,0))
    duration = 5.
    t0 = 0.
    instr = AnimationInstruction(which="Fade",mode="linear",color=color_on,t0=t0,duration=duration)
    instr2 = instr.copy()
    instr2['color'] = colors.Color((0,115,0))
    # print(instr)
    
    
    # Settings
    dt = 0.5
    
    # Define objects
    radius = 1.
    x0,v0,a0 = np.zeros(3),misc_func.npunit(0)/2.,np.zeros(3)
    x02 = misc_func.npunit(0)
    objects = [ AnimationBall(x0,v0,a0,radius=radius,instr=instr,id="A"),
                AnimationBall(x02,-v0,a0,radius=radius,instr=instr2,id="B") ]
    
    
    with utils.get_strip() as strip:
        animationloop(strip,anistrip,objects,t0=t0,dt=dt)
    
if __name__ == "__main__":
    main()