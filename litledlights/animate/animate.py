import numpy as np
import config
import colors
random_color = colors.random_color
import coords
from typing import Callable
try:
    import utils
except:
    print("animate: import utils failed")
try:
    import ledstrip
except:
    print("animate: import ledstrip failed")
import misc_func
import time

class AnimationInstruction(dict):
    # List of possible kwargs here..
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AnimationStrip(list):
    
    def __init__(self,nleds: int=None,initial_state: tuple[int]=(0,0,0),
        t0: float=0.,
        coords3d: coords.Coords3d=None):
        self.nleds = nleds if nleds is not None else config.nleds
        
        self.t = t0
        
        if coords3d is None:
            coords3d = coords.get_coords()
        self.coords3d = coords3d
        
        for i in range(self.nleds):
            if coords3d is not None:
                xyz = coords3d[i]
            else:
                xyz = None
            self.append(AnimationLed(state=initial_state,xyz=xyz))
    
    def set_off_states(self,off_state):
        for i in range(len(self)):
            self[i]._off_state = off_state
    
    def turn_all_off(self):
        for i in range(len(self)):
            self[i].turn_off()
            
    def instruct(self,ind,instruction):
        if instruction.get('t0',None) is not None:
            instruction['t0'] = self.t
        for i in ind:
            self[i].instruct(instruction)
            
    def instruct_all(self,instruction):
        self.instruct(range(len(self)),instruction)

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
        theid = instruction.get('id',None)
        overwrite = instruction.get('overwrite',False)
        
        if theid is not None: # Only allow single instruction per ID
            for i,instr in enumerate(self):
                if theid == instr['id']:
                    if overwrite:
                        self.pop(i)
                    else:
                        return False
        self.append(instruction.copy())
        # print(instruction)
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
            mode = instr.get('mode',"linear")
            
            delta = t-t0
            if delta >= duration:
                self.turn_off()
                return state
            
            c = instr['color']['hsv']
            if mode == "staythenlinear":
                tfrac = instr.get('tfrac',0.5) # frac of duration
                t1 = tfrac*duration
                if delta > t1:
                    state = colors.Color( (c[0], c[1], c[2]*(1-(t-t0-t1)/t1)) ,ctype='hsv')
                else:
                    state = instr['color']
                    # print(state)
                
            else: # linear
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
            state = c
            # self._off_state = c
            # self.turn_off()
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
    termination_func = None
    force_func = None
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
        
    def calculate_force(self,dt:float) -> np.ndarray(3):
        if self.force_func is None:
            return np.zeros(3)
        else:
            return self.force_func(self,dt)
    
    def update(self,dt:float,force:np.ndarray(3)) -> str:
        self.force = force
        self.a = force/self.m
        dv = self.a*dt
        dx = (self.v+0.5*dv)*dt
        self.v = self.v + dv
        self.x = self.x + dx
        
        return 'ok'
    
    def evaluate_termination(self,t: float):
        if self.termination_func is not None:
            return self.termination_func(self,t)
        
        # if abs(self.x[0]) > 5 or abs(self.x[1])>5 or abs(self.x[2])>5:
            # return True
        return False
    
    def set_termination_func(self, func):
        """ termination func needs to accept (self,t) see evaluate_termination
        """
        self.termination_func = func
    def set_force_func(self, func):
        """ force func needs to accept (self,dt) see evaluate_force
        """
        self.force_func = func
    
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
    
class AnimationPlane(AnimationObject):
    type = "ball"
    
    
    
    def __init__(self,*args,**kwargs):
        # print("ballinit")
        
        self.thickness = kwargs.pop('thickness',1.)
        self.normal = kwargs.pop('normal',np.zeros(3))
        norm = np.linalg.norm(self.normal)
        if norm != 0.:
            self.normal = self.normal/norm
        
        super().__init__(*args,**kwargs)
    
    def select_lights(self,coords3d: coords.Coords3d):
        ind1 = np.dot(coords3d.xyz-self.x , self.normal) <  0.5*self.thickness
        ind2 = np.dot(coords3d.xyz-self.x , self.normal) > -0.5*self.thickness
        ind = np.logical_and(ind1,ind2)
        ind = np.where( ind )[0]
        return ind
    
def animationloop(strip: 'ledstrip.ledstrip', anistrip: AnimationStrip, objects: list[type[AnimationObject]],
                  t0: float=0., dt: float=1.,
                  tmax: float=300,
                  
                  spawn_func: Callable = None,
                  spawn_odds: float = None,
                  
                  idcnt: int=0):
    
    do_spawn = spawn_func is not None and spawn_odds is not None
    
    anistrip.t = t0
    
    # Game Loop
    while True:
        timetime = time.time()
        # print("objects",t,objects)
        if do_spawn:
            if np.random.uniform() < spawn_odds:
                # print("SPAWN")
                objs_to_spawn = spawn_func(str(idcnt))
                for obj in objs_to_spawn: obj.t0 = anistrip.t
                objects.extend(objs_to_spawn)
                idcnt += 1
        
        
        # Instruct lights
        for i,obj in enumerate(objects):
            ind = obj.select_lights(anistrip.coords3d)
            # print(ind)
            anistrip.instruct(ind,obj.instruction)
        
        
        # Render
        states = anistrip.render(anistrip.t)
        
        strip.set_all(states)
        strip.show()
        
        # print(ind)
        # for i in ind:
        #     print(anistrip[i])
        
        
        # input("wait "+str(t))
        
        # Update position
        for i,obj in enumerate(objects):
            force = obj.calculate_force(dt)
            flag = obj.update(dt,force)
            # print(obj.x)
            
            kill = obj.evaluate_termination(anistrip.t)
            if kill:
                # print("KILL")
                objects.pop(i)
                i += -1
        
        print("End loop, comp.time: {0} (dt: {1})".format(time.time()-timetime,dt))
        anistrip.t += dt
        time.sleep(max(0,dt-time.time()+timetime))
        
        if anistrip.t > tmax:
            print("Max t ({0}) reached".format(tmax))
            break
    
def spawn_ball(x0,v0,a0,radius,instr,id):
    return [AnimationBall(x0,v0,a0,radius=radius,instr=instr,id=id)]
    
def terminate_distance_from_start(self,t):
    return np.linalg.norm(self.x-self.x0) > self.max_distance
def terminate_time_from_start(self,t):
    return abs(t-self.t0) > self.max_t
    
    
    
    
def spawn_random_ray(id,
        x0: np.ndarray(3)=None,v0: np.ndarray(3)=None,a0: np.ndarray(3)=None,
        v: float=None,
        color_on: colors.Color=None,
        which: str="Fade",mode: str="staythenlinear",duration: float=None, tfrac: float=None, # For instruction
        radius: float=None, max_distance: float=None,
        ):
    
    if color_on is None:
        # random color
        color_on = random_color()
    
    t0 = 0. # dummy t0
    
    duration = 0.5 if duration is None else duration
    tfrac = 0.5 if tfrac is None else tfrac # fraction of duration stay on after which linear decay
    which = "Fade" if which is None else which
    mode = "staythenlinear" if mode is None else mode
    # instruction = AnimationInstruction(which="Fade",mode="linear",color=color_on,t0=t0,duration=duration)
    instruction = AnimationInstruction(which=which,mode=mode,color=color_on,t0=t0,duration=duration,tfrac=tfrac)
    instr = instruction
    
    x0 = np.random.uniform(-1,1,3) if x0 is None else x0
    
    v0 = np.random.uniform(-1,1,3) if v0 is None else v0
    v =  np.random.uniform(3,4) if v is None else v
    v0 = v*v0/np.linalg.norm(v0)
    
    a0 = np.zeros(3) if a0 is None else a0
    
    
    radius = 0.15 if radius is None else radius
    max_distance = np.random.uniform(2.,4.) if max_distance is None else max_distance
    
    ray = AnimationBall(x0,v0,a0,radius=radius,instr=instr,id=id)
    
    ray.set_termination_func( terminate_distance_from_start )
    ray.max_distance = max_distance
    ray.x0 = x0.copy()
    
    # print("Color",color_on,color_on['rgb'])
    # print("x0",np.linalg.norm(x0),x0)
    # print("v0",np.linalg.norm(v0),v0)
    
    return [ray]

def spawn_fireworks(theid: str,
            x0: np.ndarray(3)=None,
            color_on: colors.Color=None,
            ):
    objects = []
    
    if color_on is None:
        # random color
        color_on = random_color()
    
    if x0 is None:
        x0 = np.zeros(3)
        x0 = np.random.uniform(-0.5,0.5,3)
    
    nrays = np.random.randint(12,20)
    
    theid = "{0}_{1}".format(theid,time.time())
    
    for i in range(nrays):
        objects.extend( spawn_random_ray(theid,x0=x0,color_on=color_on) )
    
    return objects
    
def fireworks(coords3d:'coords.Coords3d'=None,
            anistrip:AnimationStrip=None):
    
    if coords3d is None:
        coords3d = coords.get_coords()
    if anistrip is None:
        anistrip = AnimationStrip(coords3d=coords3d)
    
    # Settings
    t0 = 0.
    dt = 0.1
    
    spawn_odds = 0.5*dt
    
    # Define objects
    objects = []
    
    print(objects)
    
    with utils.get_strip() as strip:
        animationloop(strip,anistrip,objects,
                spawn_func=spawn_fireworks,spawn_odds=spawn_odds,
                t0=t0,dt=dt)
                
                
                
def stuiterbal(coords3d:'coords.Coords3d'=None,
            anistrip:AnimationStrip=None):
    
    if coords3d is None:
        coords3d = coords.get_coords()
    if anistrip is None:
        anistrip = AnimationStrip(coords3d=coords3d)
    
    def force_func(self,dt):
        force = np.zeros(3)
        # force[0] = -2*self.x[0]
        # force[1] = -2*self.x[1]
        for i in range(2):
            if abs(self.x[i]) > 1.5:
                self.v[i] = -self.v[i]
        if self.x[2] < -1.5:
            # force[2] = -2*self.x[2]
            self.v[2] = -self.v[2]
        else:
            force[2] = -self.m*9.81
            
        # force += -0.05*self.v*np.abs(self.v)
        force[2] += -0.05*self.v[2]*np.abs(self.v[2])
        
        # if np.linalg.norm(self.v) > 3.:
            # v = 3.*self.v/np.linalg.norm(self.v)
            
        return force
        
    def spawn_stuiterbal(id,
        x0: np.ndarray(3)=None,v0: np.ndarray(3)=None,a0: np.ndarray(3)=None,
        v: float=None,
        color_on: colors.Color=None,
        which: str="Fade",mode: str="staythenlinear",duration: float=None,tfrac: float=None,# For instruction
        radius: float=None, max_distance: float=None,
        ):
        if color_on is None:
            # random color
            color_on = random_color()
        
        theid = "{0}_{1}".format(id,time.time())
        # print(theid)
        
        t0 = 0. # dummy t0
        
        duration = 0.5 if duration is None else duration
        tfrac = 0.5 if tfrac is None else tfrac
        which = "Fade" if which is None else which
        mode = "staythenlinear" if mode is None else mode
        instruction = AnimationInstruction(which=which,mode=mode,color=color_on,t0=t0,duration=duration)
        instr = instruction
        
        x0 = np.random.uniform(-1,1,3) if x0 is None else x0
        x0[1] = 0.
        
        v0 = np.random.uniform(-1,1,3) if v0 is None else v0
        v0[1] = 0.
        v =  np.random.uniform(2.,3.) if v is None else v
        v0 = v*v0/np.linalg.norm(v0)
        
        a0 = np.zeros(3) if a0 is None else a0
        
        radius = 0.25 if radius is None else radius
        
        ball = AnimationBall(x0,v0,a0,radius=radius,instr=instr,id=theid)
        ball.set_force_func(force_func)
        
        ball.set_termination_func(terminate_time_from_start)
        ball.t0 = 0.
        ball.max_t = 10.
        
        return [ball]
    
    # Settings
    t0 = 0.
    dt = 0.1
    
    spawn_odds = dt/10
    
    # Define objects
    objects = []
    for i in range(1):
        obj = spawn_stuiterbal(i)
        objects.extend(obj)
    print(objects)
    # input()
    with utils.get_strip() as strip:
        animationloop(strip,anistrip,objects,
                spawn_func=spawn_stuiterbal,spawn_odds=spawn_odds,
                t0=t0,dt=dt)
    
def planes(coords3d:'coords.Coords3d'=None,
            anistrip:AnimationStrip=None):
    
    if coords3d is None:
        coords3d = coords.get_coords()
    if anistrip is None:
        anistrip = AnimationStrip(coords3d=coords3d)
    
    
    def spawn_plane(id,
        x0: np.ndarray(3)=None,v0: np.ndarray(3)=None,a0: np.ndarray(3)=None,
        v: float=None,
        color_on: colors.Color=None,
        which: str=None,mode: str=None,duration: float=None,tfrac: float=None,# For instruction
        thickness: float=None,normal: np.ndarray(3)=None,
        ):
        if color_on is None:
            # random color
            color_on = random_color()
        
        theid = "{0}_{1}".format(id,time.time())
        # print(theid)
        
        t0 = 0. # dummy t0
        
        thickness = 0.15 if thickness is None else thickness
        normal = np.array([1,0,1.]) if normal is None else normal # normalised internally
        
        duration = 0.25 if duration is None else duration
        tfrac = 0.5 if tfrac is None else tfrac
        which = "stay" if which is None else which
        mode = "staythenlinear" if mode is None else mode
        instruction = AnimationInstruction(which=which,mode=mode,color=color_on,t0=t0,duration=duration)
        instr = instruction
        
        x0 = np.random.uniform(-1,1,3) if x0 is None else x0
        x0[1] = 0.
        
        v0 = normal if v0 is None else v0
        v0[1] = 0.
        v =  np.random.uniform(1.,2.) if v is None else v
        speed = np.linalg.norm(v0)
        if speed != 0.:
            v0 = v*v0/speed
        
        a0 = np.zeros(3) if a0 is None else a0
        
        
        plane = AnimationPlane(x0,v0,a0,thickness=thickness,normal=normal,instr=instr,id=theid)
        
        return [plane]
        
    def movingplane_force(self,dt):
        force = np.zeros(3)
        if abs(self.x[2]) > 2.:
            self.x[2] = -self.x[2]
        return force
        
    # Settings
    t0 = 0.
    dt = 0.1
    
    spawn_odds = dt/10
    
    # Define objects
    objects = []
    for i in range(5):
        objs = spawn_plane(i,x0=np.array([0,0,0.75*(i-3)]),v0=np.zeros(3))
        objects.extend(objs)
    objs = spawn_plane("MOVING",x0=np.array([0,0,0.75*(5.2-3)]),v0=np.array([0,0,-1]),which="Fade")
    objs[0].set_force_func( movingplane_force )
    objects.extend(objs)
    # input()
    with utils.get_strip() as strip:
        animationloop(strip,anistrip,objects,
                spawn_func=None,spawn_odds=spawn_odds,
                t0=t0,dt=dt)
    
    
def main():
    print("animate")
    print("nleds",config.nleds)
    
    planes()
    # fireworks()
    
if __name__ == "__main__":
    main()
