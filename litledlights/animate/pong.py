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
from misc_func import npunit,rotationmtx,xyz_to_rthetaphi
import time

import animate.animate as ani
AnimationStrip = ani.AnimationStrip
AnimationInstruction = ani.AnimationInstruction
AnimationObject = ani.AnimationObject
AnimationBall = ani.AnimationBall
AnimationPlane = ani.AnimationPlane

from pynput import keyboard # we be using this to capture keyz


class PlayerBat(AnimationObject):
    type = "ball"
    
    
    
    def __init__(self,*args,**kwargs):
        # print("ballinit")
        
        self.dim = kwargs.pop('dim',np.zeros(3))# dimensions as a box
        self.normal = kwargs.pop('normal',np.zeros(3))
        norm = np.linalg.norm(self.normal)
        if norm != 0.:
            self.normal = self.normal/norm
        
        super().__init__(*args,**kwargs)
    
    def select_lights(self,coords3d: coords.Coords3d):
        
        nml = self.normal#.transpose()
        
        shifted = coords3d.xyz - self.x
        shifted = shifted.transpose()
        
        # rotate ind2 onto the x-axis
        phi = np.arctan2(nml[1], nml[0])
        rotmatrix = rotationmtx(npunit(2),-phi)
        # print("Angle",phi)
        nml = np.dot( rotmatrix , nml )
        rotated = np.dot( rotmatrix , shifted )
        # rotate ind2 onto the z-axis
        phi = np.arctan2(nml[0],nml[2])
        rotmatrix = rotationmtx(npunit(1),-phi)
        # print("Angle",phi)
        rotated = np.dot( rotmatrix , rotated )
        nml = np.dot( rotmatrix , nml )
        
        
        indx1 = rotated[0] <  0.5*self.dim[0]
        indx2 = rotated[0] > -0.5*self.dim[0]
        
        indy1 = rotated[1] <  0.5*self.dim[1]
        indy2 = rotated[1] > -0.5*self.dim[1]
        
        indz1 = rotated[2] <  0.5*self.dim[2]
        indz2 = rotated[2] > -0.5*self.dim[2]
        
        ind = indx1 & indx2 & indy1 & indy2 & indz1 & indz2
        ind = np.where( ind )[0]
        
        
        return ind


def bat_force(self,dt):
    force = np.zeros(3)
    
    # little bit of drag
    force = -10*self.v*abs(self.v)
    
    if self.x[2] > self.border[2]:
        self.x[2] = self.border[2]
    if self.x[2] < -self.border[2]:
        self.x[2] = -self.border[2]
    
    return force

def spawn_bats(border):
    
    
    dim = [0.15,100,0.4]
    
    border_x,border_z = border[0],border[2]
    
    dx = 0.1
    
    
    def mk_bat_instruction(color_on):
        t0 = 0. # dummy t0
        duration = 0.15# if duration is None else duration
        tfrac = 0.5# if tfrac is None else tfrac
        which = "blink"# if which is None else which
        mode = "staythenlinear"# if mode is None else mode
        overwrite = True
        instruction = AnimationInstruction(which=which,mode=mode,color=color_on,t0=t0,duration=duration,overwrite=overwrite)
        instr = instruction
        
        return instr
        
        
    v0 = np.zeros(3)
    a0 = np.zeros(3)
    
    instr = mk_bat_instruction(colors.pink)
    
    
    x0 = np.array( [border_x-dx,0,0] )
    
    normal = np.zeros(3)
    normal[0] = 1.
    
    bat1 = PlayerBat(x0,v0,a0,instr=instr,dim=dim,normal=normal,id="BAT_RIGHT")
    
    x0 = np.array( [-border_x+dx,0,0] )
    bat2 = PlayerBat(x0,v0,a0,instr=instr,dim=dim,normal=normal,id="BAT_LEFT")
    
    
    
    
    objects = [bat1,bat2]
    
    return objects
    
    

def spawn_borders(border):
    
    objects = []
    
    def mk_instr(color_on):
        
        which = "stay"
        instruction = AnimationInstruction(which=which,color=color_on)
        instr = instruction
        return instr
    
    border_x = border[0] # abs x of border
    border_z = border[2] # abs z of border
    
    a0 = np.zeros(3)
    v0 = np.zeros(3)
    
    thickness = 0.5
    
    cnt =0
    
    
    ### RIGHT
    instr = mk_instr(colors.red)
    
    x0 = np.array([ border_x+thickness, 0, 0 ] )
    normal = np.array([-1,0,0]) # normalised internally
    
    theid = "{0}_{1}".format(cnt,time.time())
    cnt += 1
    plane = AnimationPlane(x0,v0,a0,thickness=thickness,normal=normal,instr=instr,id=theid)
    objects.append(plane)
    ### LEFT
    x0 = np.array([ -(border_x+thickness), 0, 0 ] )
    normal = np.array([1,0,0]) # normalised internally
    
    theid = "{0}_{1}".format(cnt,time.time())
    cnt += 1
    plane = AnimationPlane(x0,v0,a0,thickness=thickness,normal=normal,instr=instr,id=theid)
    objects.append(plane)
    
    
    ### TOP
    instr = mk_instr(colors.blue)
    
    x0 = np.array([ 0, 0, border_x+thickness ] )
    normal = np.array([0,0,-1]) # normalised internally
    
    theid = "{0}_{1}".format(cnt,time.time())
    cnt += 1
    plane = AnimationPlane(x0,v0,a0,thickness=thickness,normal=normal,instr=instr,id=theid)
    objects.append(plane)
    ### BOTTOM
    x0 = np.array([ 0, 0, -(border_z+thickness) ] )
    normal = np.array([0,0,1]) # normalised internally
    
    theid = "{0}_{1}".format(cnt,time.time())
    cnt += 1
    plane = AnimationPlane(x0,v0,a0,thickness=thickness,normal=normal,instr=instr,id=theid)
    objects.append(plane)
    
    
    
    
    return objects
    
    
def mk_ball_instruction(color_on):
    t0 = 0. # dummy t0
    duration = 0.25# if duration is None else duration
    tfrac = 0.5# if tfrac is None else tfrac
    which = "fade"# if which is None else which
    mode = "staythenlinear"# if mode is None else mode
    overwrite = True
    instruction = AnimationInstruction(which=which,mode=mode,
                color=color_on,
                t0=t0,duration=duration,
                overwrite=overwrite)
    instr = instruction
    
    return instr
    
    
def ball_force(self,dt):
    force = np.zeros(3)
    
    # if abs(self.x[0]) > self.border[0]:
        # self.v[0] = -self.v[0]
    if abs(self.x[2]) > self.border[2]:
        self.v[2] = -self.v[2]
    
    # Constantly accelerate a little bit
    
    return force

def ball_termination_func(self,t):
    
    if   self.x[0] > self.border[0]:
        return "score_right"
    elif self.x[0] < -self.border[0]:
        return "score_left"
    
    
    return False
        
def spawn_ball(border):
    
    
    theid = "{0}_{1}".format("BALL",time.time())
    
    x0 = np.random.uniform(-0.25,0.25,3)
    x0[1] = 0.
    v0 = np.random.uniform(-0.25,0.25,3)
    while abs(v0[0]/v0[2]) < 0.5: # ensure decent x-speed
        v0 = np.random.uniform(-0.25,0.25,3)
    v0[1] = 0.
    speed = np.linalg.norm(v0)
    v = np.random.uniform(0.5,1.5)
    if speed != 0.:
        v0 = v*v0/speed
    a0 = np.zeros(3)
    
    radius = 0.25
    
    instr = mk_ball_instruction(colors.gold)
    
    ball = AnimationBall(x0,v0,a0,radius=radius,instr=instr,id=theid)
    ball.border = border
    ball.set_force_func( ball_force )
    
    ball.set_termination_func( ball_termination_func )
    
    return [ball]



class PongGameObject(object):
    
    def __init__(self,objects,border,
                  bat_speed = 1.,
                  spawn_ball_dt: float=3):
        self.objects = objects
        self.border = border
        self.score = [0,0]
        
        self.bat_speed = bat_speed
        
        self.spawn_ball_dt = spawn_ball_dt
        self.spawn_ball_t0 = 0.
        self.spawn_new_ball = False
    
    def someone_scored(self,kill,obj,t):
        if kill == "score_left":
            self.score[1] += 1
            which_scores = "Right"
        if kill == "score_right":
            self.score[0] += 1
            which_scores = "Left"
        print("{0} Scored! Score: {1}".format(which_scores,self.score))
        
        self.objects.extend( ani.spawn_fireworks( str(self.score),x0=obj.x ) )
        self.spawn_ball_t0 = t
        self.spawn_new_ball = True
        
        for which in ['BAT_LEFT','BAT_RIGHT']:
            bat = self.get_object(which)
            if bat is not None:
                bat.x[2] = 0.
                bat.v = np.zeros(3)
    
    def check_ballbat_collision(self,ball):
        
        
        for which in ['BAT_LEFT','BAT_RIGHT']:
            bat = self.get_object(which)
            if bat is not None:
                collide = check_bar_sphere_collision(bat.x,bat.dim,bat.normal, ball.x,ball.radius)
                if collide:
                    return True
        return False    
            
        
    def move_bat(self, which, direction ):
        
        bat = self.get_object(which)
        if bat is None:
            return False
        if direction == 'up':
            thedir = 1
        elif direction == 'down':
            thedir = -1
        elif direction == 'stop':
            thedir = 0
        else:
            raise Exception("direction not up or down?")
        bat.v[2] = self.bat_speed*thedir
        # print("BAT SPEED",which,direction)
        
    def get_object(self, which):
        
        try:
            return [ o for o in self.objects if which in o.id ][0]
        except:
            return None
        
        
    def key_capture(self,key):
        # print(key)
        if key == keyboard.Key.up:
            self.move_bat('BAT_RIGHT','up')
        elif key == keyboard.Key.down:
            self.move_bat('BAT_RIGHT','down')
            
        
        return True
    
    def key_release(self,key):
        
        # print('{0} released'.format(key))
        
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        elif key == keyboard.Key.up or key == keyboard.Key.down:
            self.move_bat('BAT_RIGHT','stop')
        
        return True
        
        
def check_bar_sphere_collision(bar_x,bar_dim,bar_nml, ball_x,ball_radius):
    
    nml = bar_nml
    
    # rotate ball onto bat frame
    shifted = ball_x - bar_x
    
    # rotate ind2 onto the x-axis
    phi = np.arctan2(nml[1], nml[0])
    rotmatrix = rotationmtx(npunit(2),-phi)
    # print("Angle",phi)
    nml = np.dot( rotmatrix , nml ) # 
    rotated = np.dot( rotmatrix , shifted )
    # rotate ind2 onto the z-axis
    phi = np.arctan2(nml[0],nml[2])
    rotmatrix = rotationmtx(npunit(1),-phi)
    # print("Angle",phi)
    rotated = np.dot( rotmatrix , rotated )
    nml = np.dot( rotmatrix , nml )
    
    
    # Check if inside 
    indx1 = rotated[0] <  0.5*bar_dim[0]
    indx2 = rotated[0] > -0.5*bar_dim[0]
    
    indy1 = rotated[1] <  0.5*bar_dim[1]
    indy2 = rotated[1] > -0.5*bar_dim[1]
    
    indz1 = rotated[2] <  0.5*bar_dim[2]
    indz2 = rotated[2] > -0.5*bar_dim[2]
    
    ind = indx1 & indx2 & indy1 & indy2 & indz1 & indz2
    # ind = np.where( ind )[0]
    if ind:
        return True
    
    
    return False

        
        

def gameloop(strip: 'ledstrip.ledstrip', anistrip: AnimationStrip, ponggame: PongGameObject,
                  t0: float=0., dt: float=1.,
                  tmax: float=300,
                  
                  
                  idcnt: int=0):
    
    
    anistrip.t = t0
    
    # This didnt work because?
    listener = keyboard.Listener(
                        on_press=ponggame.key_capture,
                        on_release=ponggame.key_release)
    listener.start()
    print(listener)
    
    
    # Game Loop
    while True:
        time_startloop = time.time()
        # print("objects",t,objects)
        
        # Spawn new ball if needed
        if ponggame.spawn_new_ball:
            if anistrip.t - ponggame.spawn_ball_t0 > ponggame.spawn_ball_dt:
                ponggame.objects.extend( spawn_ball(ponggame.border) )
                ponggame.spawn_new_ball = False
                
        
        
        
        # Instruct lights
        for i,obj in enumerate(ponggame.objects):
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
        for i,obj in enumerate(ponggame.objects):
            force = obj.calculate_force(dt)
            flag = obj.update(dt,force)
            # print(obj.x)
            
            if "BALL" in obj.id:
                collision = ponggame.check_ballbat_collision(obj)
                if collision:
                    obj.v[0] = -obj.v[0] # reverse x
            elif "BAT" in obj.id: # Set velocity#_LEFT
                ball = ponggame.get_object("BALL")
                if ball is not None:
                    obj.v[2] = ponggame.bat_speed*np.sign(ball.x[2]-obj.x[2])
                
            kill = obj.evaluate_termination(anistrip.t)
            if kill:
                # print("KILL")
                ponggame.objects.pop(i)
                i += -1
                
                if type(kill) is str:
                    if "score" in kill:
                        ponggame.someone_scored(kill,obj,anistrip.t)
                
        
        # print("End loop, comp.time: {0} (dt: {1})".format(time.time()-time_startloop,dt))
        anistrip.t += dt
        time.sleep(max(0,dt-time.time()+time_startloop))
        
        if anistrip.t > tmax:
            print("Max t ({0}) reached".format(tmax))
            break
        
    listener.stop()
        
        
def main():
    print("pong")
    print("nleds",config.nleds)
    coords3d,anistrip = None,None
    if coords3d is None:
        coords3d = coords.get_coords()
    if anistrip is None:
        anistrip = AnimationStrip(coords3d=coords3d)
    
    # Settings
    t0 = 0.
    dt = 0.1
    
    border = [0.9,0,1.8] # abs vals of borders
    
    bat_speed = 1
    
    # Define objects
    objects = []
    
    objs = spawn_borders(border)
    objects.extend(objs)
    
    
    ball = spawn_ball(border)
    objects.extend(ball)
    
    bats = spawn_bats(border)
    objects.extend(bats)
    
    ponggame = PongGameObject(objects,border,
                        bat_speed=bat_speed)
    
    print(objects)
    # input()
    with utils.get_strip() as strip:
        gameloop(strip,anistrip,ponggame,
                t0=t0,dt=dt)
    
    
    
    
if __name__ == "__main__":
    main()
