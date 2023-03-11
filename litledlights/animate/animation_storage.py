
def spawn_ball(x0,v0,a0,radius,instr,id):
    return AnimationBall(x0,v0,a0,radius=radius,instr=instr,id=id)
    
def main():
    print("animate")
    print("nleds",config.nleds)
    
    
    coords3d = coords.get_coords()
    
    anistrip = AnimationStrip(coords3d=coords3d)
    
    # Prepare instructon
    color_on = colors.Color((115,0,0))
    duration = 2.
    t0 = 0.
    tfrac = 0.5
    # instruction = AnimationInstruction(which="Fade",mode="linear",color=color_on,t0=t0,duration=duration)
    instruction = AnimationInstruction(which="Fade",mode="staythenlinear",color=color_on,t0=t0,duration=duration,tfrac=tfrac)
    # print(instr)
    
    
    
    # Settings
    dt = 0.1
    
    # Define objects
    radius = 0.15
    x0,v0,a0 = misc_func.npunit(2),-4.*misc_func.npunit(2),np.zeros(3)
    
    
    objects = []
    def i_to_color(i,j):
        if   i == 0:
            if j == 0:
                return (0,115,0)
            return (115,0,35)
        elif i == 1:
            if j == 0:
                return (0,0,115)
            return (0,115,0)
        else:
            if j == 0:
                return (115,0,0)
            return (0,0,115)
    
    idcnt = 0
    for i in range(3):
        # FIRST
        id = str(idcnt)
        idcnt += 1
        
        x,v = x0.copy(),v0.copy()
        instr = instruction.copy()
        
        instr['color'] = colors.Color( i_to_color(i,0) )
        x[0] = 0.75*(i-1)
        objects.append(spawn_ball(x,v,a0,radius,instr,id))
        
        # SECOND
        id = str(idcnt)
        idcnt += 1
        
        x,v = x.copy(),v.copy()
        instr = instr.copy()
        
        
        c = i_to_color(i,1)
        c = (235-c[0],235-c[1],235-c[2])
        instr['color'] = colors.Color( c )
        x[2] = -x[2]
        v[2] = -v[2]
        objects.append(spawn_ball(x,v,a0,radius,instr,id))
    
    print(objects)
    
    with utils.get_strip() as strip:
        animationloop(strip,anistrip,objects,t0=t0,dt=dt)
