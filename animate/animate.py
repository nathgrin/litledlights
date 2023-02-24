
import config
import colors

class AnimationInstruction(dict):
    
    def __init__(self):
        pass

class AnimationStrip(list):
    
    def __init__(self,nleds: int=None,initial_state: tuple[int]=(0,0,0)):
        self.nleds = nleds if nleds is not None else config.nleds
        print(nleds)
        for i in range(self.nleds):
            self.append(AnimationLed(state=initial_state))

    def add_led(self):
        self.append(led)
    

    def render(self,t: float):
        
        return [ led.render(t) for led in self ]

class AnimationLed(list):
    
    def __init__(self,state: tuple[int]=(0,0,0)):
        self.state = state
    def __repr__(self):
        return str(self.state)
    
    def instruct(self,instruction: AnimationInstruction):
        self.append(instruction)
    
    def render(self,t: float):
        # Something with instruction
        return self.state

    

def main():
    print("animate")
    print(config.nleds)
    
    anistrip = AnimationStrip()
    
    delta_t = 5.
    tstart = 0.
    instr = AnimationInstruction(type="fade",mode="linear",color=colors.red,tstart=tstart,delta_t=delta_t)
    print(instr)
    
    t = 0
    dt = 0.01
    
    while True:
        t += dt
        
        anistrip.render(t)
        
        
        print(anistrip)
        
    
    
if __name__ == "__main__":
    main()