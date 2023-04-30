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

import animate.animate as ani
AnimationStrip = ani.AnimationStrip
AnimationInstruction = ani.AnimationInstruction
AnimationObject = ani.AnimationObject
AnimationBall = ani.AnimationBall
AnimationPlane = ani.AnimationPlane

import time
import sys

import sounddevice as sd
import queue



class DeviceClass():
    """Class for a sounddevice"""
    
    def __init__(self,device=None,
                    channels=[1],samplerate=None,
                    callback: Callable=None,
                    downsample: int=1):
                    
        self.device = device
        self.device_info = sd.query_devices(device, 'input')
        
        self.channels = channels
        self.samplerate = samplerate if samplerate is not None else self.device_info['default_samplerate']
        
        self.callback = callback if callback is not None else self.audio_callback
        
        self.mapping = [ c-1 for c in channels ]
        
        self.downsample = downsample # Watch out with downsample and FFT-kinda-stuff
        
        self.q = queue.Queue()

    def get_stream(self, callback: Callable=None, blocksize: int=None):
        # is blocksize really int?
        if callback is None:
            callback = self.callback
        stream = sd.InputStream(
                    device=self.device, channels=max(self.channels),
                    samplerate=self.samplerate, callback=callback,
                    blocksize=blocksize)
                    
        return stream
        
        
    def audio_callback(self,
                indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        self.q.put(indata[::self.downsample, self.mapping])



def listenloop(strip,anistrip,
                device: 'sd.device'=None,stream: 'sd.InputStream'=None,soha: 'SoundHandle'=None,
                spawn_func: Callable=None,
                objects: list=[],
                t0: float=0, dt: float=0.1, tmax: float=300):
    
    anistrip.t = t0
    
    while True:
        time_startloop = time.time()
        
        if spawn_func is not None:
            if spawn_func(soha):
                objs = ani.spawn_fireworks(str(anistrip.t))
                objects.extend(objs)
                print("YEAH")
        
        # Instruct lights
        for i,obj in enumerate(objects):
            ind = obj.select_lights(anistrip.coords3d)
            # print(ind)
            anistrip.instruct(ind,obj.instruction)
        
        
        # Render
        states = anistrip.render(anistrip.t)
        
        strip.set_all(states)
        strip.show()
        
        # Update sound buffer
        soha.update_yarr()
        
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
        
        print("End loop, comp.time: {0} (dt: {1})".format(time.time()-time_startloop,dt))
        anistrip.t += dt
        time.sleep(max(0,dt-time.time()+time_startloop))
        
        if anistrip.t > tmax:
            print("Max t ({0}) reached".format(tmax))
            break


def equalizerloop(strip,anistrip,
                device: 'sd.device'=None,stream: 'sd.InputStream'=None,equalizer: 'EqualizerClass'=None,
                spawn_func: Callable=None,
                objects: list=[],
                t0: float=0, dt: float=0.1, tmax: float=300):
    
    anistrip.t = t0
    
    while True:
        time_startloop = time.time()
        
        if spawn_func is not None:
            if spawn_func(soha):
                objs = ani.spawn_fireworks(str(anistrip.t))
                objects.extend(objs)
                print("YEAH")
        
        # Equalizer Stuff
        equalizer.calculate_fft()
        
        
        # Instruct lights
        for i,obj in enumerate(objects):
            ind = obj.select_lights(anistrip.coords3d)
            # print(ind)
            anistrip.instruct(ind,obj.instruction)
        
        
        # Render
        states = anistrip.render(anistrip.t)
        
        strip.set_all(states)
        strip.show()
        
        # Update sound buffer
        equalizer.update_yarr()
        
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
        
        print("End loop, comp.time: {0} (dt: {1})".format(time.time()-time_startloop,dt))
        anistrip.t += dt
        time.sleep(max(0,dt-time.time()+time_startloop))
        
        if anistrip.t > tmax:
            print("Max t ({0}) reached".format(tmax))
            break

class SoundHandle(object):
    
    def __init__(self,
            device=None,
            window: int=None):
                
        self.device = device
        
        self.window = 1000 # ms
        
        
        self.length = None
        self.xarr,self.yarr = None,None
        
        if device is not None:
            self.length = int(window * device.samplerate / (1000 * device.downsample))
            
            self.xarr = np.arange(self.length)
            self.yarr = np.zeros((self.length, len(device.channels)))
    

    def update_yarr(self,device=None,yarr=None):
        if device is None:
            device = self.device
        if yarr is None:
            yarr_was_none = True
            yarr = self.yarr
        else:
            yarr_was_none = False
        
        while True:
            try:
                data = device.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            yarr = np.roll(yarr, -shift, axis=0)
            yarr[-shift:, :] = data
            # print(len(data))
        if yarr_was_none:
            self.yarr = yarr
        
        return yarr

def clap_for_fireworks():
    
    def check_if_spawn_fireworks(soha):
        if np.any(np.abs(soha.yarr)>soha.spawnfireworks_cutoff):
            return True
        return False
    
    print("SOUND")
    print("nleds",config.nleds)
    coords3d,anistrip = None,None
    if coords3d is None:
        coords3d = coords.get_coords()
    if anistrip is None:
        anistrip = AnimationStrip(coords3d=coords3d)
    
    # Settings
    t0 = 0.
    dt = 0.1
    
    
    # Define objects
    objects = []
    
    
    # Input device
    device = DeviceClass()
    stream = device.get_stream()
    
    window = 200 # ms
    soha = SoundHandle(device,window=window)
    
    # Some settings
    soha.spawnfireworks_cutoff = 0.02
    
    print(objects)
    # input()
    with stream:
        with utils.get_strip() as strip:
            listenloop(strip,anistrip,
                    device=device,stream=stream,soha=soha,
                    spawn_func=check_if_spawn_fireworks,
                    objects=objects,
                    t0=t0,dt=dt)
    
class EqualizerClass(SoundHandle):
    
    def __init__(self,*args,**kwargs):
        
        self.gain = kwargs.pop('gain',1.)
        
        self.freqrange = kwargs.pop('freqrange',[100, 2000]) # Hz
        
        self.nbins = kwargs.pop('nbins',10)
        
        # Process things like device etc
        super().__init__(*args,**kwargs)
        
        
        if self.device.downsample != 1:
            print("WARNING: device.downsample is not 1, something here will go wrong in calculating frequency range")
            sys.exit()
        
        self.delta_f = (self.freqrange[1] - self.freqrange[0]) / (self.nbins- 1)
        self.fftsize = int(np.ceil(self.device.samplerate / (self.device.downsample*self.delta_f)))
        self.low_bin = int(np.floor(self.freqrange[0] / self.delta_f)) # index of loewst bin
        
        print(self.nbins,self.fftsize)
        
        self.fft_freq = np.fft.rfftfreq(self.fftsize,d=self.device.downsample/self.device.samplerate)
    
    def calculate_fft(self):
        
        self.fft_res = np.abs(np.fft.rfft(self.yarr[:, 0], n=self.fftsize))
        self.fft_res *= self.gain / self.fftsize
        import matplotlib.pyplot as plt
        # plt.plot(self.fft_freq,self.fft_res)
        plt.plot(self.fft_freq[self.low_bin:self.low_bin + self.nbins],self.fft_res[self.low_bin:self.low_bin + self.nbins])
        # plt.plot(self.yarr)
        plt.show(block=False)
        plt.pause(0.05)
        
    
def main():
    
    
    print("SOUND")
    print("nleds",config.nleds)
    coords3d,anistrip = None,None
    if coords3d is None:
        coords3d = coords.get_coords()
    if anistrip is None:
        anistrip = AnimationStrip(coords3d=coords3d)
    
    # Settings
    t0 = 0.
    dt = 0.1
    
    
    # Define objects
    objects = []
    
    
    # Input device
    device = DeviceClass()
    stream = device.get_stream()
    
    window = 200 # ms
    equalizer = EqualizerClass(device,window=window)
    
    # Some settings
    
    print(objects)
    # input()
    try:
        with stream:
            with utils.get_strip() as strip:
                equalizerloop(strip,anistrip,
                        device=device,stream=stream,equalizer=equalizer,
                        objects=objects,
                        t0=t0,dt=dt)
    except KeyboardInterrupt:
        pass
    
    
if __name__ == "__main__":
    main()





