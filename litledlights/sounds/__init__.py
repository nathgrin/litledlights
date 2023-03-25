#!/usr/bin/env python3
import numpy as np
import sounddevice as sd

sd.query_devices()

input("AA")

duration = 10 #in seconds

def audio_callback(indata, frames, time, status):
   volume_norm = np.linalg.norm(indata) * 10
   print("|" * int(volume_norm))


stream = sd.InputStream(callback=audio_callback)
with stream:
   sd.sleep(duration * 1000)
