import config

import sys
sys.path.insert(0,"..")

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


import threading

from calibrate.findlights import find_light, reprocess,load_neuralnet
from calibrate.triangulate import combine_coords_2d_to_3d

import colors

import os
import datetime

try:
    from utils import get_strip
except:
    print("import failed, setting config.dbg=True")
    config.dbg = True

from misc_func import npunit,rotationmtx,xyz_to_rthetaphi

import FindLightApp as FLA



def coords2d_write(fname: str, coords2d: list[tuple[float,float]])->None:
    
    np.savetxt(fname,coords2d)

def coords2d_read(fname: str) -> list[tuple[float,float]]: 
    
    out = np.loadtxt(fname)
    return out

class WebcamVideoStream:
    def __init__(self, camname="Webcam", src=0, grayscale=False):
        self.camname = camname
        self.grayscale = grayscale
        
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.flag, self.frame) = self.stream.read()
        if self.flag:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        
        # start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
   
    def start(self):
        self.thread.start()
        
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            if self.stream.isOpened():
                (self.flag, self.frame) = self.stream.read()
                if self.flag:
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    def read(self):
        # return the frame most recently read
        return self.flag,self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    def exit(self):
        self.stop()
        self.stream.release()
        

def get_coords2d_from_doublecam():
    """2 cams: star and moon"""
    
    coords2d1,coords2d2 = initiate_sequential_fotography()
    
    
    

def initiate_sequential_fotography(loc: str=None,skip_to_reprocess: bool=None):
    loc = config.sequentialfotography_loc if loc is None else loc
    
    
    ok = False
    do_reprocess = config.sequentialfotography_skiptoreprocess if skip_to_reprocess is None else skip_to_reprocess
    do_fixnans = False
    while not ok:
        if do_reprocess:
            coords2d = reprocess(loc=loc)
            do_reprocess = False
        elif do_fixnans:
            # coords2d = coords2d_fix_nans_byhand(coords2d,loc=loc)
            print("FIX NANS NOT IMPLEMENTED")
            do_fixnans = False
        else:
            coords2d1,coords2d2 = sequential_fotography_doublecam(loc=loc)
        
        # print(coords2d)
        for i,coords2d in enumerate(coords2d1,coords2d2):
            print("doing coords2d",i)
            if coords2d is not None:
                # print(coords2d)
                print("NaN/tot: {}/{}".format(np.sum(np.isnan(coords2d))//2,len(coords2d))) # divide by 2 because counts x&y nan-values
                
                # img_bg = cv2.imread(os.path.join(loc,"background.png"))
                # for i in range(len(coords2d)):
                    # img_bg = cv2.putText(img_bg,str(i),coords2d[i],cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                # cv2.imshow("Background with found lights",img_bg)
                fname = os.path.join(loc,"coords2d%i_tmp.txt"%(i))
                print("coords2d saved in tmpfile:",fname)
                coords2d_write(fname,coords2d)
            
            print("You happy? Enter to accept, p to reprocess, h to redo nans by hand, anything else to redo")
            theinput = input("")
            # k = cv2.waitKey(0)
            
            ok = theinput == ""#k%256 == 10
            do_reprocess = theinput == "p"
            do_fixnans = theinput == "h"
            
        else:
            ok = False
        
        if do_reprocess:
            print("Lets reprocess..")
        elif do_fixnans:
            print("Lets Fix nans by hand")
        elif not ok:
            print("Not happy, try again")
        else:
            print(" > Happy!")
    return coords2d1,coords2d2
    
    
def sequential_fotography_doublecam(strip=None,
                            color_off = (0,0,0),
                            color_on: tuple[int,int,int] = None,
                            
                            delta_t: int = None,# in arbitrary units
                            loc: str=None,
                            
                            grayscale: bool = None,
                            save_images: bool = None,
                            
                            do_findlight: bool = None
                            ) -> np.ndarray:
    """example from stackoverflow, in turn stolen from the "docs" 
    
    like matt parker does it. 
    Turn on each light in sequence and 
    
    but with doublecam!
    
    """
    
    
    
    help_msg = "Press h for help,\n space to start or Pause,\n b for new background image,\n f to toggle background subtract of preview"
    
    # kwargs
    color_on = config.sequentialfotography_coloron if color_on is None else color_on
    delta_t = config.sequentialfotography_deltat if delta_t is None else delta_t
    loc = config.sequentialfotography_loc if loc is None else loc
    grayscale = config.sequentialfotography_grayscale if grayscale is None else grayscale
    do_findlight = config.sequentialfotography_dofindlight if do_findlight is None else do_findlight
    save_images = config.sequationalfotography_saveimages if save_images is None else save_images
    
    # Strip
    strip = get_strip() if strip is None else strip
    strip.fill( color_off )
    strip.show()
    
    
    # Cams
    ind_star = 0 # TODO put this in config
    ind_moon = 2
    stream_moon = WebcamVideoStream("Moon", ind_moon,grayscale)
    stream_star = WebcamVideoStream("Star", ind_star,grayscale)
    if stream_moon.stream is None or not stream_moon.stream.isOpened():
       raise BufferError('Error: unable to open video source (moon):', ind_moon)
    if stream_star.stream is None or not stream_star.stream.isOpened():
       raise BufferError('Error: unable to open video source (star):', ind_star)
    
    # Findlight
    findlight_kwargs = {}
    if config.findlight_method == "neuralnet":
        findlight_neuralnet = load_neuralnet(config.findlight_neuralnet_fname)
        findlight_kwargs['nnmodel'] = findlight_neuralnet
        print("Initiating neuralnet")
        if stream_moon.flag:
            find_light(stream_moon.img,findlight_kwargs)
        
    elif config.findlight_method == "simplematt":
        findlight_threshold = config.findlight_threshold# if findlight_threshold is None else findlight_threshold
        findlight_kwargs['threshold'] = findlight_threshold
        
    
    # BG img
    ret,img_bg_moon = stream_moon.read()
    ret,img_bg_star = stream_star.read()
    if grayscale:
        img_bg_moon = cv2.cvtColor(img_bg_moon, cv2.COLOR_BGR2GRAY)
        img_bg_star = cv2.cvtColor(img_bg_star, cv2.COLOR_BGR2GRAY)
    
    # Prep params
    nleds = len(strip)
    ind = -1
    t = 0
    
    started = False
    preview_subtract = True
    
    
    coords2d_moon = [ (np.nan,np.nan) for x in range(nleds) ]
    coords2d_star = [ (np.nan,np.nan) for x in range(nleds) ]
    
    start = time.time()
    
    print(" >",help_msg)
    
    
    try:
        
        stream_moon.start()
        stream_star.start()
        
        cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
        
        while True:
            # cv2.imshow("Stream2",stream_star.frame)
            
            ret,img_moon = stream_moon.read()
            ret,img_star = stream_star.frame
            
            if preview_subtract:
                preview_moon = cv2.subtract(img_moon,img_bg_moon)
                preview_star = cv2.subtract(img_star,img_bg_star)
            else:
                preview_moon = img_moon
                preview_star = img_star
                
            # Preview
            sidebyside = np.hstack((preview_moon,preview_star))
            cv2.imshow("Stream",sidebyside)

            # now, subtract bg anyway
            img_moon = cv2.subtract(img_moon,img_bg_moon)
            img_moon = cv2.subtract(img_moon,img_bg_moon)
            
            # loop
            if started:
                t += 1
            if t % delta_t == 0:
                t = 0
                
            # WAITKEY
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print(" > Escape hit, closing...")
                coords2d = None
                break
            elif k == ord('h'):
                print(help_msg)
            elif k == ord('f'):
                preview_subtract = not preview_subtract # toggle
                print("preview subtract turned",preview_subtract)
            elif k == ord('b'):
                # hit b
                # update background img
                print("update background..")
                
                # BG img
                ret,img_bg_moon = stream_moon.read()
                ret,img_bg_star = stream_star.read()
                if grayscale:
                    img_bg_moon = cv2.cvtColor(img_bg_moon, cv2.COLOR_BGR2GRAY)
                    img_bg_star = cv2.cvtColor(img_bg_star, cv2.COLOR_BGR2GRAY)
                
                if save_images:
                    cv2.imwrite(img_name, img_bg)
                
            elif k%256 == 32:
                # SPACE pressed
                
                if started:
                    print(" > PAUSE at ind {0}".format(ind))
                    started = False
                else:
                    print(" > Vamonos")
                    # t = 0 # reset t because we can
                    started = True
                
            if started and t%delta_t == 0: # Interlacing turning on/off lights and cam picture
                # coords2d[ind] = (np.nanmedian(xlist),np.nanmedian(ylist))
                print("   Done",coords2d_moon[ind],coords2d_star[ind])
                
                # xlist = []
                # ylist = []
                
                
                
                # print(t,started)
                ind += 1
                print(' - Led',ind)
                if ind == nleds:
                    print(" > We got em all")
                    break
                
                # reset
                strip[ind-1] = color_off
                # strip[ind+1]   = color_on
                # strip.show()
                # strip[ind] = color_on
                strip.show()
            #elif started and (t+delta_t//2)%delta_t == 0: # Interlacing turning on/off lights and cam picture
            
            elif started and t == 1 and ind % 50 == 0:
                print("(auto) update background..")
                
                # BG img
                ret,img_bg_moon = stream_moon.read()
                ret,img_bg_star = stream_star.read()
                if grayscale:
                    img_bg_moon = cv2.cvtColor(img_bg_moon, cv2.COLOR_BGR2GRAY)
                    img_bg_star = cv2.cvtColor(img_bg_star, cv2.COLOR_BGR2GRAY)
                if save_images:
                    img_name = os.path.join(loc,"led_{}background.png".format(ind))
                    cv2.imwrite(img_name, img_bg)
                
                # time.sleep(0.25)
                
            elif started and t == 2:
                
                strip[ind] = color_on
                strip.show()
                
                # time.sleep(0.25)
                
                
            elif started and t >2:#= delta_t//2 -1 and t <= delta_t//2+1: #t >= delta_t//3 and t <= 2*delta_t//3:#(t+delta_t//2)%delta_t == 0:#
                
                if (t+delta_t//2)%delta_t == 0 and save_images:
                    img_name = os.path.join(loc,"led_{}.png".format(ind))
                    cv2.imwrite(img_name, frame)
                    print("   {} written!".format(img_name))
                
                lower_level = 0.2 # lower level fraction of max level
                
                # The 2 here is the 2 from t>2 above here.
                factor = (t-2)*(lower_level-1.)/(delta_t-2.) + 1.     #t*(lower_level*n_pieces-n_pieces)/((n_pieces-2)*delta_t)+1-(lower_level-1)/(n_pieces-2)
                strip[ind] = (factor*color_on[0],factor*color_on[1],factor*color_on[2])
                strip.show()
            
                if do_findlight:
                    xy_moon = find_light(img_moon,**findlight_kwargs)
                    if not np.isnan(xy_moon).any():
                        coords2d_moon[ind] = xy
                    xy_star = find_light(img_star,**findlight_kwargs)
                    if not np.isnan(xy_star).any():
                        coords2d_star[ind] = xy
                    
                    # if not np.isnan(xy_moon).any() and not np.isnan(xy_star).any():
                    if not np.isnan(coords2d_moon[ind]).any() and not np.isnan(coords2d_star[ind]).any():
                        t = delta_t-1
                        # print("   Done",t,coords2d[ind])
                else:
                    xy = None
                    
                
                print("   ",t,"%.02f"%(time.time()-start),xy_moon,xy_star)
                # time.sleep(1)
              
        print()
        print("Active threads", threading.activeCount())
    finally:
        stream_moon.exit()
        stream_star.exit()
        # thread2.exit()
        cv2.destroyAllWindows()
    
    
    return coords2d_moon,coords2d_star
    

def main():
    
    loc = config.sequentialfotography_loc
    
    # From camera calibatrion
    distortions = config.distortions
    camera_matrix = config.camera_matrix
    new_camera_matrix = config.new_camera_matrix

    
    coords2d_list = None
    if config.getcoords2d_fromangles and config.connect_ledlights and not config.dbg:
        coords2d_list = get_coords2d_from_doublecam()
        coords2d_list = [c2d.transpose() for c2d in coords2d_list]
    
    
    if coords2d_list is None:
        coords2d_list = []
        for i in range(n_viewpoints):
            fname = os.path.join(loc,"coords2d_{}.txt".format(i))
            coords2d_list.append( coords2d_read(fname) )
            
    
    coords2d1,coords2d2 = coords2d_list[0],coords2d_list[1]
    
    print(" > Iterative 2d to 3d")
    coords3d = iterative_pair_coords2d_to_coords3d(coords2d1,coords2d2,
                                   camera_matrix=camera_matrix,distortions=distortions, new_camera_matrix=new_camera_matrix)
    
    if coords3d is not None:
        show_coords_onlights(coords3d)
    
    if config.save_coords3d and coords3d is not None:
        print(" > Saving coords",config.savecoords3d_fname)
        np.savetxt(config.savecoords3d_fname,coords3d)
    
    
if __name__ == "__main__":
    main()
