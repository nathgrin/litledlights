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
    def __init__(self, camname:str, src, grayscale: bool=False, resolution: tuple[int,int] = None):
        self.camname = camname
        self.grayscale = grayscale
        
        self.src = src
        
        self.cap = None
        
        
        # RESIZE but not by direct call ..
        self.cap_width,self.cap_height = config.webcam_resolution if resolution is None else resolution # _grab will resize
        # self.cap_width,self.cap_height = 160,120
        # self.cap_width,self.cap_height = 640,480
        self.do_resize = False # will check in initiate whether to do this or not
        
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        
        
        
        # start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        
        
        
    def start(self):
        self.initiate()
        self.thread.start()
        
        
   
    def initiate(self):
        
        check = False
        
        while not check: # Apparantly this loop is uselss and does nothing
            print("Initiate cam ",self.camname)
            # initialize the video camera stream and read the first frame
            # from the stream
            self.cap = cv2.VideoCapture(self.src)
            # assert self.cap.isOpened()
            # ret_val , frame = self.cap.read() # call the camera once does htis help with the set?
            # self.cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY) # only for lib4vl backend
            self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
            # self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('Y','U','Y','V'))
            
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)#320)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)#240)
            
            # self.cap.set(cv2.CAP_PROP_FPS,10.)
            
            check = self.cap.isOpened()
            if self.cap.isOpened():
                # (self.flag, self.frame) = self.cap.read()
                # flag, frame = self.cap.read()
                
                # self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self._grab()
                
                check = self.flag
                
                
                if self.flag:
                    cap_width,cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.do_resize = not (cap_width == self.cap_width and cap_height == self.cap_height)
                
                # print(self.camname,"WIDTH,HEIGHT",self.cap_width,self.cap_height,self.cap.get(cv2.CAP_PROP_FPS))
            if not check:
                self.cap.release()
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.exit()
                return
            # otherwise, read the next frame from the stream
            self._grab()
    
    def _grab(self):
        if self.cap.isOpened():
            flag, frame = self.cap.read()
            if flag:
                if self.do_resize:
                    frame = cv2.resize(frame,(self.cap_width,self.cap_height))
                if self.grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                self.flag = flag
                self.frame = frame
                
            # else:
                # self.restart_cam()
    
    
    def restart_cam(self):
        self.cap.release()
        self.initiate()
    
    def read(self):
        # return the frame most recently read
        if not self.flag:
            # print(self.camname,"FLAG",self.flag)
            # self.restart()
            if self.grayscale: # NOTE order of cap_width and height
                out = np.zeros((self.cap_height,self.cap_width,1))
            else:
                out = np.zeros((self.cap_height,self.cap_width,3))
        else:
            out = self.frame
        # print(self.camname,out.shape)
        return self.flag,out
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    def exit(self):
        self.stop()
        if self.cap is not None:
            self.cap.release()
        

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
            # simple_sequential_fotography_doublecam(loc=loc)
        
        # print(coords2d)
        for i,coords2d in enumerate((coords2d1,coords2d2)):
            print("doing coords2d",i)
            if coords2d is not None:
                if np.sum(np.isnan(coords2d)) == len(coords2d):
                    print( "ALL nan!, skip.")
                    continue
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
    
    
def simple_sequential_fotography_doublecam(strip=None,
                        loc: str="_tmp",
                            ) -> np.ndarray:
                                
    
    grayscale = False
    
    # Cams
    ind_star = config.CAMERA_ind_star
    ind_moon = config.CAMERA_ind_moon
    stream_moon = WebcamVideoStream("Moon", ind_moon)
    stream_star = WebcamVideoStream("Star", ind_star)
    stream_moon.start()
    stream_star.start()
    # if stream_moon.cap is None or not stream_moon.cap.isOpened():
       # raise BufferError('Error: unable to open video source (moon):', ind_moon)
    # if stream_star.cap is None or not stream_star.cap.isOpened():
       # raise BufferError('Error: unable to open video source (star):', ind_star)
    
    try:
        
        
        cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
        
        while True:
            # cv2.imshow("Stream2",stream_star.frame)
            
            ret,img_moon = stream_moon.read()
            ret,img_star = stream_star.read()
            # img_moon = stream_moon.frame
            # img_star = stream_star.frame
            
            # print(img_moon)
            # print(img_star)
            
            preview_moon = img_moon
            preview_star = img_star
            
            print("Shape",preview_moon.shape,preview_star.shape)
                
            # Preview
            sidebyside = np.hstack((preview_moon,preview_star))
            cv2.imshow("Stream",sidebyside)
            
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    
    finally:
        stream_moon.exit()
        stream_star.exit()
        # thread2.exit()
        cv2.destroyAllWindows()


    return None,None
    
def find_light_func(dst,img,findlight_kwargs):
    res = find_light(img,**findlight_kwargs)
    dst[0] = res[0]
    dst[1] = res[1]
    return dst
    
    
def sequential_fotography_doublecam(strip=None,
                            color_off = (0,0,0),
                            color_on: tuple[int,int,int] = None,
                            
                            delta_t: int = None,# in arbitrary units
                            loc: str=None,
                            
                            bg_autoupdate_every: int = None,
                            
                            grayscale: bool = None,
                            save_images: bool = None,
                            
                            doublecam_mode: int = None,
                            
                            do_findlight: bool = None
                            ) -> np.ndarray:
    """example from stackoverflow, in turn stolen from the "docs" 
    
    like matt parker does it. 
    Turn on each light in sequence and 
    
    but with doublecam!
    
    """
    
    
    
    help_msg = "Press h for help,\n space to start or Pause,\n b for new background image,\n f to toggle background subtract of preview\n p to toggle update_preview"
    # help_msg += "\n d for doublecam_mode (0: Moon, 1: Star, 2: Both)"
    
    # kwargs
    color_on = config.sequentialfotography_coloron if color_on is None else color_on
    delta_t = config.sequentialfotography_deltat if delta_t is None else delta_t
    loc = config.sequentialfotography_loc if loc is None else loc
    grayscale = config.sequentialfotography_grayscale if grayscale is None else grayscale
    do_findlight = config.sequentialfotography_dofindlight if do_findlight is None else do_findlight
    save_images = config.sequentialfotography_saveimages if save_images is None else save_images
    doublecam_mode = config.sequentialfotography_doublecammode if doublecam_mode is None else doublecam_mode
    bg_autoupdate_every = config.sequentialfotography_bg_autoupdate_every if bg_autoupdate_every is None else bg_autoupdate_every
    
    doublecam_do_moon = doublecam_mode == 0 or doublecam_mode == 2
    doublecam_do_star = doublecam_mode == 1 or doublecam_mode == 2
    
    # Strip
    strip = get_strip() if strip is None else strip
    strip.fill( color_off )
    strip.show()
    
    
    # Cams
    ind_moon = config.CAMERA_ind_moon
    ind_star = config.CAMERA_ind_star
    # MOON
    stream_moon = WebcamVideoStream("Moon", ind_moon,grayscale)
    if doublecam_do_moon:
        stream_moon.start()
        if stream_moon.cap is None or not stream_moon.cap.isOpened():
           raise BufferError('Error: unable to open video source (moon):', ind_moon)
    # STAR
    stream_star = WebcamVideoStream("Star", ind_star,grayscale)
    if doublecam_do_star:
        stream_star.start()
        if stream_star.cap is None or not stream_star.cap.isOpened():
           raise BufferError('Error: unable to open video source (star):', ind_star)
        
    # Findlight
    findlight_kwargs = {}
    if config.findlight_method == "neuralnet":
        findlight_neuralnet = load_neuralnet(config.findlight_neuralnet_fname)
        findlight_kwargs['nnmodel'] = findlight_neuralnet
        print("Initiating neuralnet")
        if doublecam_do_moon:
            if stream_moon.flag:
                find_light(stream_moon.read()[1],**findlight_kwargs)
        elif doublecam_do_star:
            if stream_star.flag:
                find_light(stream_star.read()[1],**findlight_kwargs)
        
    elif config.findlight_method == "simplematt":
        findlight_threshold = config.findlight_threshold# if findlight_threshold is None else findlight_threshold
        findlight_kwargs['threshold'] = findlight_threshold
        
    
    # BG img
    if doublecam_do_moon:
        ret,img_bg_moon = stream_moon.read()
    if doublecam_do_star:
        ret,img_bg_star = stream_star.read()
    
    # Prep params
    nleds = len(strip)
    ind = -1
    t = 0
    
    started = False
    preview_subtract = True
    update_preview = True
    
    
    coords2d_moon = [ (np.nan,np.nan) for x in range(nleds) ]
    coords2d_star = [ (np.nan,np.nan) for x in range(nleds) ]
    
    # if grayscale:
        # img_bg_moon = np.zeros((stream_moon.cap_height,stream_moon.cap_width))
    # else:
        # img_bg_star = np.zeros((stream_star.cap_height,stream_star.cap_width,3))
    
    start = time.time()
    
    print(" >",help_msg)
    
    
    try:
        
        
        cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
        
        while True:
            # cv2.imshow("Stream2",stream_star.frame)
            
            if doublecam_do_moon:
                ret,img_moon = stream_moon.read()
            if doublecam_do_star:
                ret,img_star = stream_star.read()
            
            if update_preview:
                if preview_subtract:
                    # print("MOON",img_moon.shape ,"BG",img_bg_moon.shape)
                    # print("STAR",img_star.shape ,"BG",img_bg_star.shape)
                    if doublecam_do_moon:
                        preview_moon = cv2.subtract(img_moon,img_bg_moon)
                    if doublecam_do_star:
                        preview_star = cv2.subtract(img_star,img_bg_star)
                else:
                    if doublecam_do_moon:
                        preview_moon = img_moon
                    if doublecam_do_star:
                        preview_star = img_star
                    
                # Preview
                if doublecam_mode == 2:
                    sidebyside = np.hstack((preview_moon,preview_star))
                    cv2.imshow("Stream",sidebyside)
                else:
                    if doublecam_do_moon:
                        cv2.imshow("Stream",preview_moon)
                    elif doublecam_do_star:
                        cv2.imshow("Stream",preview_star)
                    else:
                        print("This should never have happened?!")

            # now, subtract bg anyway
            if doublecam_do_moon:
                img_moon = cv2.subtract(img_moon,img_bg_moon)
            if doublecam_do_star:
                img_star = cv2.subtract(img_star,img_bg_star)
            
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
                if doublecam_do_moon:
                    ret,img_bg_moon = stream_moon.read()
                if doublecam_do_star:
                    ret,img_bg_star = stream_star.read()
                
                if save_images:
                    print("SAVE IMAGES NOT IMPLEMENTED")
                    cv2.imwrite(img_name, img_bg)
                
            elif k == ord('p'):
                update_preview = not update_preview # toggle
                print("update_preview turned",update_preview)
            elif False: #k == ord('d'): # NO dont do this it causes seg faults and so on.
                doublecam_mode = (doublecam_mode + 1) % 3 # toggle
                
                doublecam_do_moon = doublecam_mode == 0 or doublecam_mode == 2
                doublecam_do_star = doublecam_mode == 1 or doublecam_mode == 2
                
                print("doublecam_mode turned",doublecam_mode,"(Moon:",doublecam_do_moon,"Star:",doublecam_do_star,")")
                
                if doublecam_do_moon:
                    if stream_moon.stopped:
                        stream_moon.start()
                else:
                    stream_moon.exit()
                if doublecam_do_star:
                    if stream_star.stopped:
                        stream_star.start()
                else:
                    stream_star.exit()
                
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
                print(' - Led',ind,"(ETA: %.1f min)"%( nleds*(time.time()-start)/(60*ind) if ind != 0 else 0. ) )
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
            
            elif started and t == 1 and ind % bg_autoupdate_every == 0:
                print("(auto) update background..")
                
                # BG img
                if doublecam_do_moon:
                    ret,img_bg_moon = stream_moon.read()
                if doublecam_do_star:
                    ret,img_bg_star = stream_star.read()
                
                
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
                    
                    xy_moon = [np.nan,np.nan]
                    if doublecam_do_moon:
                        thread_moon = threading.Thread(target=find_light_func,args=(xy_moon,img_moon,findlight_kwargs))
                        thread_moon.start()
                    xy_star = [np.nan,np.nan]
                    if doublecam_do_star:
                        thread_star = threading.Thread(target=find_light_func,args=(xy_star,img_star,findlight_kwargs))
                        thread_star.start()
                    
                    
                    if doublecam_do_moon:
                        thread_moon.join()
                    if doublecam_do_star:
                        thread_star.join()
                    
                    # xy_moon = find_light(img_moon,**findlight_kwargs)
                    if not np.isnan(xy_moon).any():
                        coords2d_moon[ind] = xy_moon
                    # xy_star = find_light(img_star,**findlight_kwargs)
                    if not np.isnan(xy_star).any():
                        coords2d_star[ind] = xy_star
                    
                    # if not np.isnan(xy_moon).any() and not np.isnan(xy_star).any():
                    if (doublecam_mode ==2 and not np.isnan(coords2d_moon[ind]).any() and not np.isnan(coords2d_star[ind]).any()) or (doublecam_do_moon and not np.isnan(coords2d_moon[ind]).any()) or (doublecam_do_star and not np.isnan(coords2d_star[ind]).any()):
                        t = delta_t-1
                        # print("   Done",t,coords2d[ind])
                else:
                    xy = None
                    
                
                print("   ",t,"%.02f"%(time.time()-start),xy_moon,xy_star)
                # time.sleep(1)
              
        # print()
        # print("Active threads", threading.activeCount())
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
