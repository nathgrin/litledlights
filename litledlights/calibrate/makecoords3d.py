import config


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from calibrate.findlights import find_light, reprocess
from calibrate.triangulate import combine_coords_2d_to_3d

import colors

import os

try:
    from utils import get_strip
except:
    print("import failed, setting config.dbg=True")
    config.dbg = True

from misc_func import npunit,rotationmtx,xyz_to_rthetaphi

    
def tst():
    """example from stackoverflow, in turn stolen from the "docs" """

    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "_tmp/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

def initiate_sequential_fotography(loc: str=None,skip_to_reprocess: bool=None):
    loc = config.sequentialfotography_loc if loc is None else loc
    findlight_threshold = config.findlight_threshold# if findlight_threshold is None else findlight_threshold
    
    ok = False
    do_reprocess = config.sequentialfotography_skiptoreprocess if skip_to_reprocess is None else skip_to_reprocess
    while not ok:
        if do_reprocess:
            coords2d = reprocess(loc=loc,findlight_threshold=findlight_threshold)
            do_reprocess = False
        else:
            coords2d = sequential_fotography(loc=loc,findlight_threshold=findlight_threshold)
        # print(coords2d)
        if coords2d is not None:
            print("NaN/tot: {}/{}".format(np.sum(np.isnan(coords2d)),len(coords2d)))
            
            # img_bg = cv2.imread(os.path.join(loc,"background.png"))
            # for i in range(len(coords2d)):
                # img_bg = cv2.putText(img_bg,str(i),coords2d[i],cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # cv2.imshow("Background with found lights",img_bg)
            
            print("You happy? Enter to accept, p to reprocess, anything else to redo")
            theinput = input("")
            # k = cv2.waitKey(0)
            
            ok = theinput == ""#k%256 == 10
            do_reprocess = theinput == "p"
            
        else:
            ok = False
        if do_reprocess:
            print("Lets reprocess..")
        elif not ok:
            print("Not happy, try again")
        else:
            print(" > Happy!")
    return coords2d
    
def sequential_fotography(strip=None,
                            color_off = (0,0,0),
                            color_on: tuple[int,int,int] = None,
                            
                            delta_t: int = None,# in arbitrary units
                            loc: str=None,
                            
                            findlight_threshold: int=None
                            
                            ) -> np.ndarray:
    """example from stackoverflow, in turn stolen from the "docs" 
    
    like matt parker does it. 
    Turn on each light in sequence and 
    
    
    
    """
    
    help_msg = "Press h for help,\n space to start or Pause,\n b for new background image,\n f to toggle background subtract of preview"
    
    # kwargs
    color_on = config.sequentialfotography_coloron if color_on is None else color_on
    delta_t = config.sequentialfotography_deltat if delta_t is None else delta_t
    loc = config.sequentialfotography_loc if loc is None else loc
    
    # Strip
    strip = get_strip() if strip is None else strip
    strip.fill( color_off )
    strip.show()
    
    # Findlight settings
    findlight_threshold = config.findlight_threshold if findlight_threshold is None else findlight_threshold
    
    # setup
    cam = cv2.VideoCapture(0)
    window_name = "Cam"
    cv2.namedWindow(window_name)
    
    # params
    nleds = len(strip)
    ind = 0
    t = -1
    
    started = False
    preview_subtract = True
    
    # BG img
    ret, img_bg = cam.read()
    img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    
    # Prep
    coords2d = [None for x in range(nleds)]
    
    start = time.time()
    
    print(" >",help_msg)
    
    try:
        while True:
            
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if preview_subtract:
                preview = cv2.subtract(frame,img_bg)
            else:
                preview = frame
            # frame = cv2.absdiff(frame,img_bg)
            cv2.imshow(window_name, preview)
            
            frame = cv2.subtract(frame,img_bg) # Even if preview doesnt subtract, THIS IS Subtracted anyway
    
            if started: t += 1
            
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
                ret, img_bg = cam.read()
                img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
                img_name = os.path.join(loc,"background.png")
                cv2.imwrite(img_name, img_bg)
                
            elif k%256 == 32:
                # SPACE pressed
                
                if started:
                    print(" > PAUSE at ind {0}".format(ind))
                    started = False
                else:
                    print(" > Vamonos")
                    t = 0 # reset t because we can
                    started = True
                
            elif started and (t+delta_t//2)%delta_t == 0: # Interlacing turning on/off lights and cam picture
                
                strip[ind-1] = color_off
                strip[ind]   = color_on
                strip.show()
                
            elif started and t%delta_t == 0: # Interlacing turning on/off lights and cam picture
                # print(t,started)
                
                
                img_name = os.path.join(loc,"led_{}.png".format(ind))
                cv2.imwrite(img_name, frame)
                # print("{} written!".format(img_name))
                
                xy = find_light(frame,threshold=findlight_threshold)
                
                coords2d[ind] = xy
                
                print("Frame",ind,"%.02f"%(time.time()-start),xy)
                
                ind += 1
                if ind == nleds:
                    print(" > We got em all")
                    break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        strip.fill( color_off )
        strip.show()
    if coords2d is None:
        return None
    return np.array(coords2d)


    

def coords2d_write(fname: str, coords2d: list[tuple[float,float]])->None:
    
    np.savetxt(fname,coords2d)

def coords2d_read(fname: str) -> list[tuple[float,float]]: 
    
    out = np.loadtxt(fname)
    return out
    
def coords3d_read(fname: str) -> list[tuple[float,float,float]]:
    return np.loadtxt(fname)

def get_coords2d_from_multiple_angles(n_viewpoints: int,loc: str="_tmp") -> list:
    
    
    coords2d_list = []
    
    for i in range(n_viewpoints):
        
        coords2d = initiate_sequential_fotography()
        
        coords2d_list.append(coords2d)
        
        fname = os.path.join(loc,"coords2d_{}.txt".format(i))
        coords2d_write(fname,coords2d)
    
    return coords2d_list

    
def combine_coords3d(coords3d_list: list):
    
    
    # find common non-nans
    any_isnan = np.sum(np.isnan(coords3d_list[0]),axis=1)
    for coords3d in coords3d_list:
        any_isnan = np.logical_or(any_isnan, np.sum(np.isnan(coords3d),axis=1) )
    
    
    ind_nonans = np.where(~any_isnan)
    print("no nans!:",ind_nonans)
    print("check above /\ for which are no nans")
    
    ind1,ind2,ind3 = config.combinecoords3d_referenceinds_default
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    
    if config.connect_ledlights:
        with get_strip() as strip:
            
            while True:
                
                print("Chosen inds:",ind1,ind2,ind3)
                print("xyz:",coords3d_list[0][ind1],coords3d_list[0][ind1],coords3d_list[0][ind1])
                strip.fill( (0,0,0) )
                strip[ind1] = colors.red
                strip[ind2] = colors.blue
                strip[ind3] = colors.white
                strip.show()
                
                print('Give 3 , separated indices, or "y" if ok.' )
                theinput = input("Origin (red), z-direction/unit length (blue), x-direction (white): ")
                if theinput == "y":
                    break
                elif theinput.count(",") == 2:
                    theinput = theinput.split(',')
                    ind1,ind2,ind3 = int(theinput[0]),int(theinput[1]),int(theinput[2])
            
            strip.fill( (0,0,0) )
            strip.show()
    else:
        print("Chosen inds:",ind1,ind2,ind3)
            
    out = []
    for ind,coords3d in enumerate(coords3d_list): # SKIPPING FIRST BY HAND BAD!
        
        norm = np.sqrt( np.sum( np.square(coords3d[ind1]-coords3d[ind2])) )
        
        coords3d = coords3d.transpose()
        
        
        for i in range(len(coords3d)): # Move to coordinate system
            coords3d[i] = (coords3d[i]-coords3d[i][ind1])/norm
        
        
        # rotate ind2 onto the x-axis
        phi = np.arctan2(coords3d[1][ind2], coords3d[0][ind2])
        # print("Angle",phi)
        coords3d = np.dot( rotationmtx(npunit(2),-phi) , coords3d )
        # rotate ind2 onto the z-axis
        phi = np.arctan2(coords3d[0][ind2],coords3d[2][ind2])
        # print("Angle",phi)
        coords3d = np.dot( rotationmtx(npunit(1),-phi) , coords3d )
        # rotate ind3 onto the x-axis
        phi = np.arctan2(coords3d[1][ind3], coords3d[0][ind3])
        # print("Angle",phi)
        coords3d = np.dot( rotationmtx(npunit(2),-phi) , coords3d )
        
        out.append(coords3d)
        # print("ind2",coords3d[0][ind2],coords3d[1][ind2],coords3d[2][ind2])
        
        coords3d_spherical = xyz_to_rthetaphi(coords3d)
        
        
        c2 = coords3d#rthetaphi_to_xyz(coords3d_spherical)
        
        # for i in range(len(c2[0])):
        # i = ind1
        # ax.scatter(c2[0][i],c2[1][i],c2[2][i],marker='o')
        # ax.scatter(c2[0][ind1],c2[1][ind1],c2[2][ind1],marker='o',c='k')
        # ax.scatter(c2[0][ind2],c2[1][ind2],c2[2][ind2],marker='o',c='r')
            
        
        
        print(ind,"ind1",c2[0][ind1], c2[1][ind1], c2[2][ind1],"ind2",c2[0][ind2], c2[1][ind2], c2[2][ind2],"ind3",c2[0][ind3], c2[1][ind3], c2[2][ind3])
        
        # ax.scatter(coords3d[0], coords3d[1], coords3d[2], marker='o',c='c')
        ind = coords3d_spherical[0] < 5
        # ax.scatter(c2[0][ind], c2[1][ind], c2[2][ind], marker='o',c='k')
        # ax.scatter(c2[0][ind1], c2[1][ind1], c2[2][ind1], marker='o',c='r')
        # ax.scatter(c2[0][ind2], c2[1][ind2], c2[2][ind2], marker='o',c='r')
        # ax.scatter(c2[0][ind3], c2[1][ind3], c2[2][ind3], marker='o',c='r')
        # ax.scatter(c2[0], c2[1], c2[2], marker='o')
        # print("OPKTA")
        
        # 2d
        # plt.plot(c2[0][ind2],c2[2][ind2],c='r',ls='',marker='o')
        # plt.plot(c2[0][ind3],c2[2][ind3],c='k',ls='',marker='o')
        # plt.plot(c2[0],c2[1],c='r',ls='',marker='o')
        # plt.plot(c2[2],c2[1],c='k',ls='',marker='o')
                # plt.plot(c2[0],c2[2],c='c',ls='',marker='o')
    # plt.show()
    
    
    # for i in range(len(out)):
    #     coords3d = out[i]
    #     print(i,"nan",np.sum(np.isnan(coords3d)))
    #     plt.plot(range(len(coords3d[0])),coords3d[0],marker='o',ls='-',label=str(i))
    # plt.legend()
    # plt.show()
    # for i in range(len(out)):
    #     coords3d = out[i]
    #     print(i,"nan",np.sum(np.isnan(coords3d)))
    #     plt.plot(range(len(coords3d[0])),coords3d[1],marker='o',ls='-',label=str(i))
    # plt.legend()
    # plt.show()
    # for i in range(len(out)):
    #     coords3d = out[i]
    #     print(i,"nan",np.sum(np.isnan(coords3d)))
    #     plt.plot(range(len(coords3d[0])),coords3d[2],marker='o',ls='-',label=str(i))
    # plt.legend()
    # plt.show()
    
    which = config.combinecoords3d_ind_coords3d
    return out[which].transpose()

def calibrate_updown(coords3d): # not used
    import keyboard
    white = (155,155,155)
    red = (155,0,0)
    off = (0,0,0)
    
    
    strip = get_strip()
    
    n_leds = strip.n
    
    # Setup
    window = cv2.namedWindow("UpDown")
    
    
    try:
    
        while True:
            ind_w = np.random.randint(n_leds)
            ind_r = np.random.randint(n_leds)
            
            strip.fill(off)
            strip[ind_w] = white
            strip[ind_r] = red
            
            strip.show()
            print("Which is up? (W)hite, (R)ed or (I) don't know")
            
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(coords3d.transpose()[0],coords3d.transpose()[1],coords3d.transpose()[2],marker='o',ls='')
            plt.show()
            
            k = cv2.waitKey(0)
                
            if k%256 == 27:
                # ESC pressed
                print("ESC: Quit")
                break
            elif k == ord('w'):
                print("White above")
            elif k == ord('r'):
                print("Red above")
            elif k == ord('i'):
                print("I don't know")
                
                
            time.sleep(0.1)
    finally:
        cv2.destroyAllWindows()
        strip.fill((0,0,0))
        strip.show()

def calc_neighbour_distances(coords3d):
    # Calc all distances forwardly
    thecopy = coords3d.copy()
    fwd = np.roll(thecopy,-1,axis=0)
    fwd[-1] = np.nan * fwd[-1]
    bwd = np.roll(thecopy,1,axis=0)
    bwd[0] = np.nan * bwd[0]
    # ind1,ind2 = 103,104
    # print(coords3d[ind1],coords3d[ind2])
    # print(thecopy[ind1],thecopy[ind2])
    
    dists_fwd = np.sqrt(np.sum(np.square(coords3d-fwd),axis=1))
    dists_bwd = np.sqrt(np.sum(np.square(coords3d-bwd),axis=1))
    
    return dists_fwd,dists_bwd


def OLD_coords3d_flag_bad_coords(coords3d:np.ndarray,
                             cutoff_in: float = config.coords3dflagbadcoords_cutoff,
                             ax_3d=None,ax_distperind=None,ax_distdistr=None):
    
    flags = np.zeros(len(coords3d))
    
    
    dists_fwd,dists_bwd = calc_neighbour_distances(coords3d)
    
    global cutoff
    cutoff = 4.*np.nanmean(dists_fwd)/3. # average r is 3/4 of radius
    print("Suggested cutoff {0}".format(cutoff))
    cutoff = cutoff_in if cutoff_in is not None else cutoff
    print("Mean Distance, Cutoff",np.nanmean(dists_fwd),cutoff)
    
    
    ## Select leds
    # Prep
    coords3d = coords3d.transpose()
    eps = np.nanmean(dists_fwd)*0.01 # allow a bit of slack
    ind_nandist = np.isnan(dists_fwd)
    
    # Flag nans
    flags[np.any(np.isnan(coords3d),axis=0)] = 1
    
    # Prep
    ind = np.logical_and( dists_fwd <= cutoff + eps , dists_bwd <= cutoff + eps )
    ind_toofar = np.logical_and(~ind,~ind_nandist)
    def set_inds_for_cutoff(cutoff):
        ind = np.logical_and( dists_fwd <= cutoff + eps , dists_bwd <= cutoff + eps )
        ind_toofar = np.logical_and(~ind,~ind_nandist)
        

        # Results
        print("> Cutoff: {0}".format(cutoff))
        print("  ind dist",np.sum(ind))
        print("  nan",np.sum(np.any(np.isnan(coords3d),axis=0)),"nandist",np.sum(ind_nandist))
        print("  too far",np.sum(ind_toofar))
        return ind,ind_toofar
        
    set_inds_for_cutoff(cutoff)
    
    
    # Show on leds
    if config.connect_ledlights:
        print("Show flags on leds. Red = nan, Blue = toofar")
        with get_strip() as strip:
            strip.fill( (0,0,0) )
            strip[flags == 1] = ( 115,0,0 )
            strip[flags == 2] = ( 0,0,115 )
            strip.show()
            input("Enter to continue")
    
    
    ### Initialize fig
    if ax_3d is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(bottom=0.25)
        ax_distperind = axs[0]
        ax_distdistr = axs[1]
        axs[2].remove()
        axs[2]=fig.add_subplot(1,3,3,projection='3d')
        ax_3d = axs[2]
    
    print("Plotting distances, Black: forward, Red: backward")
    # plt.figure()
    ax_distperind.plot(range(len(dists_fwd)),dists_fwd,c='k')
    ax_distperind.plot(range(len(dists_bwd)),dists_bwd,c='r')
    
    cutoff_hline = ax_distperind.axhline(cutoff,c='k')
    # plt.show()
    
    # print(dists)
    print("Plotting distance distribution, Black: forward, Red: backward")
    # plt.figure()
    ax_distdistr.plot(np.sort(dists_fwd),range(len(dists_fwd)),c='k')
    counts, bins = np.histogram(dists_fwd,range=(np.nanmin(dists_fwd),np.nanmax(dists_fwd)),bins='auto',density=True)
    ax_distdistr.stairs(len(dists_fwd)*counts/np.max(counts), bins,ec='k') # Density times lens to share same y-range
    ax_distdistr.plot(np.sort(dists_bwd),range(len(dists_bwd)),c='r')
    counts, bins = np.histogram(dists_bwd,range=(np.nanmin(dists_bwd),np.nanmax(dists_bwd)),bins='auto',density=True)
    ax_distdistr.stairs(len(dists_bwd)*counts/np.max(counts), bins,ec='r') # Density times lens to share same y-range
    
    cutoff_vline = ax_distdistr.axvline(cutoff,c='k')
    # plt.show()
    

    # Show in fig
    print("Plot flags, Red: No dist, Cyan: too far")
    ax_3d.set_xlabel('$x$')
    ax_3d.set_ylabel('$y$')
    ax_3d.set_zlabel('$z$')
    
    ax_3d.plot(coords3d[0],coords3d[1],coords3d[2],marker='',ls='-',c='k')
    ax_3d.plot(coords3d[0][ind],coords3d[1][ind],coords3d[2][ind],marker='o',ls='',c='k')
    ax_3d.plot(coords3d[0][ind_nandist],coords3d[1][ind_nandist],coords3d[2][ind_nandist],marker='o',ls='',c='r') # ...
    
    ax_3d.plot(coords3d[0][ind_toofar],coords3d[1][ind_toofar],coords3d[2][ind_toofar],marker='o',ls='',c='c') # ...
    
    # ax.set_xlim(xmax=4)
    # ax.set_ylim(ymin=-2,ymax=2)
    # ax.invert_yaxis()
    # slider
    
    if ax_3d is None:
        from matplotlib.widgets import RangeSlider,Slider,Button
        # Create the Slider
        slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
        slider_cutoff = Slider(slider_ax, "cutoff", 0, 5,valinit=cutoff,valstep=0.01)
        
        def update_cutoff(val):
            
            global cutoff 
            cutoff = slider_cutoff.val
            
            cutoff_hline.set_ydata([cutoff,cutoff])
            cutoff_vline.set_xdata([cutoff,cutoff])
            
            set_inds_for_cutoff(cutoff)
            
            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()
            
        slider_cutoff.on_changed(update_cutoff)
        # cutoff = slider_cutoff.val
        
        # SHOW
        plt.show()
    
    print("Using cutoff: {0}".format(cutoff))
    # Apply flags
    ind,ind_toofar = set_inds_for_cutoff(cutoff)
    flags[ind_toofar] = 2
    
    return flags


def show_coords_onlights(coords3d):
    print("Showing coords on lights:")
    
    coords3d = coords3d.transpose()
    xmax = np.nanmax(np.abs(coords3d[0]))
    ymax = np.nanmax(np.abs(coords3d[1]))
    zmax = np.nanmax(np.abs(coords3d[2]))
    
    x,y,z = coords3d[0],coords3d[1],coords3d[2]
    # print("max",xmax,ymax,zmax)
    
    with get_strip() as strip:
        # for i in range(len(x)):
            
            # if not np.any(np.isnan([x[i],y[i],z[i]])):
                # strip[i] = ( int(255*(xmax-abs(x[i]))/xmax),int(255*(ymax-abs(y[i]))/ymax),int(255*(zmax-abs(z[i]))/zmax) )
            # else:
                # strip[i] = (0,0,0)
        # strip.show()
        
        # input("Showing coords.. Enter to continue")
        
        strip.fill( (0,0,0) )
        
        #  +++
        ind = np.logical_and( np.logical_and( x >= 0,y >= 0 ) , z >= 0 )
        strip[ind] = colors.red
        #  ---
        ind = np.logical_and( np.logical_and( x < 0,y < 0 ) , z < 0 )
        strip[ind] = colors.red
        #  -++
        ind = np.logical_and( np.logical_and( x < 0,y >= 0 ) , z >= 0 )
        strip[ind] = colors.blue
        #  +--
        ind = np.logical_and( np.logical_and( x > 0,y < 0 ) , z < 0 )
        strip[ind] = colors.blue
        #  +-+
        ind = np.logical_and( np.logical_and( x >= 0,y <= 0 ) , z >= 0 )
        strip[ind] = colors.green
        #  -+-
        ind = np.logical_and( np.logical_and( x < 0,y > 0 ) , z < 0 )
        strip[ind] = colors.green
        #  ++-
        ind = np.logical_and( np.logical_and( x >= 0,y >= 0 ) , z <= 0 )
        strip[ind] = colors.pink
        #  --+
        ind = np.logical_and( np.logical_and( x < 0,y < 0 ) , z > 0 )
        strip[ind] = colors.pink
        
        strip.show()
        
        print("red: +++, blue: -++, green: +-+, pink: ++-, and inverses")
        input("Showing coords.. Enter to continue")


def coords3d_fix_flagged_coords(coords3d: np.ndarray,flags: np.ndarray) -> np.ndarray:
    
    coords3d = coords3d.transpose()
    
    good = flags == 0
    print("Number not flagged:",np.sum(good))
    import scipy.interpolate as inter
    ind = np.arange(len(flags))
    for i,coord in enumerate(coords3d): # loop over xyz
        # print(coord)
        spline = inter.interp1d(ind[good],coord[good],kind=config.coords3d_fixbad_splinekind,bounds_error=False,fill_value=np.nan)
        
        coords3d[i][~good] = spline(ind[~good])
        
        
        
        # xarr = np.linspace(0,len(flags),30*len(flags))
        # plt.plot(xarr,spline(xarr),c='r')
        
        # plt.plot(ind[good],coord[good],marker='o',c='k',ls='')
        # plt.plot(ind[~good],coord[~good],marker='o',c='c',ls='')
        # plt.show()
        
    
    print("Plot fixed coords")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.plot(coords3d[0][good],coords3d[1][good],coords3d[2][good],marker='o',ls='',c='k')
    ax.plot(coords3d[0][~good],coords3d[1][~good],coords3d[2][~good],marker='o',ls='',c='c')
    plt.show()
    
    
    return coords3d.transpose()

def OLD_linear_flow(coords2d_list,n_viewpoints:int,
                    camera_matrix: np.ndarray=None,distortions: np.ndarray=None, new_camera_matrix: np.ndarray=None):
    coords3d_list = None
    print("> Combine coords2d to 3d (triangulate)")
    if config.do_2d_to_3d:
        coords3d_list = combine_coords_2d_to_3d(coords2d_list,n_viewpoints=n_viewpoints,camera_matrix=camera_matrix,distortions=distortions,new_camera_matrix=new_camera_matrix)
    else:
        print("  Load coords3d files")
        coords3d_list = []
        for i in range(n_viewpoints*(n_viewpoints-1)//2):
            fname = os.path.join("_tmp","coords3d_{}.txt".format(i))
            coords3d_list.append( coords3d_read(fname) )
    
    # for i,coords3d in enumerate(coords3d_list):
        # Swap x and y for physics convention for xyz
        # Mirror in y (artifact of CV y-axis convention)
        # well.. this only matters if rotation calibaration doesnt work!
        # coords3d = coords3d.transpose()
        # tmp1 = coords3d[1].copy()
        # tmp2 = coords3d[2].copy()
        # coords3d[1],coords3d[2] = tmp2,-tmp1 
        # coords3d_list[i] = coords3d.transpose()
    
    
    
    # Combine
    print("> Combine coords3d")
    coords3d = None
    # this doesnt combine, just picks one and rotates it around
    coords3d = combine_coords3d(coords3d_list) 
    
    
    # for now, just pick one of them
    if coords3d is None:
        coords3d_ind = config.ind_coords3d
        coords3d = coords3d_list[ coords3d_ind ]
    
    if config.connect_ledlights:
        show_coords_onlights(coords3d)
    
    # Find bad
    print("> Find bad")
    flags = coords3d_flag_bad_coords(coords3d)
        
    
    # Fix missing
    coords3d = coords3d_fix_flagged_coords(coords3d,flags)
    
    # Calibrate direction of axes
    # calibrate_updown(coords3d)
    
    # Done, save
    print("Saving=",config.save_coords3d)
    if config.save_coords3d:
        np.savetxt(config.savecoords3d_fname,coords3d,header="x\ty\tz")
    
    
    print("> Plotting")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.plot(coords3d.transpose()[0],coords3d.transpose()[1],coords3d.transpose()[2],marker='o',ls='',c='k')
    plt.show()
    
    
def coords3d_flag_bad_coords(coords3d,cutoff,dists_fwd=None,dists_bwd=None):
    if dists_fwd is None:
        dists_fwd,dists_bwd = calc_neighbour_distances(coords3d)
    if cutoff is None:
        cutoff = 4.*np.nanmean(dists_fwd)/3. # average r is 3/4 of radius
        print("Suggested cutoff {0}".format(cutoff))
        cutoff = config.coords3dflagbadcoords_cutoff if config.coords3dflagbadcoords_cutoff is not None else cutoff
        print("Mean Distance, Cutoff",np.nanmean(dists_fwd),cutoff)
    
    # ini
    flags = np.zeros(len(coords3d))
    
    # flag nans
    flags[np.any(np.isnan(coords3d),axis=1)] = 1
    
    # flag too far
    ind_toofar = np.logical_and(dists_fwd > cutoff, dists_bwd > cutoff)
    flags[ind_toofar] = 2
    
    return flags
    
class coords2dto3dObject(object):
    
    def __init__(self,*args,**kwargs):
        
        pass
    
    def initialize_fig(self,*args,**kwargs):
        
        from matplotlib.widgets import Slider,Button
        
        distcutoff = 1.#kwargs.get('distcutoff',1.)
        
        ##### Prepare figure
        self.fig = plt.figure(figsize=(10,10))
        self.gs = self.fig.add_gridspec(4,5)
        # row,column (y,x in a sway)
        self.ax_2d = self.fig.add_subplot(self.gs[0, 0])
        self.ax_2d_lines = [0,1]# Lines are overwritten in SET! # because set_data only works for same length arr
        
        self.ax_2d.invert_yaxis()
        
        self.ax_distperind = self.fig.add_subplot(self.gs[1, 0])
        self.ax_distperind_lines = [self.ax_distperind.axvline(distcutoff,c='k')]
        self.ax_distdistr  = self.fig.add_subplot(self.gs[2, 0])
        self.ax_distdistr_lines = [self.ax_distdistr.axvline(distcutoff,c='k')]
        
        # coords3d
        self.ax_3d = self.fig.add_subplot(self.gs[0:2, 1:3],projection='3d')
        
        self.ax_3d.set_xlabel('$x$')
        self.ax_3d.set_ylabel('$y$')
        self.ax_3d.set_zlabel('$z$')
        
        self.ax_3d2 = self.fig.add_subplot(self.gs[0:2, 3:5],projection='3d')
        
        
        
        ### Sliders
        self.gs_sliders = self.gs[3,0].subgridspec(5, 1)
        
        self.ax_slider_distcutoff = self.fig.add_subplot(self.gs_sliders[0])
        self.slider_distcutoff = Slider(self.ax_slider_distcutoff, "Dist cutoff", 0., 3., valinit=distcutoff, valstep=0.01)
        self.slider_distcutoff.on_changed(self.slider_distcutoff_update)
        
        self.ax_button_update = self.fig.add_subplot(self.gs_sliders[-1])
        self.button_update = Button(self.ax_button_update, "Update")
        self.button_update.on_clicked(self.button_update_update)
        
    def slider_distcutoff_update(self,val):
    
        self.distcutoff = val
        distcutoff = val
        self.ax_distperind_lines[0].set_xdata([distcutoff,distcutoff])
        self.ax_distdistr_lines[0].set_xdata([distcutoff,distcutoff])
        
        
        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()
    
        
        # self.update()
    
    def button_update_update(self,event):
        
        self.update()
    
    def update(self):
        print("Update")
        distcutoff = self.distcutoff
        ## Calculation nation
        from calibrate.triangulate import combine_pair_coords2d
        coords3d = combine_pair_coords2d(self.coords2d1,self.coords2d2,self.camera_matrix)
        coords3d =  h cd x c                                              xc coords3d
        self.coords3d = coords3d
        
        dists_fwd,dists_bwd = calc_neighbour_distances(self.coords3d)
        
        
    
        suggested_distcutoff = 4.*np.nanmean(dists_fwd)/3. # average r is 3/4 of radius
        print("Suggested distcutoff {0}".format(suggested_distcutoff))
        print("Mean Distance, Cutoff",np.nanmean(dists_fwd),distcutoff)
        
        
    
        # print(self.coords3d)
        flags = coords3d_flag_bad_coords(coords3d,distcutoff)
        
        ind_toofar = flags == 2
        ind_nan = flags == 1
        
        ## Plots
        # Clear
        self.ax_distperind.clear()
        self.ax_distdistr.clear()
        self.ax_3d.clear()
        
        # Distance plots
        self.ax_distperind_lines[0] = self.ax_distperind.axvline(distcutoff,c='k')
        self.ax_distdistr_lines[0] = self.ax_distdistr.axvline(distcutoff,c='k')
        
        print("Plotting distances, Black: forward, Red: backward")
        # plt.figure()
        self.ax_distperind.plot(dists_fwd,range(len(dists_fwd)),c='k')
        self.ax_distperind.plot(dists_bwd,range(len(dists_bwd)),c='r')
        
        # plt.show()
        
        # print(dists)
        print("Plotting distance distribution, Black: forward, Red: backward")
        # plt.figure()
        self.ax_distdistr.plot(np.sort(dists_fwd),range(len(dists_fwd)),c='k')
        counts, bins = np.histogram(dists_fwd,range=(np.nanmin(dists_fwd),np.nanmax(dists_fwd)),bins='auto',density=True)
        self.ax_distdistr.stairs(len(dists_fwd)*counts/np.max(counts), bins,ec='k') # Density times lens to share same y-range
        self.ax_distdistr.plot(np.sort(dists_bwd),range(len(dists_bwd)),c='r')
        counts, bins = np.histogram(dists_bwd,range=(np.nanmin(dists_bwd),np.nanmax(dists_bwd)),bins='auto',density=True)
        self.ax_distdistr.stairs(len(dists_bwd)*counts/np.max(counts), bins,ec='r') # Density times lens to share same y-range
        
        # 3d
        self.ax_3d.plot(*self.coords3d.transpose(),c='k',marker='o',ls='')
        
        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()
    
    def initialize_data(self,coords2d1,coords2d2,
            distcutoff: float = None,
            camera_matrix: np.ndarray=None):
        
        self.coords2d1 = np.ma.asarray(coords2d1)
        self.coords2d2 = np.ma.asarray(coords2d2)
        
        self.ax_2d_lines[0] = self.ax_2d.plot(self.coords2d1[:,0],self.coords2d1[:,1],c='k',marker='o',ls='')
        self.ax_2d_lines[1] = self.ax_2d.plot(self.coords2d2[:,0],self.coords2d2[:,1],c='r',marker='o',ls='')
        
        self.distcutoff = distcutoff if distcutoff is not None else config.coords3dflagbadcoords_cutoff
        self.slider_distcutoff.set_val(self.distcutoff)
        
        self.camera_matrix = camera_matrix
        
        
        self.ind = np.ones(coords2d1.shape[0],dtype=bool)
        self.coords3d = np.ma.asarray( np.nan*np.zeros((self.coords2d1.shape[0],3)) )
        print(self.coords3d)
        
        
        self.update()
    
def iterative_pair_coords2d_to_coords3d(coords2d1,coords2d2,
                                   camera_matrix: np.ndarray=None,distortions: np.ndarray=None, new_camera_matrix: np.ndarray=None):
    
    if distortions is not None:# and new_camera_matrix is not None: # This doesnt work!
        print("Undistorting!")
        if new_camera_matrix is None:
            new_camera_matrix = camera_matrix
            
        undistorted = cv2.undistortPoints(coords2d1, camera_matrix, distortions, P=new_camera_matrix) 
        undistorted = np.squeeze(undistorted)
        coords2d1 = undistorted
        undistorted = cv2.undistortPoints(coords2d2, camera_matrix, distortions, P=new_camera_matrix) 
        undistorted = np.squeeze(undistorted)
        coords2d2 = undistorted
    
    
    
    
    
    coordObject = coords2dto3dObject()
    coordObject.initialize_fig()
    coordObject.initialize_data(coords2d1,coords2d2,
                    distcutoff=None,# defaults to config
                    camera_matrix=camera_matrix)
    
    # ax_2d = coordObject.ax_2d
    ax_distperind = coordObject.ax_distperind
    ax_distdistr = coordObject.ax_distdistr
    ax_3d = coordObject.ax_3d
    
    ###
    # coords2d plot
    # ax_2d.plot(*coords2d1.transpose(),c='k',marker='o',ls='')
    # ax_2d.plot(*coords2d2.transpose(),c='r',marker='o',ls='')
    # ax_2d.invert_yaxis()
    
    
    
    # plt.show()
    
    
    
    
    
    
    ### Show and save
    plt.savefig(os.path.join("_tmp","coords2d_to_3d"+".png"),bbox_inches='tight')
    plt.show()
    
    return coords3d

def main():
    
    # From calibatrion
    distortions = config.distortions
    camera_matrix = config.camera_matrix
    new_camera_matrix = config.new_camera_matrix

    n_viewpoints = config.getcoords2d_nviewpoints # how many images do we use
    
    coords2d_list = None
    if config.getcoords2d_fromangles and not config.dbg:
        coords2d_list = get_coords2d_from_multiple_angles(n_viewpoints)
        coords2d_list = [c2d.transpose() for c2d in coords2d_list]
    
    
    if coords2d_list is None:
        coords2d_list = []
        for i in range(n_viewpoints):
            fname = "_tmp/"+"coords2d_{}.txt".format(i)
            coords2d_list.append( coords2d_read(fname) )
            
    
    coords2d1,coords2d2 = coords2d_list[0],coords2d_list[1]
    
    iterative_pair_coords2d_to_coords3d(coords2d1,coords2d2,
                                   camera_matrix=camera_matrix,distortions=distortions, new_camera_matrix=new_camera_matrix)
    
    
    
    
if __name__ == "__main__":
    main()
