
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import color

import os

try:
    from utils import get_strip
except:
    print("import failed")


    
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

def initiate_sequential_fotography():
    ok = False
    while not ok:
        coords2d = sequential_fotography()
        # print(coords2d)
        if coords2d is not None:
            print("NaN/tot: {}/{}".format(np.sum(np.isnan(coords2d)),len(coords2d)))
            
            # img_bg = cv2.imread(os.path.join(loc,"background.png"))
            # for i in range(len(coords2d)):
                # img_bg = cv2.putText(img_bg,str(i),coords2d[i],cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # cv2.imshow("Background with found lights",img_bg)
            
            print("You happy? Enter to accept, anything else to redo")
            theinput = input("")
            # k = cv2.waitKey(0)
            
            ok = theinput == ""#k%256 == 10
            
        else:
            ok = False
        if not ok:
            print("Not happy, try again")
        else:
            print(" > Happy!")
    return coords2d
    
def sequential_fotography(strip=None,
                            color_off = (0,0,0),
                            color_on = (255,255,255),
                            
                            delta_t = 10,# in arbitrary units
                            loc = "_tmp"
                            ):
    """example from stackoverflow, in turn stolen from the "docs" 
    
    like matt parker does it. 
    Turn on each light in sequence and 
    
    
    
    """
    
    help_msg = "Press h for help,\n space to start,\n b for new background image,\n f to toggle background subtract of preview"
    
    # Strip
    strip = get_strip() if strip is None else strip
    strip.fill( color_off )
    strip.show()
    
    
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
            
            frame = cv2.subtract(frame,img_bg)
    
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

                print(" > Vamonos")
                t = 0 # reset t because we can
                started = True
                
            elif started and (t+delta_t//2)%delta_t == 0:
                
                strip[ind-1] = color_off
                strip[ind]   = color_on
                strip.show()
                
            elif started and t%delta_t == 0:
                # print(t,started)
                
                
                img_name = os.path.join(loc,"led_{}.png".format(ind))
                cv2.imwrite(img_name, frame)
                # print("{} written!".format(img_name))
                
                xy = find_light(frame)
                
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
    
    return coords2d

def find_light(img,
               blur_radius: int=9, # has to be odd!
               )->tuple[float,float]:
    """
    follow matt, simply get brightest pixels
    """
    
    
    blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
    
    ind = blur > 80#100
    y,x = np.mean(np.where(ind),axis=1)
    
    
    # ind = img > 180
    
    # x,y = np.mean(np.where(ind), axis=1)
        
    
    # print(x,y)
    
    return (x,y)
    

def coords2d_write(fname: str, coords2d: list[tuple[float,float]])->None:
    import json
    
    
    with open(fname,'w') as thefile:
        thefile.write(json.dumps(coords2d))

def coords2d_read(fname: str) -> list[tuple[float,float]]: 
    import json
    with open(fname,'r') as thefile:
        out = json.loads(thefile.readline())
    return np.array(out)
    
def coords3d_read(fname: str) -> list[tuple[float,float,float]]:
    return np.loadtxt(fname)

def get_coords2d_from_multiple_angles(n_images: int,loc: str="_tmp") -> list:
    
    
    coords2d_list = []
    
    for i in range(n_images):
        
        coords2d = initiate_sequential_fotography()
        
        coords2d_list.append(coords2d)
        
        fname = os.path.join(loc,"coords2d_{}.txt".format(i))
        coords2d_write(fname,coords2d)
    
    return coords2d_list


def combine_coords_2d_to_3d(coords2d_list: list[list[tuple[float,float]]],n_images: int=None,camera_matrix=None,distortions:np.ndarray=None,new_camera_matrix:np.ndarray=None) -> list[tuple[float,float,float]]:
    
    
    # print(coords2d)
    
    # Undistort
    
    if distortions is not None:# and new_camera_matrix is not None: # This doesnt work!
        print("Undistorting!")
        for i in range(len( coords2d_list )):
            # print(distortions)
            # undistorted = cv2.undistortPoints(coords2d_list[i], camera_matrix, distortions,None,new_camera_matrix)
            if new_camera_matrix is None:
                new_camera_matrix = camera_matrix
            undistorted = cv2.undistortPoints(coords2d_list[i], camera_matrix, distortions, P=new_camera_matrix) 
            undistorted = np.squeeze(undistorted)
            # print(undistorted)
            coords2d_list[i] = undistorted
    
    import itertools
    
    coords3d_list = []
    cnt = 0
    
    # For each pair of coord lists (image)
    # for coords2d1,coords2d2 in itertools.combinations(coords2d_list,2):
    for ind1, ind2 in itertools.combinations(range(len(coords2d_list)),2):
        print("cnt",cnt,"ind1 ind2",ind1,ind2)
        coords2d1,coords2d2 = coords2d_list[ind1],coords2d_list[ind2]
        
        
        # plt.plot(coords2d1[0],coords2d1[1],marker='o',ls='')
        plt.plot(coords2d1.transpose()[0],coords2d1.transpose()[1],marker='o',ls='',c='k')
        plt.plot(coords2d2.transpose()[0],coords2d2.transpose()[1],marker='o',ls='',c='r')
        plt.gca().invert_yaxis()
        plt.show()
        
        coords3d = triangulate( coords2d1,coords2d2 ,camera_matrix=camera_matrix)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(coords3d.transpose()[0],coords3d.transpose()[1],coords3d.transpose()[2],marker='o',ls='',c='k')
        
        plt.show()
        
        # print(coords3d)
        coords3d_list.append(coords3d)
        
        np.savetxt( "_tmp/"+"coords3d_{}.txt".format(cnt) ,coords3d)
        
        cnt += 1
        # coords3d = coords3d.transpose()
    
    return coords3d_list
    
    
    
def filter_nans(pts1,pts2):
    # Filter nans
    thenans = []
    for i in range(len(pts1)):
        if np.any(np.isnan(pts1[i])):
            thenans.append(i)
    for i in range(len(pts2)):
        if np.any(np.isnan(pts2[i])):
            thenans.append(i)
    
    # convert to int?
    pts1 = [ pts1[i] for i in range(len(pts1)) if i not in thenans ]
    pts2 = [ pts2[i] for i in range(len(pts2)) if i not in thenans ]
    # pts1 = [ np.int32(pts1[i]) for i in range(len(pts1)) if i not in thenans ]
    # pts2 = [ np.int32(pts2[i]) for i in range(len(pts2)) if i not in thenans ]
    pts1,pts2 = np.array(pts1),np.array(pts2)
    # no more nans
    # print(pts2)
    return pts1,pts2
    
def opencvexample(pts1,pts2):
    
    pts1,pts2 = filter_nans(pts1,pts2)
    import cv2 as cv
    
    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)
    
    
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    
    print(F,mask)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
def triangulate(pts1,pts2, camera_matrix=None):
    """
    from https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units
    """
    pts1,pts2 = np.array(pts1),np.array(pts2)
    # print(pts1,pts2)
    # ind = np.logical_or( np.isnan(pts1) , np.isnan(pts2) )
    # print(ind)
    # pts1,pts2 = pts1[ind],pts2[ind]
    
    if camera_matrix is None:
        cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
    else:
        cameraMatrix = camera_matrix
    F,m1 = cv2.findFundamentalMat(pts1, pts2) # apparently not necessary

    # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations: 
    E,m2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
    # Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results. 
    # K_l = cameraMatrix
    # K_r = cameraMatrix
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts1, pts2, cameraMatrix,distanceThresh=0.5)
    # retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts_l_norm, pts_r_norm, cameraMatrix,distanceThresh=0.5)

    # given R,t you can  explicitly find 3d locations using projection 
    M_r = np.concatenate((R,t),axis=1)
    M_l = np.concatenate((np.eye(3,3),np.zeros((3,1))),axis=1)
    proj_r = np.dot(cameraMatrix,M_r)
    proj_l = np.dot(cameraMatrix,M_l)
    points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    points_3d = points_4d[:3, :].T
    return points_3d
    
def triangulate_full(pts1,pts2, camera_matrix=None):
    """
    from https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units
    """
    pts1,pts2 = np.array(pts1),np.array(pts2)
    # print(pts1,pts2)
    # ind = np.logical_or( np.isnan(pts1) , np.isnan(pts2) )
    # print(ind)
    # pts1,pts2 = pts1[ind],pts2[ind]
    
    if camera_matrix is None:
        cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
    else:
        cameraMatrix = camera_matrix
    F,m1 = cv2.findFundamentalMat(pts1, pts2) # apparently not necessary

    # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations: 
    E,m2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
    Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results. 
    K_l = cameraMatrix
    K_r = cameraMatrix
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts1, pts2, cameraMatrix,distanceThresh=0.5)
    # retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts_l_norm, pts_r_norm, cameraMatrix,distanceThresh=0.5)

    # given R,t you can  explicitly find 3d locations using projection 
    M_r = np.concatenate((R,t),axis=1)
    M_l = np.concatenate((np.eye(3,3),np.zeros((3,1))),axis=1)
    proj_r = np.dot(cameraMatrix,M_r)
    proj_l = np.dot(cameraMatrix,M_l)
    points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    points_3d = points_4d[:3, :].T
    return points_3d
  
def xyz_to_rthetaphi(xyz): # https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    
    xy = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    phi = np.arctan2(xyz[1], xyz[0])
    return np.array([r,theta,phi])
    
def rthetaphi_to_xyz(rthetaphi):
    x = rthetaphi[0]*np.sin(rthetaphi[1])*np.cos(rthetaphi[2])
    y = rthetaphi[0]*np.sin(rthetaphi[1])*np.sin(rthetaphi[2])
    z = rthetaphi[0]*np.cos(rthetaphi[1])
    return np.array([x,y,z])


def rotationmtx(axis:np.ndarray,angle:float) -> np.array:
    #https://en.wikipedia.org/wiki/Rotation_matrix
    axis = axis/np.linalg.norm(axis)
    x,y,z = axis[0],axis[1],axis[2]
    costheta = np.cos(angle)
    sintheta = np.sin(angle)
    onemincos = 1.-costheta
    R = np.array( [
        [ costheta + x*x*onemincos , x*y*onemincos-z*sintheta , x*z*onemincos+y*sintheta ],
        [ y*x*onemincos + z*sintheta , costheta+y*y*onemincos , y*z*onemincos - x*sintheta ],
        [ z*x*onemincos - y*sintheta , z*y*onemincos+x*sintheta , costheta+z*z*onemincos ]
    ] )
    return R

def npunit(index:int,size=3):
    arr = np.zeros(size)
    arr[index] = 1.
    return arr

def combine_coords3d(coords3d_list: list):
    
    
    # find common non-nans
    any_isnan = np.sum(np.isnan(coords3d_list[0]),axis=1)
    for coords3d in coords3d_list:
        any_isnan = np.logical_or(any_isnan, np.sum(np.isnan(coords3d),axis=1) )
    
    ind_nonans = np.where(~any_isnan)
    
    # ind1,ind2 = ind_nonans[0][0],ind_nonans[0][1]
    ind1,ind2,ind3 = 150,142,185
    print("no nans!:",ind_nonans)
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    
    
    with get_strip() as strip:
        
        while True:
            
            print("Chosen inds:",ind1,ind2,ind3)
            strip.fill( (0,0,0) )
            strip[ind1] = color.red
            strip[ind2] = color.blue
            strip[ind3] = color.white
            strip.show()
            
            print('Give 3 , separated indices, or "y" if ok.' )
            theinput = input("Origin (red), z-point/unit lengthp (blue), x-point (red): ")
            if theinput == "y":
                break
            elif theinput.count(",") == 2:
                theinput = theinput.split(',')
                ind1,ind2,ind3 = int(theinput[0]),int(theinput[1]),int(theinput[2])
        
        strip.fill( (0,0,0) )
        strip.show()
        
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
    
    which = 1
    return out[which].transpose()

def calibrate_updown(coords3d):
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

def get_coords(fname:str="coords.txt"):
    return np.loadtxt(fname)


def coords3d_flag_bad_coords(coords3d:np.ndarray):
    
    
    # Calc all distances forwardly
    thecopy = coords3d.copy()
    thecopy = np.roll(thecopy,-1,axis=0)
    thecopy[-1] = np.nan * thecopy[-1]
    # ind1,ind2 = 103,104
    # print(coords3d[ind1],coords3d[ind2])
    # print(thecopy[ind1],thecopy[ind2])
    
    dists = np.sum(np.square(coords3d-thecopy),axis=1) # SQUARED distances
    
    cutoff = 5.*np.nanmean(dists)/3. # average r squared is 3/5 of radius
    print("Cutoff",cutoff)
    
    
    
    # print(dists)
    # plt.plot(np.sort(dists),range(len(dists)))
    # plt.show()
    
    coords3d = coords3d.transpose()
    eps = np.nanmean(dists)*0.01 # allow a bit of slack
    ind = dists <= cutoff + eps
    ind_nan = np.isnan(dists)
    
    print("ind dist",np.sum(ind))
    print("nan",np.sum(np.isnan(coords3d)),"nandist",np.sum(ind_nan))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    
    ax.plot(coords3d[0],coords3d[1],coords3d[2],marker='',ls='-',c='k')
    ax.plot(coords3d[0][ind],coords3d[1][ind],coords3d[2][ind],marker='o',ls='',c='k')
    ax.plot(coords3d[0][ind_nan],coords3d[1][ind_nan],coords3d[2][ind_nan],marker='o',ls='',c='r') # ...
    
    ind_other = np.logical_and(~ind,~ind_nan)
    ax.plot(coords3d[0][ind_other],coords3d[1][ind_other],coords3d[2][ind_other],marker='o',ls='',c='c') # ...
    
    # ax.invert_yaxis()
    plt.show()

def firstcalibration():

    
    # From calibatrion
    camera_matrix = np.array([ [1.06996313e+03,0.00000000e+00,3.15927309e+02],
                               [0.00000000e+00,7.98554626e+02,2.30317334e+02],
                               [0.00000000e+00,0.00000000e+00,1.00000000e+00]]) # "new camera mtx"
    new_camera_matrix = None
    # camera_matrix = None#np.array([ [5.532296390058696716e+02,0.000000000000000000e+00,3.092664256633688638e+02],
                            #    [0.000000000000000000e+00,5.004930172490122686e+02,2.534310435736532838e+02],
                            #    [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00] ]) # first camera mtx
    distortions = None#np.array([ 2.03990656e-01,-4.10106338e+01,3.88358091e-02,5.41687259e-02,3.86933501e+02 ]) # this didnt work, i suspect the calibration is imperfect


def show_coords(coords3d):
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
        strip[ind] = color.red
        #  ---
        ind = np.logical_and( np.logical_and( x < 0,y < 0 ) , z < 0 )
        strip[ind] = color.red
        #  -++
        ind = np.logical_and( np.logical_and( x <= 0,y >= 0 ) , z >= 0 )
        strip[ind] = color.blue
        #  +--
        ind = np.logical_and( np.logical_and( x > 0,y < 0 ) , z < 0 )
        strip[ind] = color.blue
        #  +-+
        ind = np.logical_and( np.logical_and( x >= 0,y <= 0 ) , z >= 0 )
        strip[ind] = color.green
        #  -+-
        ind = np.logical_and( np.logical_and( x < 0,y > 0 ) , z < 0 )
        strip[ind] = color.green
        #  ++-
        ind = np.logical_and( np.logical_and( x >= 0,y >= 0 ) , z <= 0 )
        strip[ind] = color.pink
        #  --+
        ind = np.logical_and( np.logical_and( x < 0,y < 0 ) , z > 0 )
        strip[ind] = color.pink
        
        strip.show()
        
        print("red: +++, blue: -++, green: +-+, pink: ++-, and inverses")
        input("Showing coords.. Enter to continue")

def main():
    
    # From calibatrion
    # Somehow things work better when new_camera_mtx=new_camera_mtx = new_camera_mtx
    # camera_matrix = np.array( [[794.0779295,0.,334.41476339],
    #                            [  0.,790.21709526,248.42875997],
    #                            [  0.,0.,1.        ]] )  # first camera mtx
    distortions = np.array( [[ 2.05088975e-01 ,-1.09124274e+00 , 4.90025360e-04 , 1.83144614e-02   ,    2.58532256e+00]] )
    camera_matrix = np.array( [[802.89550781,0,340.40239924],
                               [  0,793.20324707 ,247.94272481],
                               [  0,0,1.        ]]) # newcameramtx
    new_camera_matrix = None

    n_images = 4 # how many images do we use
    
    coords2d_list = None
    # coords2d_list = get_coords2d_from_multiple_angles(n_images)
    
    
    if coords2d_list is None:
        coords2d_list = []
        for i in range(n_images):
            fname = "_tmp/"+"coords2d_{}.txt".format(i)
            coords2d_list.append( coords2d_read(fname) )
            # coords2d_list[-1] = coords2d_list[-1].transpose()
            # print(coords2d_list[-1])
            # plt.plot(coords2d_list[-1][0],coords2d_list[-1][1],marker='o',ls='')
            # plt.gca().invert_yaxis()
            # plt.show()
    
    coords3d_list = None
    # coords3d_list = combine_coords_2d_to_3d(coords2d_list,n_images=n_images,camera_matrix=camera_matrix,distortions=distortions,new_camera_matrix=new_camera_matrix)
    
    if coords3d_list is None:
        coords3d_list = []
        for i in range(n_images*(n_images-1)//2):
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
    coords3d = None
    # this doesnt combine, just picks one and rotates it around
    coords3d = combine_coords3d(coords3d_list) 
    
    
    # for now, just pick one of them
    if coords3d is None:
        coords3d_ind = 1
        coords3d = coords3d_list[ coords3d_ind ]
    
    show_coords(coords3d)
    
    # Find bad
    # coords3d_flag_bad_coords(coords3d)
    
    # Fix missing
    
    
    # Calibrate direction of axes
    # calibrate_updown(coords3d)
    
    # Done, save
    np.savetxt("coords.txt",coords3d,header="x\ty\tz")
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.plot(coords3d.transpose()[0],coords3d.transpose()[1],coords3d.transpose()[2],marker='o',ls='',c='k')
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    main()
