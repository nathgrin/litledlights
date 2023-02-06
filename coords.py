
import cv2
import time
import numpy as np

import os

try:
    from utils import get_strip,clear

    from leds import blink_binary
except:
    print("imports failed..")

    
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

    
def sequential_fotography(strip=None,
                            color_off = (0,0,0),
                            color_on = (255,255,255),
                            
                            delta_t = 10,# in arbitrary units
                            loc = "_tmp/"
                            ):
    """example from stackoverflow, in turn stolen from the "docs" 
    
    like matt parker does it. 
    Turn on each light in sequence and 
    
    
    
    """
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
    
    print(" > Press space to start,\n b for new background image,\n f to toggle background subtract of preview")
    
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
            elif k == ord('f'):
                preview_subtract = not preview_subtract # toggle
                print("preview subtract turned",preview_subtract)
            elif k == ord('b'):
                # hit b
                # update background img
                print("update background..")
                ret, img_bg = cam.read()
                img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
                
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
                
                
                img_name = loc+"led_{}.png".format(ind)
                cv2.imwrite(img_name, frame)
                # print("{} written!".format(img_name))
                
                xy = find_light(frame)
                
                coords2d[ind] = xy
                
                print("Frame",ind,time.time()-start,xy)
                
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
    
    ind = blur > 100
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

def get_coords2d_from_multiple_angles(n_images):
    
    
    
    coords2d_list = []
    
    for i in range(n_images):
        ok = False
        while not ok:
            coords2d = sequential_fotography()
            # print(coords2d)
            if coords2d is not None:
                print("NaN/tot: {}/{}".format(np.sum(np.isnan(coords2d)),len(coords2d)))
                isok = input("You happy? Enter to accept, anything else to redo: ")
                ok = isok == ""
            else:
                ok = False
            if not ok:
                print("Not happy, try again")
            else:
                print(" > Next!")
        
        coords2d_list.append(coords2d)
        
        fname = os.path.join("_tmp","coords2d_{}.txt".format(i))
        coords2d_write(fname,coords2d)
    
    return coords2d_list


def combine_coords_2d_to_3d(coords2d_list: list[list[tuple[float,float]]],n_images: int=None,camera_matrix=None) -> list[tuple[float,float,float]]:
    
    if coords2d_list is None:
        coords2d_list = []
        for i in range(n_images):
            fname = "_tmp/"+"coords2d_{}.txt".format(i)
            coords2d_list.append( coords2d_read(fname) )
        #     coords2d_list[-1] = coords2d_list[-1].transpose()
        #     print(coords2d_list[-1])
        #     import matplotlib.pyplot as plt
        #     plt.plot(coords2d_list[-1][0],coords2d_list[-1][1],marker='o',ls='')
        # plt.show()
    
    # print(coords2d)
    
    
    import itertools
    
    coords3d_list = []
    cnt = 0
    
    # For each pair of coord lists (image)
    # for coords2d1,coords2d2 in itertools.combinations(coords2d_list,2):
    for ind1, ind2 in itertools.combinations(range(len(coords2d_list)),2):
        print(ind1,ind2)
        coords2d1,coords2d2 = coords2d_list[ind1],coords2d_list[ind2]
        
        
        
        coords3d = triangulate( coords2d1,coords2d2 ,camera_matrix=camera_matrix)
        
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
    

def main():
    
    # From calibatrion
    camera_matrix = np.array([ [1.06996313e+03,0.00000000e+00,3.15927309e+02],
                               [0.00000000e+00,7.98554626e+02,2.30317334e+02],
                               [0.00000000e+00,0.00000000e+00,1.00000000e+00]])

    
    n_images = 6 # how many images do we use
    
    coords2d_list = None
    # coords2d_list = get_coords2d_from_multiple_angles(n_images)
    # input("DONE")
    
    coords3d_list = None
    # coords3d_list = combine_coords_2d_to_3d(coords2d_list,n_images=n_images,camera_matrix=camera_matrix)
    
    if coords3d_list is None:
        coords3d_list = []
        for i in range(n_images*(n_images-1)//2):
            fname = os.path.join("_tmp","coords3d_{}.txt".format(i))
            coords3d_list.append( coords3d_read(fname) )
    
    print(len(coords3d_list))
    
    # find common non-nans
    any_isnan = np.sum(np.isnan(coords3d_list[0]),axis=1)
    for coords3d in coords3d_list:
        any_isnan = np.logical_or(any_isnan, np.sum(np.isnan(coords3d),axis=1) )
    
    ind_nonans = np.where(~any_isnan)
    
    ind1,ind2 = ind_nonans[0][0],ind_nonans[0][1]
    print(ind_nonans)
    print(ind1,ind2)
    
    print(coords3d_list[0][ind1])
    
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for coords3d in coords3d_list:
        
        
        norm = np.sqrt( np.sum( np.square(coords3d[ind1]-coords3d[ind2])) )
        
        coords3d = coords3d.transpose()
        
        for i in range(len(coords3d)):
            coords3d[i] = (coords3d[i]-coords3d[i][ind1])/norm
        ax.scatter(coords3d[0], coords3d[1], coords3d[2], marker='o')
    
    
    plt.show()
    
    # Combine
    
    # Fix missing
    
    
    
    
if __name__ == "__main__":
    main()
