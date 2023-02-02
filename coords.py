
import cv2
import time
import numpy as np

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
                            
                            delta_t = 5,# in arbitrary units
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
    cv2.namedWindow("Cam")
    
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
    led_xy = [None for x in range(nleds)]
    
    start = time.now()
    
    print(" > Press space to start, b for new background image, f to toggle background subtract of preview")
    
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
            cv2.imshow("Cam", preview)
            
            frame = cv2.subtract(frame,img_bg)
    
            if started: t += 1
            
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print(" > Escape hit, closing...")
                led_xy = None
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
                
            elif k%256 == 32 or t%delta_t == 0:
                # SPACE pressed
                # print(t,started)
                if started:
                    print("TIME",delta_t,time.now()-start)
                    strip[ind-1] = color_off
                    strip[ind]   = color_on
                    strip.show()
                    
                    img_name = loc+"led_{}.png".format(ind)
                    cv2.imwrite(img_name, frame)
                    # print("{} written!".format(img_name))
                    
                    xy = find_light(frame)
                    
                    led_xy[ind] = xy
                    
                    
                    ind += 1
                    if ind == nleds:
                        print(" > We got em all")
                        break
                else:
                    print(" > Vamonos")
                    t = 0 # reset t because we can
                    started = True
    finally:
        cam.release()
        cv2.destroyAllWindows()
        strip.fill( color_off )
        strip.show()
    
    return led_xy

def find_light(img)->tuple[float,float]:
    """
    follow matt, simply get brightest pixels
    """
    
    
    
    
    ind = img > 180
    
    x,y = np.mean(np.where(ind), axis=1)
        
    
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
    return out
    

def get_coords2d_from_multiple_angles(n_images):
    
    
    
    coords2d_list = []
    
    for i in range(n_images):
        ok = False
        while not ok:
            led_xy = sequential_fotography()
            # print(led_xy)
            isok = input("You happy? Enter to accept, anything else to redo")
            ok = isok == ""
            if not ok:
                print("Not happy, try again")
            else:
                print(" > Next!")
        
        coords2d_list.append(led_xy)
        
        fname = "_tmp/"+"coords2d_{}.txt".format(i)
        coords2d_write(fname,led_xy)
    
    return coords2d_list


def combine_coords_2d_to_3d(coords2d_list: list[list[tuple[float,float]]],n_images: int=None) -> list[tuple[float,float,float]]:
    
    if coords2d_list is None:
        coords2d_list = []
        for i in range(n_images):
            fname = "_tmp/"+"coords2d_{}.txt".format(i)
            coords2d_list.append( coords2d_read(fname) )
    
    # print(coords2d)
    
    
    import itertools
    
    coords3d_list = []
    
    # For each pair of coord lists (image)
    for coords2d1,coords2d2 in itertools.combinations(coords2d_list,2):
        
        coords3d = triangulate( coords2d1,coords2d2 )
        
        # print(coords3d)
        coords3d_list.append(coords3d)

        
        coords3d = coords3d.transpose()
    
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
    
def triangulate(pts1,pts2):
    """
    from https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units
    """
    pts1,pts2 = np.array(pts1),np.array(pts2)
    # print(pts1,pts2)
    # ind = np.logical_or( np.isnan(pts1) , np.isnan(pts2) )
    # print(ind)
    # pts1,pts2 = pts1[ind],pts2[ind]
    
    
    cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
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
    n_images = 2 # how many images do we use
    
    coords2d_list = None
    coords2d_list = get_coords2d_from_multiple_angles(n_images)
    input("DONE")
    coords3d_list = combine_coords_2d_to_3d(coords2d_list,n_images=n_images)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for coords3d in coords3d_list:
        coords3d = coords3d.transpose()
        ax.scatter(coords3d[0], coords3d[1], coords3d[2], marker='o')
    
    
    plt.show()
    
    # Combine
    
    # Fix missing
    
    
    
    
if __name__ == "__main__":
    main()
