# import cv2 as cv # ...stupid example
import cv2
import numpy as np
import os
import config

def getpictures():
    """example from stackoverflow, in turn stolen from the "docs" """

    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    
    img_counter = 0
    img_list = []
    
    print(" > Space to save img, Escape to quit ")
    
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
            img_name = os.path.join("_tmp","calibrateframe_{}.png".format(img_counter))
            cv2.imwrite(img_name, frame)
            img_list.append(frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    
    return img_list
    
def get_objpoints(img_list,ncorners_xy):
    n,m = ncorners_xy#7,7#4,3
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n*m,3), np.float32)
    objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    nfound = 0
    for i,img in enumerate(img_list):
        print("image {0}, space to continue".format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (m,n),  flags=None)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("  found corners")
            nfound += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) # refine corners
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (m,n), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            print("  failed corners")
            
        # input()
    
    cv2.destroyAllWindows()
    
    print("found",nfound)
    
    return imgpoints,objpoints,gray.shape[::-1]

def main():
    """
    https://docs.opencv2.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    print("Calibratecamera")
    
    
    loc = "_tmp"
        
    img_list = None
    theinput = input("Get new pictures? (y to do so, otherwise no): ")
    if theinput == "y":
        img_list = getpictures()
    
    if img_list is None:
        img_list = []
        for img_counter in range(config.calibratecamera_nimg):
            img_name = os.path.join(loc,"calibrateframe_{}.png".format(img_counter))
            print(img_name,os.path.exists(img_name))
            img = cv2.imread(img_name)
            img_list.append(img)
    
    ncorners_xy = (7,7)
    while True:
        print("Using ncorners_xy={0}".format(ncorners_xy))
        try:
            theinput = input("ok? (y to continue, tuple of ints to try again with new ncorners_xy): ")
            if theinput == "y":
                ok = True
                break
            else:
                ncorners_xy =  tuple(int(x) for x in theinput.split(","))
                break
        except:
            pass
        print("Failed input.. noob")
    
    imgpoints,objpoints,resolution = get_objpoints(img_list,ncorners_xy)
    input("done get_objpoints, enter to continue")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, resolution, None, None)
    
    
    fname = os.path.join(loc,"cameramtx.txt")
    print("fname",fname)
    distortion_str = "distortions = np.array({0})".format(dist )
    print(distortion_str)
    
    camera_matrix_str = "camera_matrix = np.array({0})".format(mtx)
    print(camera_matrix_str)
    
    
    # img = cv2.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    new_camera_matrix_str = "new_camera_matrix = np.array({0})".format(newcameramtx)
    print(new_camera_matrix_str)
    
    with open(fname,'a') as thefile:
        thefile.write(distortion_str)
        thefile.write(camera_matrix_str)
        thefile.write(new_camera_matrix_str)
    
if __name__ == "__main__":
    main()

