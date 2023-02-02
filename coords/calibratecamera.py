import cv2 as cv # ...stupid example
import cv2
import numpy as np

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
            img_name = "_tmp/calibrateframe_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            img_list.append(frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    
    return img_list
    
def get_objpoints(img_list):
    n,m = 4,3
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n*m,3), np.float32)
    objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    nfound = 0
    
    for img in img_list:
        # print("image")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (m,n),  flags=None)
        # cv.imshow('img', img)
        # cv.waitKey(0)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("found corners")
            nfound += 1
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (m,n), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(50)
        else:
            print("failed corners")
            
        # input()
    
    cv.destroyAllWindows()
    
    print("found",nfound)
    
    return imgpoints,objpoints,gray.shape[::-1]

def main():
    """
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    print("Calibrate")
    
    img_list = None
    # img_list = getpictures()
    
    if img_list is None:
        img_list = []
        for img_counter in range(9):
            img_name = "_tmp/calibrateframe_{}.png".format(img_counter)
            img = cv2.imread(img_name)
            img_list.append(img)
            
    
    imgpoints,objpoints,resolution = get_objpoints(img_list)
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, resolution, None, None)
    
    print("first matrix",mtx)
    np.savetxt("_tmp/"+"cameramtx.txt", mtx)
    
    # img = cv.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    print("secondmatrix", newcameramtx)
    np.savetxt("_tmp/"+"newcameramtx.txt",newcameramtx)
    
if __name__ == "__main__":
    main()

