import cv2
import numpy as np

def find_light(img,blob_params=None):
    
    
    
    cv2.imshow("Cam",img)
    
    if blob_params is not None:
        detector = cv2.SimpleBlobDetector_create(blob_params)
    else:
        detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(img)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    import numpy as np
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("blob",im_with_keypoints)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def main():
    blob_params = cv2.SimpleBlobDetector_Params()
    # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    blob_params.filterByColor = True
    blob_params.blobColor = 0 
    blob_params.minThreshold = 100 # from where to start filtering the image
    blob_params.maxThreshold = 200.0 # where to end filtering the image
    blob_params.thresholdStep = 25 # steps to go through
    for i in range(24,50):
        img = cv2.imread("_tmp/led_%i.png"%(i), cv2.IMREAD_GRAYSCALE)
        find_light(img,blob_params=blob_params)
    
    
if __name__ == "__main__":
    main()