import cv2
import numpy as np

def try_blobdetect_find_light(img,blob_params=None):
    
    
    
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
    print([ (x.pt) for x in keypoints])
    cv2.imshow("blob",im_with_keypoints)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
def try_blobdetect():
    
    blur_radius = 9
    
    blob_params = cv2.SimpleBlobDetector_Params()
    # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    blob_params.filterByColor = False
    blob_params.blobColor = 0 
    blob_params.minThreshold = 100 # from where to start filtering the image
    blob_params.maxThreshold = 200.0 # where to end filtering the image
    blob_params.thresholdStep = 5 # steps to go through
    for i in range(24,50):
        img = cv2.imread("_tmp/led_%i.png"%(i), cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
        try_blobdetect_find_light(blur,blob_params=blob_params)

def blur_find_light(img):
    blur_radius = 9 # has to be odd
    print(img)
    cv2.imshow("tst",img)
    
    blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
    print("img",maxVal,maxLoc)
    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)
    # blur = cv2.circle(blur, maxLoc, blur_radius, (255, 255, 0), 2)
    
    
    ind = img > 180
    x,y = np.mean(np.where(ind), axis=1)
    print("img",x,y)
    ind = blur > 120
    x,y = np.mean(np.where(ind), axis=1)
    print("blur",x,y)
    
    
    print(np.max(blur))
    print("blur",maxVal,maxLoc)
    print(blur[blur>200])
    print(np.sum(blur>150))
    cv2.imshow("blur",blur)
    
    
    import matplotlib.pyplot as plt
    
    
    for i in range(256):
        ind = img > i
        plt.plot(i,np.sum(ind),marker='o',c='k')
        ind = blur > i
        # print(np.where(ind))
        plt.plot(i,np.sum(ind),marker='o',c='r')
    plt.show()
    
    print(img.shape)
    print(blur.shape)
    hist_img = np.histogram(img.ravel(),bins=256,range=(0,256))
    hist_blur = np.histogram(blur.ravel(),bins=256,range=(0,256))
    plt.stairs(hist_img[0],hist_img[1],color='k')
    plt.stairs(hist_blur[0],hist_blur[1],color='r')
    plt.gca().set_yscale('log')
    plt.show()
    
    
    cv2.waitKey(0)

def blobdetect_secondtry(img):
    def get_blobparams():
        
        blob_params = cv2.SimpleBlobDetector_Params()

        # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
        blob_params.filterByColor = False
        blob_params.blobColor = 0 

        # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
        blob_params.filterByArea = True
        blob_params.minArea = 3. # Highly depending on image resolution and dice size
        blob_params.maxArea = 400. # float! Highly depending on image resolution.

        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.07 # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
        blob_params.maxCircularity = 1. # infinity.

        blob_params.filterByConvexity = False
        blob_params.minConvexity = 0.0001
        blob_params.maxConvexity = 1.

        blob_params.filterByInertia = True # a second way to find round blobs.
        blob_params.minInertiaRatio = 0.7 # 1 is round, 0 is anywhat 
        blob_params.maxInertiaRatio = 3.4028234663852886e+38 # infinity again

        blob_params.minThreshold = 50 # from where to start filtering the image
        blob_params.maxThreshold = 255.0 # where to end filtering the image
        blob_params.thresholdStep = 5 # steps to go through
        blob_params.minDistBetweenBlobs = 3.0 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution! 
        blob_params.minRepeatability = 2 # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >= minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.
        return blob_params
    
    blur_radius = 9
    blob_params = get_blobparams()
    
    for i in range(24,50):
        print('img ',i)
        img = cv2.imread("_tmp/led_%i.png"%(i), cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
        try_blobdetect_find_light(blur,blob_params=blob_params)

def blur_threshold(img):
    
    blur_radius = 9 # has to be odd
    print(img)
    cv2.imshow("tst",img)
    
    blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
    
    ind = blur > 100
    xy = np.mean(np.where(ind),axis=1)
    print(xy)
    print(tuple(xy))
    print(tuple(int(xx) for xx in xy[::-1]))
    blur = cv2.circle(blur, tuple(int(xx) for xx in xy[::-1]), blur_radius, (255, 255, 0), 2)
    cv2.imshow("blur",blur)
    
    cv2.waitKey(0)
    

def main():
    import os
    for i in range(164,200):
        img = cv2.imread(os.path.join("_tmp","led_%i.png"%(i)), cv2.IMREAD_GRAYSCALE)
        # blur_find_light(img)
        # blobdetect_secondtry(img)
        blur_threshold(img)
    
    
if __name__ == "__main__":
    main()