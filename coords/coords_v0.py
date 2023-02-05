
import cv2
import time

from utils import get_strip,clear

# from leds import blink_binary

    
def tst():
    """example from stackoverflow"""

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
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


def interrupted_imaging(strip=None,order:list[str]=[],
                        fmt_fname: callable=None,
                        loc="_tmp/"
                        ):
    def instruct_lights(strip,current,binary_ind,bins,color_on=(128,128,128),color_off=(0,0,0)):
        if current == "on":
            strip.fill(color_on)
        elif current == "binary":
            for l in range(len(strip)):
                strip[l] = color_on if int(bins[l][binary_ind]) else color_off
            
        else:
            strip.fill(color_off)
            
        strip.show()
            
        
    nbits = None
    
    # strip
    strip = get_strip() if strip is None else strip
    
    # Bit preparation
    inlist = range(len(strip))
    bins = [format(n, 'b') for n in inlist]
    lens = [ len(n) for n in bins ]
    nbits = max(lens) if nbits is None else max(max(lens),nbits) # Watch out, theres no warning here!
    bins = [ x.zfill(nbits) for x in bins ]
    
    
    # Setup
    color_on = (32,32,32)
    
    # initialise
    ind = 0
    current = order[ind]
    binary_ind = 0
    
    strip.fill( (0,0,0) )
    strip.show()
 
    # CAM
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("tst")
 
    with strip:
        
        try:
            
            while True:
                ret,frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                cv2.imshow("tst",frame)
                
                k = cv2.waitKey(25)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = fmt_fname(current,binary_ind)
                    cv2.imwrite(loc+img_name, frame)
                    print("{} written!".format(loc+img_name))
                    
                    
                    if current != "binary":
                        ind += 1
                        current = order[ind]
                    else:
                        binary_ind += 1
                        if binary_ind == nbits:
                            print("DONE binary")
                            break
                    instruct_lights(strip,current,binary_ind,bins,color_on=color_on)
            
        finally: 
            cam.release()
            cv2.destroyAllWindows()
    
 
def get_blob_params():
    # some guy on stack overflow put this together claiming it fit his purpose (edited)
    blob_params = cv2.SimpleBlobDetector_Params()
    # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    blob_params.filterByColor = True
    blob_params.blobColor = 255

    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    # blob_params.filterByArea = True
    # blob_params.minArea = 3. # Highly depending on image resolution and dice size
    # blob_params.maxArea = 400. # float! Highly depending on image resolution.

    # blob_params.filterByCircularity = True
    # blob_params.minCircularity = 0. # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
    # blob_params.maxCircularity = 3.4028234663852886e+38 # infinity.

    # blob_params.filterByConvexity = False
    # blob_params.minConvexity = 0.
    # blob_params.maxConvexity = 3.4028234663852886e+38

    # blob_params.filterByInertia = True # a second way to find round blobs.
    # blob_params.minInertiaRatio = 0.55 # 1 is round, 0 is anywhat 
    # blob_params.maxInertiaRatio = 3.4028234663852886e+38 # infinity again

    blob_params.minThreshold = 150 # from where to start filtering the image
    blob_params.maxThreshold = 255.0 # where to end filtering the image
    blob_params.thresholdStep = 10 # steps to go through
    blob_params.minDistBetweenBlobs = 3.0 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution! 
    blob_params.minRepeatability = 2 # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >= minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.
    return blob_params
 
def img_find_lights(order: list[str]=[],
                    loc="_tmp/",
                    fmt_fname: callable=None):
    import matplotlib.pyplot as plt
    
    current = order[2]
    
    cv2.namedWindow("tst")
    
    img_on  = cv2.imread(loc+fmt_fname("on",0))
    img_off = cv2.imread(loc+fmt_fname("off",0))
    img = cv2.difference(img_off,img_on)
    # img = img_on
    # cv2.imwrite("img_subtract.png",img)
    # img = cv2.bitwise_not(img) # blob detection wants inverted img
    # img = cv2.medianBlur(img, 9)
 
    cv2.imshow("tst",img_on)
    blob_params = get_blob_params()
    # blob_params = cv2.SimpleBlobDetector_Params()
 
    # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    # blob_params.filterByColor = False
    # blob_params.blobColor = 0 
    # blob_params.minDistBetweenBlobs = 3.0 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution! 
    # blob_params.minRepeatability = 2 # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >= minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.
    hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist_full)
    plt.show()
    
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(img)
    print(keypoints[0])
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    import numpy as np
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
 
    cv2.waitKey(0)
 
 
 
    cv2.destroyAllWindows()
    
    
    
 
def img_find_lights_bgsubtraction(order: list[str]=[],
                    loc="_tmp/",
                    fmt_fname: callable=None):
    import matplotlib.pyplot as plt
    
    current = order[2]
    
    cv2.namedWindow("tst")
    
    img_on  = cv2.imread(loc+fmt_fname("on",0))
    img_off = cv2.imread(loc+fmt_fname("off",0))
    
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    fgmask = fgbg.apply(img_on)
    
    cv2.imshow("tst",fgmask)
    
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    
    

def main():
    def fmt_fname(current,binary_ind):
        fname = "img_"
        if current == "binary":
            fname += "binary{}".format(binary_ind)
        else:
            fname += current
        fname += ".png"
        return fname
        

    # strip = get_strip()

    loc = "_tmp/"
    order = [ "rdycheck","off","on","binary" ]

    # interrupted_imaging(strip=strip,order=order,
    #                  loc=loc,
    #                  fmt_fname=fmt_fname)

    img_find_lights_bgsubtraction(order,loc=loc,fmt_fname=fmt_fname)
 

    
    
if __name__ == "__main__":
    main()
