import config
import os
import cv2
import numpy as np


def find_light(img,
               blur_radius: int=9, # has to be odd!
               threshold: int=None,
               return_ind: float=False,
               )->tuple[float,float]:
    """
    follow matt, simply get brightest pixels
    """
    
    
    blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
    
    threshold = config.findlight_threshold if threshold is None else threshold
    # print(threshold)
    ind = blur > threshold#80#100
    # print(ind)
    # print( np.mean(np.where(ind),axis=1))
    y,x = np.mean(np.where(ind),axis=1)
    
    # ind = img > 180
    
    # x,y = np.mean(np.where(ind), axis=1)
        
    # cv2.imshow("Blur",blur)
    # print(x,y)
    
    if return_ind:
        return (x,y),ind
    return (x,y)
    
    
def postprocess(loc: str="_tmp",nleds: int=None, dt: int=15, findlight_threshold: int=None):
    findlight_threshold = config.findlight_threshold if findlight_threshold is None else findlight_threshold
    nleds = config.nleds if nleds is None else nleds
    
    coords2d = [None for x in range(nleds)]
    
    cv2.namedWindow("Images")
    cv2.namedWindow("Zoomed")
    cv2.namedWindow("Thresholded")
    
    radius = config.postprocess_drawradius
    for ind in range(nleds):
        # Get image
        fname = os.path.join(loc,"led_{0}.png".format(ind))
        print(ind,fname)
        img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
        # print(img)
        
        # Find light
        xy,ind_thresh = find_light(img,threshold=findlight_threshold,return_ind=True)
        coords2d[ind] = xy
                
        print("  result",xy)
        crop = cv2.bitwise_and(img[:2*radius,:2*radius],0)
        if not np.isnan(xy[0]):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Convert to c
            
            x,y = int(xy[0]),int(xy[1])
            img = cv2.circle(img,(x,y),radius,(0,155,0),0)
            
            crop = img[y-radius:y+radius,x-radius:x+radius]
        
        
        # Show
        cv2.imshow("Images",img)
        
        crop = cv2.resize(crop,(400,400))
        cv2.imshow("Zoomed",crop)
        
        thresh = img.copy()
        thresh[ind_thresh]  = 1
        thresh[~ind_thresh] = 0
        
        cv2.imshow("Thresholded",thresh)
        
        k = cv2.waitKey(dt)
        if k%256 == 27:
            # ESC pressed
            print(" > Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            
            print("PAUSING")
            k2 = cv2.waitKey(0)
    return np.array(coords2d)
    
def main():
    
    print("findlights")
    print(config.sequentialfotography_loc)
    loc = config.sequentialfotography_loc
    
    postprocess(loc)
    
    
if __name__ == "__main__":
    main()
