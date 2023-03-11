import config
import os
import cv2
import numpy as np


def find_light(img,
               blur_radius: int=9, # has to be odd!
               threshold: int=None,
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
    
    return (x,y)
    
    
def main():
    
    print("findlights")
    print(config.sequentialfotography_loc)
    loc = config.sequentialfotography_loc
    
    
    cv2.namedWindow("Images")
    cv2.namedWindow("Zoomed")
    # cv2.namedWindow("Blur")
    
    radius = 15
    
    for ind in range(700):
        # Get image
        fname = os.path.join(loc,"led_{0}.png".format(ind))
        print(ind,fname)
        img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
        # print(img)
        
        # Find light
        res = find_light(img)
        print("  result",res)
        if not np.isnan(res[0]):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Convert to c
            
            x,y = int(res[0]),int(res[1])
            img = cv2.circle(img,(x,y),radius,(0,155,0),0)
            
            crop = img[y-radius:y+radius,x-radius:x+radius]
            crop = cv2.resize(crop,(400,400))
            cv2.imshow("Zoomed",crop)
        
        
        # Show
        cv2.imshow("Images",img)
        
        k = cv2.waitKey(15)
        if k%256 == 27:
            # ESC pressed
            print(" > Escape hit, closing...")
            break
        
    
if __name__ == "__main__":
    main()
