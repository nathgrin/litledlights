
import cv2
import time
import numpy as np

from utils import get_strip,clear

from leds import blink_binary

    
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
    Turn on each light in sequence and get brightest pixels
    
    
    
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
    t = 0
    
    # BG img
    ret, img_bg = cam.read()
    img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    
    led_xy = [None for x in range(nleds)]
        
    
    try:
        while True:
            
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            frame = cv2.subtract(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),img_bg)
            cv2.imshow("Cam", frame)
    
            t += 1
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32 or t%delta_t == 0:
                # SPACE pressed
                
                strip[ind-1] = color_off
                strip[ind]   = color_on
                strip.show()
                
                img_name = loc+"led_{}.png".format(ind)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                
                xy = find_light(frame)
                
                led_xy[ind] = xy
                
                
                ind += 1
                if ind == nleds:
                    print("We got em all")
                    break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        strip.fill( color_off )
        strip.show()
    
    return led_xy

def find_light(img)->tuple[float,float]:
    
    
    ind = img > 180
    
    x,y = np.mean(np.where(ind), axis=1)
        
    
    # print(x,y)
    
    return (x,y)
    
    

def main():
    
    # img = cv2.imread("_tmp/leD_24")
    # find_light(img)
    
        
    led_xy = sequential_fotography()
    print(led_xy)
    
    
if __name__ == "__main__":
    main()
