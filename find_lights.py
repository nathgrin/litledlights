import cv2
import numpy as np

def find_light(img):
    
    
    cv2.namedWindow("Cam")
    
    cv2.imshow("Cam",img)
    otsu_threshold, image_result = cv2.threshold(img, 160, 255 ,cv2.THRESH_BINARY)
    cv2.imshow("new",image_result)
    ind = np.where( img > 180 )
    print(ind)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def main():
    img = cv2.imread("_tmp/led_24.png", cv2.IMREAD_GRAYSCALE)
    find_light(img)
    
    
if __name__ == "__main__":
    main()