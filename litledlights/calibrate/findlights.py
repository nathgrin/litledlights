import config
import os
import cv2
import numpy as np


def load_neuralnet(fname: str=None):
    
    fname = config.findlight_neuralnet_fname if fname is None else fname
    
    from ultralytics import YOLO
    
    nnmodel = YOLO(fname)
    
    return nnmodel
    
    

def find_light(img,
               *args,
               method:str = None,
               **kwargs
               )->tuple[float,float]:
    
    method = config.findlight_method if method is None else method
    
    if method == "simplematt":
        x,y = find_light_matt(img,*args,**kwargs)
    elif method == "neuralnet":
        x,y = find_light_neuralnet(img,*kwargs,**kwargs)
    else:
        raise ValueError("find_light: kwarg 'method' not recognized, has to be one of 'simplematt' or 'neuralnet'.")
    
    return (x,y)
    
def find_light_neuralnet(img,
                         nnmodel=None,return_ind=False)->tuple[float,float]:
    
    nnmodel = load_neuralnet() if nnmodel is None else nnmodel
    
    
    print("Neuralnet!")
    
    result = nnmodel(img)
    
    print(result)
    
    if True:
        pass
    else:
        y,x = np.nan,np.nan
        
    if return_ind:
        return (x,y),ind
    return (x,y)
    

def find_light_matt(img,
               blur_radius: int=9, # has to be odd!
               threshold: int=None,
               return_ind: float=False,
               )->tuple[float,float]:
    """
    follow matt, simply get brightest pixels
    """
    
    blur = cv2.GaussianBlur(img, (blur_radius,blur_radius),cv2.BORDER_DEFAULT)
    
    threshold = config.findlight_simplematt_threshold if threshold is None else threshold
    # print(threshold)
    ind = blur > threshold#80#100
    # print(ind)
    # print( np.mean(np.where(ind),axis=1))
    if np.sum(ind):
        y,x = np.average(np.where(ind),axis=1,weights=blur[ind])
    else:
        y,x = np.nan,np.nan
    
    # ind = img > 180
    
    # x,y = np.mean(np.where(ind), axis=1)
        
    # cv2.imshow("Blur",blur)
    # print(x,y)
    
    if return_ind:
        return (x,y),ind
    return (x,y)
    
    
def reprocess(loc: str="_tmp",nleds: int=None, dt: int=15, findlight_threshold: int=None):
    findlight_threshold = config.findlight_threshold if findlight_threshold is None else findlight_threshold
    nleds = config.nleds if nleds is None else nleds
    
    interrupted = False
    
    coords2d = [None for x in range(nleds)]
    
    cv2.namedWindow("Images")
    cv2.namedWindow("Zoomed")
    cv2.namedWindow("Thresholded")
    
    radius = config.reprocess_drawradius
    try:
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
                
                ymin,ymax = y-radius,y+radius
                xmin,xmax = x-radius,x+radius
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                crop = img[ymin:ymax,xmin:xmax]
            
            
            # Show
            cv2.imshow("Images",img)
            print(crop.shape)
            crop = cv2.resize(crop,(400,400))
            cv2.imshow("Zoomed",crop)
            
            thresh = img.copy()
            thresh[ind_thresh]  = 255
            thresh[~ind_thresh] = 0
            
            cv2.imshow("Thresholded",thresh)
            
            k = cv2.waitKey(dt)
            if k%256 == 27:
                # ESC pressed
                print(" > Escape hit, closing...")
                interrupted = True
                break
            elif k%256 == 32:
                # SPACE pressed
                
                print("PAUSING")
                k2 = cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
    
    if interrupted:
        return None
    return np.array(coords2d)
    
def main():
    
    print("findlights")
    print(config.sequentialfotography_loc)
    loc = config.sequentialfotography_loc
    
    reprocess(loc)
    
    
if __name__ == "__main__":
    main()
