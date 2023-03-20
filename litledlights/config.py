import numpy as np
from coords import get_coords
print(".This.Is.Config.")

###
nleds = 700
dbg = False

connect_ledlights = True


coords3d_fname = "coords.txt"
coords3d = get_coords(coords3d_fname)

### Calibrate

## makecoords3d
getcoords2d_nviewpoints = 2

getcoords2d_fromangles = False
do_2d_to_3d = True

# Sequential_photography
sequentialfotography_skiptoreprocess = False # For the first viewpoint, skip to ?
sequentialfotography_coloron = (115,115,115) # tuple[int,int,int]
sequentialfotography_deltat = 6 # int, how many cycles (frames) between images
sequentialfotography_loc = "_tmp" # str, location for storing imgs

# Find light in image
findlight_threshold = 80
# reprocessing find lights
reprocess_drawradius = 15

# combine_coords3d: Combine multiple 3d to single set
combinecoords3d_referenceinds_default = 306,300,148 # Origin (red), z-point/unit length (blue), x-point (red): 
combinecoords3d_ind_coords3d = 0 # for now, simply pick one

# Fix bad coords
coords3dflagbadcoords_cutoff = 0.6#None# 1.3#None# Nones calculates this on fly
coords3d_fixbad_splineorder = 2

# Saving
save_coords3d = True
savecoords3d_fname = coords3d_fname

## Camera calibration
calibratecamera_nimg = 13
# Somehow things work better when new_camera_mtx=new_camera_mtx = new_camera_mtx
# Better camera
distortions = np.array([[ 0.0432809  ,-0.17178702 , 0.00300214 ,-0.00268438 , 0.10663246]])
camera_matrix = np.array([[638.72088052  , 0.      ,   309.964957  ],
 [  0.   ,      634.23924183 ,248.54710424],
 [  0.    ,       0.        ,   1.        ]])
new_camera_matrix = np.array([[633.53369141  , 0.  ,       307.62247084],
 [  0.        , 628.33099365, 249.52211068],
 [  0.      ,     0.   ,        1.        ]])


# distortions = np.array([[-0.03816212,0.44906195,0.01597279,0.00651138,-0.66210417]])
# camera_matrix = np.array([[627.83081055,0,330.48518022],
 # [  0,621.51800537, 277.32219422],
 # [  0,0,1.        ]])
new_camera_matrix= None
# new_camera_matrix = np.array([[627.83081055,0,330.48518022],
 # [  0,621.51800537, 277.32219422],
 # [  0,0,1.        ]])


# distortions = np.array([[-0.00745411  ,0.19816095 , 0.00370116 , 0.00352454 ,-0.44702301]])
# camera_matrix = np.array([[535.52442405,0,336.52314375],
 # [  0,539.77321451,289.86230064],
 # [  0,0,1.        ]]) # first camera mtx
# new_camera_matrix = np.array([[338.83865356 ,0.      ,   436.99901032],
 # [  0.      ,   327.93417358 ,361.97895928],
 # [  0.     ,      0.         ,  1.        ]]) # new camera mtx
# new_camera_matrix = None


# CRAPPY CAMERA
# camera_matrix = np.array( [[794.0779295,0.,334.41476339],
#                            [  0.,790.21709526,248.42875997],
#                            [  0.,0.,1.        ]] )  # first camera mtx
# distortions = np.array( [[ 2.05088975e-01 ,-1.09124274e+00 , 4.90025360e-04 , 1.83144614e-02   ,    2.58532256e+00]] )
# camera_matrix = np.array( [[802.89550781,0,340.40239924],
                            # [  0,793.20324707 ,247.94272481],
                            # [  0,0,1.        ]]) # newcameramtx
# new_camera_matrix = None
