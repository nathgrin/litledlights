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
getcoords2d_nviewpoints = 3

getcoords2d_fromangles = False
do_2d_to_3d = True

# Sequential_photography
sequentialfotography_coloron = (215,215,215) # tuple[int,int,int]
sequentialfotography_deltat = 10 # int, how many cycles (frames) between images
sequentialfotography_loc = "_tmp" # str, location for storing imgs

# Find light in image
findlight_threshold = 30

# combine_coords3d: Combine multiple 3d to single set
combinecoords3d_referenceinds_default = 152,156,150 # Origin (red), z-point/unit length (blue), x-point (red): 
combinecoords3d_ind_coords3d = 2 # for now, simply pick one

# Fix bad coords
coords3dflagbadcoords_cutoff = 1.3#None
coords3d_fixbad_splineorder = 3

# Saving
save_coords3d = True
savecoords3d_fname = coords3d_fname

## Camera calibration
# Somehow things work better when new_camera_mtx=new_camera_mtx = new_camera_mtx
# Better camera
distortions = np.array([[-0.00745411  ,0.19816095 , 0.00370116 , 0.00352454 ,-0.44702301]])
# camera_matrix = [[535.52442405   0.         336.52314375]
 # [  0.         539.77321451 289.86230064]
 # [  0.           0.           1.        ]] # first camera mtx
camera_matrix = np.array([[338.83865356 ,0.      ,   436.99901032],
 [  0.      ,   327.93417358 ,361.97895928],
 [  0.     ,      0.         ,  1.        ]]) # new camera mtx
new_camera_matrix = None


# CRAPPY CAMERA
# camera_matrix = np.array( [[794.0779295,0.,334.41476339],
#                            [  0.,790.21709526,248.42875997],
#                            [  0.,0.,1.        ]] )  # first camera mtx
# distortions = np.array( [[ 2.05088975e-01 ,-1.09124274e+00 , 4.90025360e-04 , 1.83144614e-02   ,    2.58532256e+00]] )
# camera_matrix = np.array( [[802.89550781,0,340.40239924],
                            # [  0,793.20324707 ,247.94272481],
                            # [  0,0,1.        ]]) # newcameramtx
# new_camera_matrix = None
