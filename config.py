import numpy as np
print(".This.Is.Config.")

###
nleds = 700
dbg = False



### Calibrate

## Get coords
getcoords2d_nviewpoints = 4

getcoords2d_fromangles = False
do_2d_to_3d = False

# Find light in image
findlight_threshold = 80

# combine_coords3d: Combine multiple 3d to single set
combinecoords3d_referenceinds_default = 148,144,230 # Origin (red), z-point/unit length (blue), x-point (red): 
ind_coords3d = 5 # for now, simply pick one

# Fix bad coords
coords3d_fixbad_splineorder = 3

# Saving
save_coords3d = True
savecoords3d_fname = "coords.txt"

## Camera calibration
# Somehow things work better when new_camera_mtx=new_camera_mtx = new_camera_mtx
# camera_matrix = np.array( [[794.0779295,0.,334.41476339],
#                            [  0.,790.21709526,248.42875997],
#                            [  0.,0.,1.        ]] )  # first camera mtx
distortions = np.array( [[ 2.05088975e-01 ,-1.09124274e+00 , 4.90025360e-04 , 1.83144614e-02   ,    2.58532256e+00]] )
camera_matrix = np.array( [[802.89550781,0,340.40239924],
                            [  0,793.20324707 ,247.94272481],
                            [  0,0,1.        ]]) # newcameramtx
new_camera_matrix = None