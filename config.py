nleds = 700
dbg = False



# Calibrate

## Get coords
getcoords2d_nviewpoints = 2

getcoords2d_fromangles = True
do_2d_to_3d = False

### Find light in image
findlight_threshold = 80

### combine_coords3d: Combine multiple 3d to single set
combinecoords3d_referenceinds_default = 150,142,185 # Origin (red), z-point/unit length (blue), x-point (red): 
ind_coords3d = 0 # for now, simply pick one

### Saving
save_coords3d = False
savecoords3d_fname = "coords.txt"