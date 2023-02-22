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
