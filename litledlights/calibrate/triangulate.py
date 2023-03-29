import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def triangulate(pts1,pts2, camera_matrix=None):
    """
    from https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units
    """
    pts1,pts2 = np.asarray(pts1),np.asarray(pts2)
    # print(pts1,pts2)
    # ind = np.logical_or( np.isnan(pts1) , np.isnan(pts2) )
    # print(ind)
    # pts1,pts2 = pts1[ind],pts2[ind]
    
    if camera_matrix is None:
        cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
    else:
        cameraMatrix = camera_matrix
    F,m1 = cv2.findFundamentalMat(pts1, pts2) # apparently not necessary

    # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations: 
    E,m2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
    # Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results. 
    # K_l = cameraMatrix
    # K_r = cameraMatrix
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts1, pts2, cameraMatrix,distanceThresh=0.5)
    # retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts_l_norm, pts_r_norm, cameraMatrix,distanceThresh=0.5)

    # given R,t you can  explicitly find 3d locations using projection 
    M_r = np.concatenate((R,t),axis=1)
    M_l = np.concatenate((np.eye(3,3),np.zeros((3,1))),axis=1)
    proj_r = np.dot(cameraMatrix,M_r)
    proj_l = np.dot(cameraMatrix,M_l)
    points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    points_3d = points_4d[:3, :].T
    return points_3d


def triangulate_iterate(inpts1,inpts2, camera_matrix=None):
    """
    DOESNT WORK
    from https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units
    """
    inpts1,inpts2 = np.asarray(inpts1),np.asarray(inpts2)
    # print(inpts1,inpts2)
    # ind = np.logical_or( np.isnan(inpts1) , np.isnan(inpts2) )
    # print(ind)
    # inpts1,inpts2 = inpts1[ind],inpts2[ind]
    
    if camera_matrix is None:
        cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
    else:
        cameraMatrix = camera_matrix
    
    leninpts = len(inpts1)
    rng = np.random.default_rng()
    
    outpts3d = []
    
    for i in range(50): # Iterate!
        # Select n random pts to calculate the transfo matrices
        # Then triangulate all and record
        ind = rng.choice(leninpts,size=650,replace=False)
        
        pts1,pts2 = inpts2[ind],inpts2[ind]
        
        F,m1 = cv2.findFundamentalMat(pts1, pts2) # apparently not necessary

        # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations: 
        E,m2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
        # Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

        # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results. 
        # K_l = cameraMatrix
        # K_r = cameraMatrix
        retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts1, pts2, cameraMatrix,distanceThresh=0.5)
        # retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts_l_norm, pts_r_norm, cameraMatrix,distanceThresh=0.5)

        # given R,t you can  explicitly find 3d locations using projection 
        M_r = np.concatenate((R,t),axis=1)
        M_l = np.concatenate((np.eye(3,3),np.zeros((3,1))),axis=1)
        proj_r = np.dot(cameraMatrix,M_r)
        proj_l = np.dot(cameraMatrix,M_l)
        points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, np.expand_dims(inpts1, axis=1), np.expand_dims(inpts2, axis=1))
        points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
        points_3d = points_4d[:3, :].T
        
        outpts3d.append(points_3d)

        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot(points_3d.transpose()[0],points_3d.transpose()[1],points_3d.transpose()[2],marker='o',ls='',c='k')
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$y$")
        # ax.set_zlabel("$z$")
        # plt.show()
    
    return np.nanmedian(outpts3d,axis=0)

def coords3d_from_iterative_LS_triangulation(coords2d1,coords2d2,camera_matrix):
    # First calculate projection matrix following opencv triangulate example
    # Second iterative LS triangulation from stolen triangulate lib
    
    
    pts1,pts2 = np.asarray(coords2d1),np.asarray(coords2d2)
    # print(inpts1,inpts2)
    # ind = np.logical_or( np.isnan(inpts1) , np.isnan(inpts2) )
    # print(ind)
    # inpts1,inpts2 = inpts1[ind],inpts2[ind]
    
    if camera_matrix is None:
        cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
    else:
        cameraMatrix = camera_matrix
        
    F,m1 = cv2.findFundamentalMat(pts1, pts2) # apparently not necessary

    # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations: 
    E,m2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
    # Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results. 
    # K_l = cameraMatrix
    # K_r = cameraMatrix
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts1, pts2, cameraMatrix,distanceThresh=0.5)
    # retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts_l_norm, pts_r_norm, cameraMatrix,distanceThresh=0.5)

    # given R,t you can  explicitly find 3d locations using projection 
    M_r = np.concatenate((R,t),axis=1)
    M_l = np.concatenate((np.eye(3,3),np.zeros((3,1))),axis=1)
    proj_r = np.dot(cameraMatrix,M_r)
    proj_l = np.dot(cameraMatrix,M_l)
    
    # NOW we use the stolen function. It appears to give almost exact same results as opencv but well..
    from calibrate.triangulation_stolen import iterative_LS_triangulation
    coords3d,flags = iterative_LS_triangulation(coords2d1,proj_l,coords2d2,proj_r)
    
    ind = flags != 1
    coords3d[ind] = np.nan
    # print(coords3d,flags)
    return coords3d
    
def triangulate_full(pts1,pts2, camera_matrix=None):
    """
    from https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units
    """
    pts1,pts2 = np.array(pts1),np.array(pts2)
    # print(pts1,pts2)
    # ind = np.logical_or( np.isnan(pts1) , np.isnan(pts2) )
    # print(ind)
    # pts1,pts2 = pts1[ind],pts2[ind]
    
    if camera_matrix is None:
        cameraMatrix = np.array([[1, 0,0],[0,1,0],[0,0,1]])        
    else:
        cameraMatrix = camera_matrix
    F,m1 = cv2.findFundamentalMat(pts1, pts2) # apparently not necessary

    # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations: 
    E,m2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
    Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results. 
    K_l = cameraMatrix
    K_r = cameraMatrix
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts1, pts2, cameraMatrix,distanceThresh=0.5)
    # retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E,pts_l_norm, pts_r_norm, cameraMatrix,distanceThresh=0.5)

    # given R,t you can  explicitly find 3d locations using projection 
    M_r = np.concatenate((R,t),axis=1)
    M_l = np.concatenate((np.eye(3,3),np.zeros((3,1))),axis=1)
    proj_r = np.dot(cameraMatrix,M_r)
    proj_l = np.dot(cameraMatrix,M_l)
    points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    points_3d = points_4d[:3, :].T
    return points_3d
  
  
  
def combine_pair_coords2d(coords2d1: np.ndarray,coords2d2: np.ndarray,
                          camera_matrix=None,distortions:np.ndarray=None,new_camera_matrix:np.ndarray=None):
    
    
    # First undistort
    if distortions is not None:# and new_camera_matrix is not None: # This doesnt work!
        print("Undistorting!")
        if new_camera_matrix is None:
            new_camera_matrix = camera_matrix
            
        undistorted = cv2.undistortPoints(coords2d1, camera_matrix, distortions, P=new_camera_matrix) 
        undistorted = np.squeeze(undistorted)
        coords2d1 = undistorted
        undistorted = cv2.undistortPoints(coords2d2, camera_matrix, distortions, P=new_camera_matrix) 
        undistorted = np.squeeze(undistorted)
        coords2d2 = undistorted
    
    # Then triangulate
    from calibrate.triangulate import coords3d_from_iterative_LS_triangulation
    coords3d = coords3d_from_iterative_LS_triangulation(coords2d1,coords2d2,camera_matrix)
    
    return coords3d
    

def combine_coords_2d_to_3d(coords2d_list: list[np.ndarray],n_viewpoints: int=None,camera_matrix=None,distortions:np.ndarray=None,new_camera_matrix:np.ndarray=None) -> list[tuple[float,float,float]]:
    # this needs to be reorganised to a do_one_ type functions
    
    # print(coords2d)
    
    # Undistort
    input("I BROKE THIS FUNC")
    import itertools
    
    coords3d_list = []
    cnt = 0
    
    # For each pair of coord lists (image)
    # for coords2d1,coords2d2 in itertools.combinations(coords2d_list,2):
    for ind1, ind2 in itertools.combinations(range(len(coords2d_list)),2):
        print("cnt",cnt,"ind1 (k) ind2 (r)",ind1,ind2)
        coords2d1,coords2d2 = coords2d_list[ind1],coords2d_list[ind2]
        
        coords3d = combine_pair_coords2d(coords2d1,coords2d2, camera_matrix=camera_matrix)
        
        # print(coords3d)
        coords3d_list.append(coords3d)
        
        np.savetxt( os.path.join("_tmp","coords3d_{}.txt".format(cnt)) ,coords3d)
    
        cnt += 1
        # coords3d = coords3d.transpose()
    
    return coords3d_list
