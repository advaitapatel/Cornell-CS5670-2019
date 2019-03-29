import math
import cv2
import numpy as np
import random

import scipy
from scipy import spatial

eTranslate = 0
eHomography = 1

def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature 
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    A = np.zeros((num_rows, 9))

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN
        row = i * 2

        A[row] = [a_x, a_y, 1, 0, 0, 0, -b_x *a_x, -b_x*a_y, -b_x]
        A[row+1] = [0, 0, 0, a_x, a_y, 1, -b_y*a_x, -b_y*a_y, -b_y ]

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    # s is a 1-D array of singular values sorted in descending order
    # U, Vt are unitary matrices
    # Rows of Vt are the eigenvectors of A^TA.
    # Columns of U are the eigenvectors of AA^T.

    # Homography to be calculated
    H = np.eye(3)

    # BEGIN TODO 3
    # Fill the homography H with the appropriate elements of the SVD
    # TODO-BLOCK-BEGIN
    i = Vt.shape[0] - 1
    H[0] = Vt[i][0:3]
    H[1] = Vt[i][3:6]
    H[2] = Vt[i][6:9]

    H /= H[2][2]
    # TODO-BLOCK-END
    # END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''    

    # BEGIN TODO 4
    # Write this entire method.  You need to handle two types of 
    # motion models, pure translations (m == eTranslation) and 
    # full homographies (m == eHomography).  However, you should
    # only have one outer loop to perform the RANSAC code, as 
    # the use of RANSAC is almost identical for both cases.

    # Your homography handling code should call compute_homography.
    # This function should also call get_inliers and, at the end,
    # least_squares_fit.
    # TODO-BLOCK-BEGIN
    inliers = []

    for i in range(nRANSAC):
        idx = i % len(matches) 

        if (m == eTranslate):
            f1_point = f1[matches[idx].queryIdx]
            x_pt1 = f1_point.pt[0]
            y_pt1 = f1_point.pt[1]

            f2_point = f2[matches[idx].trainIdx]
            x_pt2 = f2_point.pt[0]
            y_pt2 = f2_point.pt[1]

            H = np.eye(3)
            H[0][2] = x_pt2-x_pt1
            H[1][2] = y_pt2-y_pt1

        else:
            match_set = []
            match_set.extend([matches[idx], matches[(idx+1)%len(matches)],
                matches[(idx+2)%len(matches)], matches[(idx+3)%len(matches)]])
            H = computeHomography(f1, f2, match_set)

        inliers_new = getInliers(f1, f2, matches, H, RANSACthresh)
        if len(inliers_new) > len(inliers):
            inliers = inliers_new

    M = leastSquaresFit(f1, f2, matches, m, inliers)
    M = M/M[2][2]
    # TODO-BLOCK-END
    # END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        # BEGIN TODO 5
        # Determine if the ith matched feature f1[id1], when transformed 
        # by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        #TODO-BLOCK-BEGIN
        f1_mat = np.zeros((3,1))
        f2_mat = np.zeros((3,1))
        f1_mat[:,0] = [f1[matches[i].queryIdx].pt[0],f1[matches[i].queryIdx].pt[1],1 ]
        f2_mat[:,0] = [f2[matches[i].trainIdx].pt[0], f2[matches[i].trainIdx].pt[1], 1]

        f_transform = np.dot(M, f1_mat)
        f_transform /= f_transform[2][0]
        distance = scipy.spatial.distance.euclidean(f_transform, f2_mat)
        if (distance < RANSACthresh):
            inlier_indices.append(i)
        # TODO-BLOCK-END
        # END TODO
    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate) 
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        # For spherically warped images, the transformation is a 
        # translation and only has two degrees of freedom.
        # Therefore, we simply compute the average translation vector
        # between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            # BEGIN TODO 6
            # Use this loop to compute the average translation vector
            # over all inliers.
            # TODO-BLOCK-BEGIN
            index = inlier_indices[i]
            f1_inlier = f1[matches[index].queryIdx]
            f2_inlier = f2[matches[index].trainIdx]
            u += (f2_inlier.pt[0] - f1_inlier.pt[0])
            v += (f2_inlier.pt[1] - f1_inlier.pt[1])
            # TODO-BLOCK-END
            # END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        # BEGIN TODO 7
        # Compute a homography M using all inliers.
        # This should call computeHomography.
        # TODO-BLOCK-BEGIN
        matches_indices = []
        for index in inlier_indices:
            matches_indices.append(matches[index])
        M = computeHomography(f1, f2, matches_indices)
        M = M/M[2][2]
        # TODO-BLOCK-END
        # END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

