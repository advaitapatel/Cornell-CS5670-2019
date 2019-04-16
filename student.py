# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions. # 9, 3 
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. # 9, [1920, 1080, 3]
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.

        # [1920, 1080, 3]
    """
    
    L_transpose = lights.T

    first_term = np.linalg.inv(np.matmul(L_transpose, lights)) 
    print np.shape(first_term)

    print np.shape(images)
    print np.shape(lights)

    shape = np.shape(images)
    
    img_matrix = np.zeros((shape[1], shape[2], shape[0], shape[3]))

    for count in xrange(shape[0]):
        for i in xrange(shape[1]):
            for j in xrange(shape[2]):
                img_matrix[i][j][count] = images[count][i][j]
    print np.shape(img_matrix)

    second_term = np.zeros((shape[1], shape[2], 3, 3))

    G = np.zeros((shape[1], shape[2], 3, 3))

    for i in xrange(shape[1]):
        for j in xrange(shape[2]):
            second_term[i][j] = np.matmul(L_transpose,img_matrix[i][j])
            G[i][j] = np.matmul(first_term,second_term[i][j])


    k_d = np.zeros((shape[1], shape[2], 3))
    N = np.zeros((shape[1], shape[2], 3))
    
    for i in xrange(shape[1]):
        for j in xrange(shape[2]):
            for channel in xrange(3):
                k_d[i][j][channel] = np.linalg.norm(G[i][j][channel])                
                N[i][j][channel] = G[i][j][:][channel]//k_d[i][j][channel]

    print np.shape(k_d)
    #print np.shape(N)
    return k_d




def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points # 2,2,3
    Output:
        projections -- height x width x 2 array of 2D projections
    """

    point_shape = np.shape(points)
    p = np.zeros((point_shape[0],point_shape[0],3))

    calib = np.dot(K, Rt)
    print np.shape(calib)

    projection = np.zeros((point_shape[0],point_shape[0],2))

    for i in xrange(point_shape[0]):
        for j in xrange(point_shape[0]):    
            val = np.ones((1,4))
            val[0, 0:3] = points[i][j][:] 
            proj  = np.dot(calib, val.T)
            proj_values = [proj[0]/proj[2], proj[1]/proj[2]]
            projection[i][j] = proj_values

    return projection



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x112, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    
    # (1) Compute and subtract the mean.

    print "preprocess_ncc_impl"

    image_shape = np.shape(image)
    channels = image_shape[2]
    height = image_shape[0]
    width = image_shape[1]

    normalized = np.zeros((height, width, (channels * ncc_size**2)))
    ncc_half = ncc_size/2

    # take window of ncc/2 up down left right if in boundary else put value 0 
    for i in xrange(height):
        for j in xrange(width):
            if i+ncc_half < height and i-ncc_half >= 0 and j+ncc_half < width and j-ncc_half >= 0:
                for ch in xrange(channels):
                    patch = np.ndarray.flatten(image[(i-ncc_half):(i+ncc_half+1),(j-ncc_half):(j+ncc_half+1),ch])
                    mean = np.mean(patch)
                    val = patch - mean
                    normalized[i][j][ch*ncc_size**2:(ch+1)*ncc_size**2] = val
               
                std = np.sqrt(np.sum(normalized[i][j][:]**2))
                if std < 1e-6:
                    normalized[i][j][:] = 0
                else:
                    normalized[i][j][:] /= std


    # loop over every pixel and mean the patch of the pixel per channel and store it as a flattened vector of every pixel

    # subtract mean of patch from the patch

    # ssd of pixel patch across all channel 

    # divide flatten vector by ssd if greater than 0

    # (2) Normalize the vector.
    
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """

    print "compute ncc impl"
    dot = image1 * image2
    ncc = np.sum(dot, axis = 2)
    return ncc