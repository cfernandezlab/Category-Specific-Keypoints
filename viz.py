import os
import numpy as np
import trimesh
from pathlib import Path
import glob
import scipy.io as sio
from numpy import linalg as LA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances


def cluster(data, epsilon,N): 
    db = DBSCAN(eps=epsilon, min_samples=N).fit(data)
    labels = db.labels_ #labels of the found clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #number of clusters
    clusters = [data[labels == i] for i in range(n_clusters)] #list of clusters
    return clusters, n_clusters


def apply_nms(keypoints_pred, minDistance):
    """ Non-Maximal Suppression to get final keypoints. """
    clusters, n_clusters = cluster(keypoints_pred.transpose(),minDistance,1)
    keypoints = np.asarray([np.mean(c,0) for c in clusters])
    return keypoints


def get_reflection_operator(n_pl):
    norm_npl = LA.norm(n_pl)
    n_x = n_pl[0][0]/norm_npl
    n_y = 0/norm_npl
    n_z = n_pl[0][1]/norm_npl
    refl_mat = [[1-2*n_x*n_x, -2*n_x*n_y, -2*n_x*n_z], [-2*n_x*n_y, 1-2*n_y*n_y, -2*n_y*n_z], [-2*n_x*n_z, -2*n_y*n_z, 1-2*n_z*n_z]]
    return refl_mat


def get_kpts_from_non_rigid_modelling(params_pred):
    """ Apply non rigid deformation modelling with instance-wise symmetry.
    
    Parameters
    ----------
    params_pred: network prediction:
        basis: low-rank shape basis with instance-wise symmetry. [Kx3xkp/2]
        coef: coefficients that linearly combines the basis [K]
        n_pl: normal vector of the plane of symmetry passing through the origin 

    Returns
    -------
    keypoints: category-specific keypoints por the specific instance
    """
    basis = params_pred['BasisShapes'] 
    coef = params_pred['defCoefs'][0] 
    keypoints_pred = np.einsum('ijk,i->jk', basis, coef) # [3xkp/2]
    
    refl_mat = get_reflection_operator(params_pred['n_pl'])
    keypoints_pred_mirror = np.asarray(refl_mat).dot(keypoints_pred)
    keypoints_pred = np.concatenate((keypoints_pred, keypoints_pred_mirror), axis=1) # [3xkp]

    keypoints = apply_nms(keypoints_pred, minDistance = 0.2)
    return keypoints


if __name__ == "__main__":

    dataset = 'ShapeNet'
    category = 'airplane'
    ckpt_model = 'airplane_10b'

    root_data = './' + dataset + '/data/' + category + '/' 
    root_results = './' + dataset + '/results/' + ckpt_model + '/' 
    list_el = glob.glob(os.path.join(root_results, '*.mat'))


    for i, name in enumerate(list_el):
        # read input model
        input_id = Path(name).stem
        pc = np.load(os.path.join(root_data, '%s.npy' % input_id))
        if pc.shape[1] > 3:
            pc = pc[:,:3]
        pc_tri = trimesh.PointCloud(vertices=pc)

        # get prediction
        params_pred = sio.loadmat(name)
        keypoints = get_kpts_from_non_rigid_modelling(params_pred)
        keypoints_tri = trimesh.PointCloud(vertices=keypoints)

        # visualize
        scene = pc_tri.scene()
        for kp in keypoints:
            sphere = trimesh.primitives.Sphere(center=kp, radius=0.02)
            sphere.visual.face_colors = [255., 0., 0., 255.]
            scene.add_geometry(sphere)
        scene.show()


            

    
