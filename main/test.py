import os
import os.path as osp
import json
import numpy as np
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import trimesh
from utils.display_utils import *
import torch

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    DATA_DIR = osp.join(ROOT_DIR, 'data')
    IMG_DIR = osp.join(DATA_DIR, 'Image')
    ANN_DIR = osp.join(DATA_DIR, '2D_json')
    SHAPE_PARAM_DIR = osp.join(DATA_DIR, 'Shape_param')

    # load json
    filename = '3D_08_F160D_250.json'
    actor_id = filename.split('_')[2]
    with open(osp.join(ROOT_DIR, filename), 'r') as f:
        ann = json.load(f)['annotations']
    rot = [[[ 8.7800e+01],
            [ 7.5433e+01],
            [ 8.7050e+01]],

           [[-5.3100e+00],
            [-8.1100e-01],
            [-3.8020e+01]],

           [[ 2.7700e+00],
            [ 1.7035e+01],
            [-3.5941e+01]],

           [[ 6.9900e-01],
            [ 1.1000e-02],
            [-1.3910e+00]],

           [[-3.2870e+00],
            [ 4.1810e+00],
            [ 3.7937e+01]],

           [[-3.5330e+00],
            [-2.0205e+01],
            [ 3.7044e+01]],

           [[ 4.3800e-01],
            [-3.1000e-02],
            [-1.5690e+00]],

           [[ 0.0000e+00],
            [-0.0000e+00],
            [ 0.0000e+00]],

           [[ 0.0000e+00],
            [-0.0000e+00],
            [ 0.0000e+00]],

           [[ 3.5000e-01],
            [-1.0000e-02],
            [-6.8000e-01]],

           [[ 6.6670e+00],
            [ 1.6540e+00],
            [-1.2459e+01]],

           [[-3.8540e+00],
            [-9.1200e-01],
            [-8.6800e+00]],

           [[-5.4900e-01],
            [-5.6000e-01],
            [ 3.5530e+00]],

           [[ 0.0000e+00],
            [-0.0000e+00],
            [ 0.0000e+00]],

           [[ 0.0000e+00],
            [-0.0000e+00],
            [ 0.0000e+00]],

           [[-1.9600e-01],
            [-4.7000e-01],
            [ 1.0660e+00]],

           [[ 6.3871e+01],
            [ 1.9220e+00],
            [-1.1482e+01]],

           [[-6.0841e+01],
            [-1.7846e+01],
            [-2.1906e+01]],

           [[ 1.0706e+01],
            [ 2.1100e+00],
            [-6.0265e+01]],

           [[-7.5220e+00],
            [ 5.4840e+00],
            [-7.3940e+00]],

           [[-3.0220e+00],
            [-2.3870e+00],
            [ 6.4000e-02]],

           [[-1.0000e+00],
            [-1.0770e+00],
            [-5.9000e-02]],

           [[ 0.0000e+00],
            [-0.0000e+00],
            [ 0.0000e+00]],

           [[ 0.0000e+00],
            [-0.0000e+00],
            [ 0.0000e+00]]]
    rot_t = []
    for i, r in enumerate(rot):
        x, y, z = r[0][0], r[1][0], r[2][0]
        x, y, z =  z, y, -x
        # if i == 0:
        #     x += 90
            # y += 180
            # z -= 90
        rot_t.append([[x], [y], [z]])


    ann['3d_rot'] = np.array(rot_t)
    # load parameter
    with open(osp.join(SHAPE_PARAM_DIR, '{}.json'.format(actor_id)), 'r') as f:
        shape_t = np.array(json.load(f)['shape_param'])
        shape_t = np.concatenate((shape_t, np.zeros(300 - len(shape_t))))
        shape_params = torch.Tensor(np.expand_dims(shape_t, axis=0))

    pose_t = np.apply_along_axis(eulerAnglesToRotationMatrix, 1, np.array(ann['3d_rot']))
    pose_params = torch.Tensor(np.expand_dims(pose_t.flatten(), axis=0))
    # trans_params = torch.Tensor(np.array(ann['trans_params']))
    trans_params = torch.Tensor(np.array([0, 0, 0]))

    # load shape parameter
    gender = 'female' if actor_id[0]=='F' else 'male'


    # init smpl
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=gender,
        model_root='smplpytorch/native/models')

    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params, th_trans=trans_params)
    vertice, faces = verts.numpy()[0], smpl_layer.th_faces.numpy()
    mesh = trimesh.Trimesh(vertice, faces)
    save_obj(mesh.vertices, mesh.faces, '{}.obj'.format(actor_id))
