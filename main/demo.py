from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from utils.display_utils import *
import torch
import trimesh
import numpy as np
import os
import os.path as osp
import json
import re


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    DATA_DIR = osp.join(ROOT_DIR, 'data')
    IMG_DIR = osp.join(DATA_DIR, 'Image')
    ANN_3D_DIR = osp.join(DATA_DIR, '3D_json')
    SHAPE_PARAM_DIR = osp.join(DATA_DIR, 'Shape_param')
    CAM_DIR = osp.join(DATA_DIR, 'Camera_json')
    # get logger
    logger = get_logger(logging.INFO)

    datasets = []
    # step (select by some frame)
    step = 1

    # load datasets
    logger.info('start loading datasets ...')
    for folder_name in os.listdir(ANN_3D_DIR):
        folder_path = osp.join(ANN_3D_DIR, folder_name)
        file_list = os.listdir(folder_path)
        sorted_file_list = sorted(file_list, key=lambda x: x.split('_')[-1].replace('.json', ''))
        for i in range(0, len(sorted_file_list), step):
            filename = sorted_file_list[i]
            file_path = osp.join(folder_path, filename)
            frame_no = filename.split('_')[-1].replace('.json', '')
            with open(file_path) as f:
                json_file = json.load(f)
            info = json_file['info']
            ann = json_file['annotations']
            # load camera parameter
            pattern = '{}_{}_.*'.format(info['action_category_id'], info['actor_id'])
            regex_cam = re.compile(pattern)

            cam_infos = []
            for (root, dirs, files) in os.walk(CAM_DIR):
                for file in files:
                    if not re.match(regex_cam, file):
                        continue
                    with open(osp.join(root, file)) as f:
                        json_file = json.load(f)
                    cam_infos.append({
                        'camera_no': json_file['camera_no'],
                        'intrinsics': np.array(json_file['intrinsics']),
                        'extrinsics': np.array(json_file['extrinsics'])
                    })
            # load shape params
            with open(osp.join(SHAPE_PARAM_DIR, '{}.json'.format(info['actor_id'])), 'r') as f:
                shape_t = np.array(json.load(f)['shape_param'])
                shape_t = np.concatenate((shape_t, np.zeros(300 - len(shape_t))))
            shape_params = np.expand_dims(shape_t, axis=0)
            # load pose params
            pose_t = np.apply_along_axis(eulerAnglesToRotationMatrix, 1, np.array(ann['3d_rot']))
            pose_params = np.expand_dims(pose_t.flatten(), axis=0)
            # load trans params
            trans_params = np.array(ann['trans_params'])

            for cam_info in cam_infos:
                # load images
                folder_name = '{}_{}_{}'.format(info['action_category_id'],info['actor_id'],cam_info['camera_no'])
                file_name = '{}_{}_{}_{}.jpg'.format(info['action_category_id'],info['actor_id'],cam_info['camera_no'], frame_no)
                img_path = osp.join(IMG_DIR, folder_name, file_name)
                # append to datasets
                datasets.append({
                    'gender': 'female' if info['actor_id'][0]=='F' else 'male',
                    'actor_id': info['actor_id'],
                    'shape_params': shape_params,
                    'pose_params': pose_params,
                    'trans_params': trans_params,
                    'intrinsics': cam_info['intrinsics'],
                    'extrinsics': cam_info['extrinsics'],
                    'frame_no': frame_no,
                    'folder_name': folder_name,
                    'file_name': file_name,
                    'image_path': img_path,
                    'image': cv2.imread(img_path)
                })
    logger.info('... end loading datasets')

    logger.info('start process')
    for dataset in datasets:
        # init smpl
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender=dataset['gender'],
            model_root='smplpytorch/native/models'
        )
        pose_params = torch.Tensor(dataset['pose_params'])
        shape_params = torch.Tensor(dataset['shape_params'])
        trans_params = torch.Tensor(dataset['trans_params'])
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params, th_trans=trans_params)
        vertice, faces = verts.numpy()[0], smpl_layer.th_faces.numpy()
        mesh = trimesh.Trimesh(vertice, faces)
        # save_obj(mesh.vertices, mesh.faces, '{}.obj'.format(dataset['folder_name']))
        image = dataset['image']
        mesh_vis = display_model2(image, mesh.vertices, dataset['intrinsics'], dataset['extrinsics'], image.shape[1], image.shape[0])
        cv2.imwrite('{}_{}.jpg'.format(dataset['folder_name'], dataset['frame_no']), mesh_vis)

    logger.info('end process')
    logger.info('result: ')



