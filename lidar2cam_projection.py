import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()
np.set_printoptions(precision=4, suppress=True)
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


FILENAME = 'tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
print(f'There are {len(list(dataset))} frames in the dataset')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    break

(range_images, camera_projections, _,
 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)



points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose)
points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose,
    ri_index=1)

# 3d points in vehicle frame.
points_all_ri1 = np.concatenate(points, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)
points_all = np.concatenate([points_all_ri1, points_all_ri2], axis=0)
# camera projection corresponding to each point.
cp_points_all_ri1 = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
cp_points_all = np.concatenate([cp_points_all_ri1, cp_points_all_ri2], axis=0)



images = sorted(frame.images, key=lambda i: i.name)
cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)
# The distance between lidar points and vehicle frame origin.
distance_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
# image 0's mask
img_idx = np.random.choice(len(images))
mask = tf.equal(cp_points_all_tensor[..., 0], images[img_idx].name)

cp_points_all_tensor = tf.cast(tf.gather_nd(
    cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
distance_all_tensor = tf.gather_nd(distance_all_tensor, tf.where(mask))

projected_points_all_from_raw_data = tf.concat(
    [cp_points_all_tensor[..., 1:3], distance_all_tensor], axis=-1).numpy()



import cv2
calibrations = sorted(frame.context.camera_calibrations, key=lambda c: c.name)
img_idx = np.random.choice(len(images))
c = calibrations[img_idx] 
h, w = c.height, c.width
print(c.name, h, w)
extrinsic = np.array(c.extrinsic.transform).reshape([4, 4])
extrinsic = np.linalg.inv(extrinsic)
rmat = extrinsic[0:3,0:3]
rvec, jacobian = cv2.Rodrigues(rmat)
tvec = extrinsic[0:3,3]
f_u, f_v, c_u, c_v, k_1, k_2, p_1, p_2, k_3 = c.intrinsic
cameraMatrix = np.array([
    [f_u, 0,   c_u],
    [0,   f_v, c_v],
    [0,   0,     1]])
distCoeffs = np.array([k_1, k_2, p_1, p_2, k_3])

distance = np.linalg.norm(points_all, axis=-1, keepdims=True)
# option1: use cv projection, not working...
imagePoints, jacobian = cv2.projectPoints(points_all, rvec, tvec, cameraMatrix, distCoeffs) 
imagePoints = imagePoints.squeeze()
print(imagePoints.shape)
# option2: manual projection, TODO: add distortion
xyz = (rmat @ points_all.T).T + tvec
xy = -xyz[:,1:3]/ xyz[:,[0]]
r2 = np.sum(xy**2, axis=-1)
# distortion
r_d = 1.0 + k_1 * r2 + k_2 * r2**2 + k_3 * r2**3
x_ = xy[:,0] * r_d + 2.0 * p_1 * xy[:,0] * xy[:,1] + p_2 * (r2 + 2.0 * xy[:,0]**2)
y_ = xy[:,1] * r_d + p_1 * (r2 + 2.0 * xy[:,1]**2) + 2.0 * p_2 * xy[:,0] * xy[:,1]
xy_ = np.concatenate([x_[...,np.newaxis], y_[...,np.newaxis]], axis=-1)
imagePoints = xy_ * np.array([f_u, f_v]) + np.array([c_u, c_v])

# filter
imagePoints = np.concatenate([imagePoints, distance], axis=-1)
mask = (imagePoints[:,0] > 0) & (imagePoints[:,0] < w) & (imagePoints[:,1] > 0) & (imagePoints[:,1] < h)
imagePoints = imagePoints[mask]
print(imagePoints.shape)