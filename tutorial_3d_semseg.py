from tqdm import tqdm
from waymo_open_dataset.protos import segmentation_submission_pb2
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.enable_eager_execution()


FILENAME = '/home/derek/tools/waymo-open-dataset/waymo_format/validation/segment-260994483494315994_2797_545_2817_545_with_camera_labels.tfrecord'

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
for i, data in tqdm(enumerate(dataset)):
    frame_i = open_dataset.Frame()
    frame_i.ParseFromString(bytearray(data.numpy()))
    if frame_i.lasers[0].ri_return1.segmentation_label_compressed:
        frame = frame_i
        print(f'{i}th data has segmentation')
        break

print(frame.context.name)
print(frame.context.stats)

"""
Parse range images and camera projections given a frame.

Args:
frame: open dataset frame proto

Returns:
range_images: A dict of {laser_name,
	[range_image_first_return, range_image_second_return]}.
camera_projections: A dict of {laser_name,
	[camera_projection_from_first_return,
	camera_projection_from_second_return]}.
seg_labels: segmentation labels, a dict of {laser_name,
	[seg_label_first_return, seg_label_second_return]}
	NOTE: for each seg_label_n_return:
    	instance_id_image = semseg_label_image_tensor[...,0] 
  		semantic_class_image = semseg_label_image_tensor[...,1]
range_image_top_pose: range image pixel pose for top lidar.
"""
(range_images, camera_projections, segmentation_labels,
 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

print(
    f'segmentation_labels: {segmentation_labels[open_dataset.LaserName.TOP][0].shape.dims}')


# Visualize Segmentation Labels in Range Images
# plt.figure(figsize=(64, 20))

# def plot_range_image_helper(data, name, layout, vmin=0, vmax=1, cmap='gray'):
#     """Plots range image.

#     Args:
#       data: range image data
#       name: the image title
#       layout: plt layout
#       vmin: minimum value of the passed data
#       vmax: maximum value of the passed data
#       cmap: color map
#     """
#     plt.subplot(*layout)
#     plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#     plt.title(name)
#     plt.grid(False)
#     plt.axis('off')


# def get_semseg_label_image(laser_name, return_index):
#     """Returns semseg label image given a laser name and its return index."""
#     return segmentation_labels[laser_name][return_index]


# def show_semseg_label_image(semseg_label_image, layout_index_start=1):
#     """Shows range image.

#     Args:
#       show_semseg_label_image: the semseg label data of type MatrixInt32.
#       layout_index_start: layout offset
#     """
#     semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
#     semseg_label_image_tensor = tf.reshape(
#         semseg_label_image_tensor, semseg_label_image.shape.dims)
#     instance_id_image = semseg_label_image_tensor[..., 0]
#     semantic_class_image = semseg_label_image_tensor[..., 1]
#     plot_range_image_helper(instance_id_image.numpy(), 'instance id',
#                             [8, 1, layout_index_start], vmin=-1, vmax=200, cmap='Paired')
#     plot_range_image_helper(semantic_class_image.numpy(), 'semantic class',
#                             [8, 1, layout_index_start + 1], vmin=0, vmax=22, cmap='tab20')


# frame.lasers.sort(key=lambda laser: laser.name)
# show_semseg_label_image(get_semseg_label_image(
#     open_dataset.LaserName.TOP, 0), 1)
# show_semseg_label_image(get_semseg_label_image(
#     open_dataset.LaserName.TOP, 1), 3)


# Point Cloud Conversion and Visualization
def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
  """Convert segmentation labels from range images to point clouds.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
      points that are not labeled. 
      NOTE: two columns: 
      	instance_id_image = semseg_label_image_tensor[...,0] 
  		semantic_class_image = semseg_label_image_tensor[...,1]
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  point_labels = []
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0 # range > 0

    if c.name in segmentation_labels:
      sl = segmentation_labels[c.name][ri_index]
      sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
      sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    else:
      num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
      sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

    point_labels.append(sl_points_tensor.numpy())
  return point_labels


"""Convert range images to point cloud.
Args:
frame: open dataset frame
range_images: A dict of {laser_name, [range_image_first_return,
	range_image_second_return]}.
camera_projections: A dict of {laser_name,
	[camera_projection_from_first_return,
	camera_projection_from_second_return]}.
range_image_top_pose: range image pixel pose for top lidar.
ri_index: 0 for the first return, 1 for the second return.
keep_polar_features: If true, keep the features from the polar range image
	(i.e. range, intensity, and elongation) as the first features in the
	output range image.

Returns:
points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
	(NOTE: Will be {[N, 6]} if keep_polar_features is true.
cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
"""
points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose)
points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

point_labels = convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels)
point_labels_ri2 = convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels, ri_index=1)

# 3d points in vehicle frame. (ri2: second return)
points_all = np.concatenate(points, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)
print(f'points_all: {points_all.shape}')
print(f'points_all_ri2: {points_all_ri2.shape}')
# point labels.
point_labels_all = np.concatenate(point_labels, axis=0)
point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
print(f'point_labels_all: {point_labels_all.shape}')
print(f'point_labels_all_ri2: {point_labels_all_ri2.shape}')
# camera projection corresponding to each point.
cp_points_all = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
print(f'cp_points_all: {cp_points_all.shape}')
print(f'cp_points_all_ri2: {cp_points_all_ri2.shape}')
'''
points_all: (129605, 3)
points_all_ri2: (7362, 3)
point_labels_all: (129605, 2)
point_labels_all_ri2: (7362, 2)
cp_points_all: (129605, 6)
cp_points_all_ri2: (7362, 6)
'''