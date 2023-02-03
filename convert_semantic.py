
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os, random
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset.protos import segmentation_pb2 as spb # spb.Segmentation.Type.TYPE_TRUCK=2



'''
message Segmentation {
  enum Type {
    TYPE_UNDEFINED = 0;
    TYPE_CAR = 1;
    TYPE_TRUCK = 2;
    TYPE_BUS = 3;
    // Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction
    // vehicles, RV, limo, tram).
    TYPE_OTHER_VEHICLE = 4;
    TYPE_MOTORCYCLIST = 5;
    TYPE_BICYCLIST = 6;
    TYPE_PEDESTRIAN = 7;
    TYPE_SIGN = 8;
    TYPE_TRAFFIC_LIGHT = 9;
    // Lamp post, traffic sign pole etc.
    TYPE_POLE = 10;
    // Construction cone/pole.
    TYPE_CONSTRUCTION_CONE = 11;
    TYPE_BICYCLE = 12;
    TYPE_MOTORCYCLE = 13;
    TYPE_BUILDING = 14;
    // Bushes, tree branches, tall grasses, flowers etc.
    TYPE_VEGETATION = 15;
    TYPE_TREE_TRUNK = 16;
    // Curb on the edge of roads. This does not include road boundaries if
    // there’s no curb.
    TYPE_CURB = 17;
    // Surface a vehicle could drive on. This include the driveway connecting
    // parking lot and road over a section of sidewalk.
    TYPE_ROAD = 18;
    // Marking on the road that’s specifically for defining lanes such as
    // single/double white/yellow lines.
    TYPE_LANE_MARKER = 19;
    // Marking on the road other than lane markers, bumps, cateyes, railtracks
    // etc.
    TYPE_OTHER_GROUND = 20;
    // Most horizontal surface that’s not drivable, e.g. grassy hill,
    // pedestrian walkway stairs etc.
    TYPE_WALKABLE = 21;
    // Nicely paved walkable surface when pedestrians most likely to walk on.
    TYPE_SIDEWALK = 22;
  }
}
'''
from glob import glob
from tqdm import tqdm
import open3d as o3d


SEGMENTATION_COLOR_MAP = {
    spb.Segmentation.TYPE_UNDEFINED: [0, 0, 0],
    spb.Segmentation.TYPE_CAR: [0, 0, 142],
    spb.Segmentation.TYPE_TRUCK: [0, 0, 70],
    spb.Segmentation.TYPE_BUS: [0, 60, 100],
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    spb.Segmentation.TYPE_OTHER_VEHICLE: [61, 133, 198],
    spb.Segmentation.TYPE_MOTORCYCLIST: [180, 0, 0],
    spb.Segmentation.TYPE_BICYCLIST: [255, 0, 0],
    spb.Segmentation.TYPE_PEDESTRIAN: [220, 20, 60],
    spb.Segmentation.TYPE_SIGN: [246, 178, 107],
    spb.Segmentation.TYPE_TRAFFIC_LIGHT: [250, 170, 30],
    spb.Segmentation.TYPE_POLE: [153, 153, 153], # Lamp post, traffic sign pole etc.
    spb.Segmentation.TYPE_CONSTRUCTION_CONE: [230, 145, 56], # Construction cone/pole.
    spb.Segmentation.TYPE_BICYCLE: [119, 11, 32],
    spb.Segmentation.TYPE_MOTORCYCLE: [0, 0, 230],
    spb.Segmentation.TYPE_BUILDING: [70, 70, 70],
    spb.Segmentation.TYPE_VEGETATION: [107, 142, 35], # Bushes, tree branches, tall grasses, flowers etc.
    spb.Segmentation.TYPE_TREE_TRUNK: [111, 168, 220],
    spb.Segmentation.TYPE_CURB: [234, 153, 153], # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
    spb.Segmentation.TYPE_ROAD: [128, 64, 128], # Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    spb.Segmentation.TYPE_LANE_MARKER: [234, 209, 220], # Marking on the road that’s specifically for defining lanes such as single/double white/yellow lines.
    spb.Segmentation.TYPE_OTHER_GROUND: [102, 102, 102], # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    spb.Segmentation.TYPE_WALKABLE: [217, 210, 233], # Most horizontal surface that’s not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    spb.Segmentation.TYPE_SIDEWALK: [244, 35, 232], # Nicely paved walkable surface when pedestrians most likely to walk on.
}


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
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 
    0 for points that are not labeled.
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  point_labels = []
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    if c.name in segmentation_labels:
      sl = segmentation_labels[c.name][ri_index]
      sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
      sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    else:
      num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
      sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
      
    point_labels.append(sl_points_tensor.numpy())
  return point_labels


'''
Utilizing two returns yielded higher recall values of the building boundaries for the test data. 
The maximum improvement was from 0.7417 to 0.7948 for test data 1, and 0.7691 to 0.7851 for test 
data 2 in terms of recall value.
'''

def extract_points_images_labels(frame, returns=2):
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
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
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
        (NOTE: Will be {[N, 6]} if keep_polar_features is true. i.e. (range, intensity, elongation, x, y, z)
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)
    
    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels)
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1)
    
    # 3d points in vehicle frame.
    points_all_ri1 = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    points_all = np.concatenate([points_all_ri1, points_all_ri2], axis=0)
    # point labels.
    point_labels_all_ri1 = np.concatenate(point_labels, axis=0)
    point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    point_labels_all = np.concatenate([point_labels_all_ri1, point_labels_all_ri2], axis=0)
    # camera projection corresponding to each point.
    cp_points_all_ri1 = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
    cp_points_all = np.concatenate([cp_points_all_ri1, cp_points_all_ri2], axis=0)
    # NOTE points: x, y, z, range, intensity, and elongation
    # NOTE labels: instance_id, semantic_id
    if returns==2:
        return points_all, point_labels_all
    elif returns==1:
        return points_all_ri1, point_labels_all_ri1
    else:
        raise NotImplementedError
    

def save_pcd(points, labels, path):
    assert isinstance(points, np.ndarray)
    assert isinstance(labels, np.ndarray)
    labels = labels[:, 1]
    colors = np.array([SEGMENTATION_COLOR_MAP[i] for i in labels])/255
    # colors = colors[:, [2, 1, 0]]
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3d.core.Tensor(points[:, 3:6])
    # https://github.com/waymo-research/waymo-open-dataset/issues/93
    intensity = np.tanh(points[:, [1]])
    pcd.point['intensity'] = o3d.core.Tensor(intensity)
    pcd.point['colors'] = o3d.core.Tensor(colors)
    pcd.point['label'] = o3d.core.Tensor(np.expand_dims(labels, axis=1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.t.io.write_point_cloud(path, pcd, write_ascii=False)


def process_record_frame(i_data_record):
    i, data, tfrecord = i_data_record
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    if frame.lasers[0].ri_return1.segmentation_label_compressed:
        # print(f'processing:{tfrecord}-{i}, name:{frame.context.name}')
        path = f'{tfrecord}.{i}.pcd'.replace('waymo_format', 'waymo_semantic')
        if os.path.exists(path):
            return None
        points_all, point_labels_all = extract_points_labels(frame)
        save_pcd(points_all, point_labels_all, path)
        print(f'saved to {path}')
        return path
    

def process_record(tfrecord):
    print(f'processing: {tfrecord}')
    dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        path = f'{tfrecord}.{i}.pcd'.replace('waymo_format', 'waymo_semantic')
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            # print(f'processing:{tfrecord}-{i}, name:{frame.context.name}')
            # if os.path.exists(path):
            #     return None
            points_all, point_labels_all = extract_points_labels(frame)
            save_pcd(points_all, point_labels_all, path)
            print(f'saved to {path}')
            return path
        # elif 'testing' in tfrecord:
        #     points_all, point_labels_all = extract_points_labels(frame)
        #     save_pcd(points_all, point_labels_all, path)
        #     print(f'saved to {path}')

if __name__ == '__main__':
    dataset = 'training' # 'validation' # 'testing'
    records = glob(f'waymo/waymo_format/{dataset}/*.tfrecord')
    for record in tqdm(records):
        process_record(record)
    
    # random.shuffle(records)
    ## multithreading
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     list(tqdm(executor.map(process_record, records), total=len(records)))
    
    ## multiprocessing
    # for tfrecord in tqdm(records):
    #     print(f'processing: {tfrecord}')
    #     dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
    #     mapping_tuple = []
    #     for i, data in tqdm(enumerate(dataset), desc=tfrecord):
    #         mapping_tuple.append((i, data, tfrecord))
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         list(tqdm(executor.map(process_record_frame, mapping_tuple), total=len(mapping_tuple)))
    
    ## generate pcd list
    pcd_train = glob('waymo/waymo_semantic/training/*.pcd')
    pcd_train = [p.replace('waymo/waymo_semantic/', '') for p in pcd_train]
    print(f'train: {len(pcd_train)}')
    pcd_val = glob('waymo/waymo_semantic/validation/*.pcd')
    pcd_val = [p.replace('waymo/waymo_semantic/', '') for p in pcd_val]
    print(f'val: {len(pcd_val)}')
    pcd_test = glob('waymo/waymo_semantic/testing/*.pcd')
    pcd_test = [p.replace('waymo/waymo_semantic/', '') for p in pcd_test]
    print(f'test: {len(pcd_test)}')
    with open('waymo/waymo_semantic/train.txt', 'w') as f:
        f.write('\n'.join(pcd_train))
    with open('waymo/waymo_semantic/val.txt', 'w') as f:
        f.write('\n'.join(pcd_val))
    with open('waymo/waymo_semantic/test.txt', 'w') as f:
        f.write('\n'.join(pcd_test))
    with open('waymo/waymo_semantic/trainval.txt', 'w') as f:
        f.write('\n'.join(pcd_train + pcd_val))