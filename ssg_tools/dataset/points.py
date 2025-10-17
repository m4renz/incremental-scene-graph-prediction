
import numpy as np

def sample_by_instance_id(instance_ids: np.ndarray, npoints: int = 4096) -> np.ndarray:
    """Sample a specific number of points while ensuring, that all unique values in instance_ids are
    still represented by equal ratio in the resulting sampling. If the total number of instance_ids is 
    smaller than npoints, no sampling is performed.

    Args:
        instance_ids: The instance ids to sample from
        npoints: The number of points to sample. Defaults to 4096.

    Returns:
        The sampled indices.
    """
    # instance_labels is the 1-dim instance_idx above.
    if instance_ids.shape[0] > npoints:
        sampling_ratio = npoints / instance_ids.shape[0]

        all_idxs = []                                       # scene-level instance_idx of points being selected this time
        for iid in np.unique(instance_ids):              # sample points on object-level
            indices = (instance_ids == iid).nonzero()[0]     # locate these points of a specific instance_idx
            end = int(sampling_ratio * len(indices)) + 1        # num_of_points_to_be_sampled + 1
            np.random.shuffle(indices)                          # uniform sampling among each object instance
            selected_indices = indices[:end]                   # get the selected points
            all_idxs.extend(selected_indices)                   # append them to the scene-level list
        valid_idxs = np.array(all_idxs)
    else:
        valid_idxs = np.ones(instance_ids.shape, dtype=bool) # no sampling is required
    return valid_idxs


def filter_invalid_instances(instance_ids: np.ndarray, valid_instance_ids: set[int]):
    """Filter all invalid instances from the given instance ids. The array is checked 
    against invalid ids (id <= 0) and any ids not in the given valid_instance_ids set.

    Args:
        instance_ids: The instance ids to check for validity
        valid_instance_ids: The set of valid instance ids. Any ids not in this set will be discarded
    Returns:
        _description_
    """
    unique_instances_points = set(np.unique(instance_ids))
    diff = unique_instances_points.difference(valid_instance_ids)
    ninvalid = np.sum(instance_ids <= 0)
    if diff or ninvalid:
        valid_indices_mask = instance_ids > 0 # keep only the valid indices
        for d in diff:
            mask_current = instance_ids != d
            valid_indices_mask *= mask_current
    else:
        valid_indices_mask = np.ones(instance_ids.shape[0], dtype=np.bool_)
    return valid_indices_mask


def normalize_points(points):
    pc_ = points["points"]
    centroid = np.mean(pc_, axis=0)
    pc_ -= centroid
    m = np.max(np.sqrt(np.sum(pc_ ** 2, axis=1)))
    pc_ = pc_ / m
    
    return points


def farthest_point_sample(points: np.ndarray, nsample: int):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    point_data = points["points"]
    if point_data.shape[0] < nsample:
        return np.arange(point_data.shape[0])
    h = 5 if points.shape[0] <= 4096 else 7
    import fpsample
    indices = fpsample.bucket_fps_kdline_sampling(point_data, nsample, h=h)
    return indices