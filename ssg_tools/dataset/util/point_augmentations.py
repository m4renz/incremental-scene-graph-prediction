import numpy as np
from ssg_tools.dataset.mesh import mesh_get_colors
import numpy.lib.recfunctions as rcf
import torch

def augment_points(points: np.ndarray, jitter=False, flip=False, rot=False):
    pts = points["points"]

    # construct a random rotation matrix
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.1
    if flip:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if rot:
        theta = np.random.rand() * 2 * np.pi
        m = np.matmul(m, [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])  # rotation

    centroid = pts.mean(0)

    pts -= centroid
    pts = np.dot(pts, m.T)
    points["points"][:] = pts

    # transform the normals as well
    if "normals" in points.dtype.fields:
        points["normals"][:] = np.dot(points["normals"], m.T)
    return points


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


def zero_mean(point: torch.Tensor, normalize: bool):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    if normalize:
        # find maximum distance for each n -> [n]
        furthest_distance = point.pow(2).sum(1).sqrt().max()
        point /= furthest_distance
    return point


def normalize_tensor(points: torch.Tensor):
    assert points.ndim == 2
    assert points.shape[1] == 3
    centroid = torch.mean(points, dim=0)  # N, 3
    points -= centroid  # n, 3, npts
    # find maximum distance for each n -> [n]
    furthest_distance = points.pow(2).sum(1).sqrt().max()
    points /= furthest_distance
    return points


def point_descriptor(points: np.ndarray):
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''

    pts = torch.from_numpy(points[:, :3])
    # centroid [n, 3]
    centroid_pts = pts.mean(0) 
    # # std [n, 3]
    std_pts = pts.std(0)
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    # volume [n, 1]
    segment_volume = (segment_dims[0]*segment_dims[1]*segment_dims[2]).unsqueeze(0)
    # length [n, 1]
    segment_lengths = segment_dims.max().unsqueeze(0)
    
    return torch.cat([centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths],dim=0)