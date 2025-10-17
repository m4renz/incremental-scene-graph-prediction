from __future__ import annotations
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    import pyvista as pv
import dataclasses
import numpy as np
from functools import cached_property

@dataclasses.dataclass(frozen=True)
class Camera:
    """A minimal camera implementation.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    size: tuple[int, int]
    transform: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        if self.transform is None:
            object.__setattr__(self, "transform", np.eye(4))
    @classmethod
    def from_camera_matrix(cls, camera_matrix: np.ndarray, size: tuple[int, int], transform: Optional[np.ndarray] = None) -> Camera:
        camera_matrix = np.asarray(camera_matrix).reshape(3,3)
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        obj = cls(fx, fy, cx, cy, size, transform=transform)
        #obj.camera_matrix = camera_matrix # skip recomputation
        return obj
    
    @cached_property
    def camera_matrix(self):
        mat = np.eye(3, dtype=np.float32)
        mat[0, 0] = self.fx
        mat[1, 1] = self.fy
        mat[0, 2] = self.cx
        mat[1, 2] = self.cy
        return mat
    
    def to_viewer(self, clipping_range: tuple[float, float] = (0.0001, 0.3)) -> pv.Camera:
        """Create a vtk camera from intrinsic pinhole parameters
        """
        import pyvista as pv
        vcamera = pv.Camera()
        vcamera.position = (0.0, 0.0, 0.0)
        vcamera.focal_point = (0, 0, 1)
        vcamera.up = (0, -1, 0)
        vcamera.clipping_range = clipping_range

        ny, nx = self.size
        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2 * (self.cx - nx / 2) / nx
        wcy =  2 * (self.cy - ny / 2) / ny
        vcamera.SetWindowCenter(wcx, wcy)

        view_angle = np.degrees(2.0 * np.arctan2( ny/2.0, self.fy))
        vcamera.view_angle = view_angle
        vcamera.aspect = nx / ny
        vcamera.model_transform_matrix = self.transform
        return vcamera