from ssg_tools.dataset.dataset_interface import ScanInterface, Camera
import dataclasses
import numpy as np
import pyvista as pv
from skimage.transform import rotate
from skimage.util import img_as_ubyte
from skimage.io import imsave
from pathlib import Path
from scipy.ndimage import find_objects


@dataclasses.dataclass
class RendererdScan:
    pose_index: int
    color: np.ndarray
    depth: np.ndarray
    labels: np.ndarray
    instances: np.ndarray
    bboxes: np.ndarray

    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        base_filename = f"frame-{self.pose_index:06d}.rendered"
        color_filename = output_dir / (base_filename + ".color.jpg")
        imsave(color_filename, self.color)

        depth_filename = output_dir / (base_filename + ".depth.png")
        imsave(depth_filename, self.depth)

        labels_filename = output_dir / (base_filename + ".labels.png")
        imsave(labels_filename, self.labels)

        instances_filename = output_dir / (base_filename + ".instances.png")
        imsave(instances_filename, self.instances)

        bboxes_filename = output_dir / (base_filename + ".bb.txt")
        np.savetxt(bboxes_filename, self.bboxes, fmt="%d")



class ScanRenderer:
    def __init__(self, scan: ScanInterface, debug: bool = False):
        self.scan = scan
        self.camera = scan.camera()
        self.plotter = None
        self.color_to_instance = None
        self.debug = debug
        self.fig = None

    def _init(self):
        if self.plotter is None:
            obj_mesh = self.scan.color_mesh()
            label_mesh = self.scan.label_mesh()

            self.plotter = pv.Plotter(off_screen=True, notebook=False, window_size=(self.camera.size[1], self.camera.size[0]))
            self.plotter.background_color = "black"
            self.plotter.camera = self.camera.to_viewer(clipping_range=(0.001, 10.0))
            #print("SIZE", self.camera.size, self.camera.cx, self.camera.cy)
            self.obj_mesh_actor = self.plotter.add_mesh(obj_mesh, texture=np.asarray(obj_mesh.visual.material.image), lighting=False)
            pv_label_mesh = pv.wrap(label_mesh)
            pv_label_mesh.point_data.set_scalars(label_mesh.vertex_attributes["colors"], name="colors")
            self.label_mesh_actor = self.plotter.add_mesh(pv_label_mesh, rgb=True, lighting=False)
            self.plotter.show(auto_close=False)

            scene_graph = self.scan.scene_graph(raw=True)
            nodes = scene_graph["nodes"]
            # map the unique colors to instances
            self.color_to_instance = {pv.Color(obj['instance_color']).int_rgb: id for id, obj in nodes.items()} 

    def _draw_color_depth(self):
        # make the textured obj visible
        self.obj_mesh_actor.visibility = True
        self.label_mesh_actor.visibility = False
        self.plotter.render()
        
        # extract the color and depth images
        color_image = img_as_ubyte(self.plotter.image)
        depth_image = (np.abs(self.plotter.image_depth) * 1000).astype(np.uint16)

        return color_image, depth_image

    def _draw_labels_instances(self):
        # make the labels mesh visible
        self.obj_mesh_actor.visibility = False
        self.label_mesh_actor.visibility = True
        self.plotter.render()

        label_image = self.plotter.image#, -90, resize=True)
        label_image = img_as_ubyte(label_image)

        unique_colors, unique_colors_map = np.unique(label_image.reshape(-1, 3), return_inverse=True, axis=0)

        mapped_instances = np.array([self.color_to_instance.get(tuple(c.tolist()), 0) for c in unique_colors], dtype=np.uint16)

        instances_image = mapped_instances[unique_colors_map].reshape(label_image.shape[:2])
        
        #unique_colors_map = unique_colors_map.reshape(label_image.shape[:2])
        bboxes = []
        nonzero_instances = np.nonzero(mapped_instances)[0]
        object_location = find_objects(instances_image)
        for idx in nonzero_instances:
            id = mapped_instances[idx]
            if id == 0:
                continue # skip 0 as invalid
            bbox = object_location[id-1] # ignores 0
            bboxes.append(np.array([id, bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop], dtype=np.int32))

        if bboxes:
            bboxes = np.sort(np.stack(bboxes), axis=0)
        # reverse color as in the original conversion it is interpreted as bgr
        label_image = label_image[..., ::-1]

        return label_image, instances_image, bboxes
    
    def render(self, pose_index: int):
        self._init()
        transform = self.scan.pose(pose_index)
        mat = pv.utilities.arrays.vtkmatrix_from_array(transform)
        mat.Invert()
        transform = pv.utilities.arrays.array_from_vtkmatrix(mat)
        self.plotter.camera.model_transform_matrix = transform

        color, depth = self._draw_color_depth()
        labels, instances, bboxes = self._draw_labels_instances()

        if self.debug:
            import matplotlib.pyplot as plt

            if self.fig is None:
                plt.ion()
                self.fig, self.axs = plt.subplots(1, 4)
                self.plots = [self.axs[i].imshow(img) for i,img in enumerate((color, depth, labels, instances))]
            else:
                for i,img in enumerate((color, depth, labels, instances)):
                    self.plots[i].set_data(img)
                plt.draw()
                plt.pause(1e-6)
        return RendererdScan(pose_index, color, depth, labels, instances, bboxes)
