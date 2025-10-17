import pyvista as pv
from pathlib import Path
import numpy as np
import networkx as netx
import argparse
import dataclasses
from imgui_bundle import imgui
from pyvista_imgui import ImguiPlotter
from imgui_bundle import immapp, hello_imgui  
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
from ssg_tools.dataset.mesh import mesh_get_colors

def imgui_all_actors(actor: pv.Actor, id=None):
    imgui.push_id(str(id))
    _, visibility = imgui.checkbox("Visible", actor.GetVisibility())
    actor.SetVisibility(visibility)
    imgui.pop_id()

def imgui_default_actor(actor: pv.Actor, id=None):
    imgui_all_actors(actor, id=id)
    imgui.push_id(str(id))
    _, opacity = imgui.slider_float("Opacity", actor.GetProperty().GetOpacity(), 0.0, 1.0, format="%.1f")
    actor.GetProperty().SetOpacity(opacity)
    imgui.pop_id()

def imgui_labels_actor(actor: pv.Actor, id=None):
    imgui_all_actors(actor, id=id)
    imgui.push_id(str(id))
    _, background_opacity = imgui.slider_float("Opacity", actor.GetMapper().GetBackgroundOpacity(), 0.0, 1.0, format="%.1f")
    actor.GetMapper().SetBackgroundOpacity(background_opacity)
    _, background_color = imgui.color_edit3("Background Color", actor.GetMapper().GetBackgroundColor())
    actor.GetMapper().SetBackgroundColor(background_color)
    imgui.pop_id()

def imgui_relationships_actors(labels_actor, lines_actor, id=None):
    imgui.push_id(str(id))

    _, visibility = imgui.checkbox("Labels Visible", labels_actor.GetVisibility())
    labels_actor.SetVisibility(visibility)
    _, visibility = imgui.checkbox("Lines Visible", lines_actor.GetVisibility())
    lines_actor.SetVisibility(visibility)

    _, background_opacity = imgui.slider_float("Label Background Opacity", labels_actor.GetMapper().GetBackgroundOpacity(), 0.0, 1.0, format="%.1f")
    labels_actor.GetMapper().SetBackgroundOpacity(background_opacity)

    _, background_color = imgui.color_edit3("Label Background Color", labels_actor.GetMapper().GetBackgroundColor())
    labels_actor.GetMapper().SetBackgroundColor(background_color)

    _, line_opacity = imgui.slider_float("Line Opacity", lines_actor.GetProperty().GetOpacity(), 0.0, 1.0, format="%.1f")
    lines_actor.GetProperty().SetOpacity(line_opacity)

    _, line_color = imgui.color_edit3("Line Color", lines_actor.GetProperty().GetColor())
    lines_actor.GetProperty().SetColor(line_color)
    imgui.pop_id()

@dataclasses.dataclass
class ViewPlane:
    plane: pv.PolyData
    texture: pv.Texture
    opacity: float = 0.75


class SceneVisualizer:
    def __init__(self, scene: ScanInterface):
        self.scene = scene

        # load the dataset and build the vtk mesh from it
        label_mesh = self.scene.label_mesh()
        self.labels_mesh = pv.wrap(label_mesh)
        colors = mesh_get_colors(label_mesh)
        self.labels_mesh.point_data.active_normals = label_mesh.vertex_normals
        self.labels_mesh.point_data.set_scalars(colors, "colors")

        # load and parse the scene graph information
        self.scene_graph = self.scene.scene_graph()
        self.scan_id = self.scene.scan_id

        scene_objects = self.scene_graph["nodes"]
        scene_relationships = self.scene_graph["edges"]
        
        labels = {id: f"{o['rio27_name']}:{id}" for id, o in scene_objects.items()}
        centers = {id: np.asarray(obb["center"], dtype=np.float32) for id, obb in self.scene.instance_obbs().items()}
        id_to_center = {id: (centers[id], label) for id, label in labels.items()}

        # build the graph from the relationships
        self.relationship_graph = netx.DiGraph()
        self.relationship_types = set()

        for id_from, id_to, _, label in scene_relationships:
            line = np.array([id_to_center[id_from][0], id_to_center[id_to][0]])
            node_from = id_to_center[id_from][1]
            node_to = id_to_center[id_to][1]
            self.relationship_types.add(label)
            if not self.relationship_graph.has_edge(node_from, node_to):
                self.relationship_graph.add_edge(node_from, node_to, label={label}, line=line, center=(line[0] + line[1]) / 2)
            else:
                self.relationship_graph.edges[node_from, node_to]["label"].add(label)
        self.active_relationship_types = self.relationship_types.copy()

        ny, nx = self.scene.image_size('color')
        aspect = nx / ny
        print("NX NY", nx, ny, aspect)
        camera = self.scene.camera("color").to_viewer()

        poses = []
        frustums = pv.MultiBlock()
        view_planes = []

        t_coords = np.array([[0,0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        # load the pose information
        for idx in range(self.scene.nimages):
            pose = self.scene.pose(idx)
            poses.append(pose[:3, 3])

            # pose is given as camera -> world so invert it for the camera transformation
            mat = pv.utilities.arrays.vtkmatrix_from_array(pose)
            mat.Invert()
            pose = pv.utilities.arrays.array_from_vtkmatrix(mat)
            camera.model_transform_matrix = pose
            frustum = camera.view_frustum(aspect)
            frustums.append(frustum)

            plane = pv.PolyData.from_regular_faces(points=frustum.points[:4], 
                                                        faces=np.array([[0, 1, 2, 3]]))
            plane.active_texture_coordinates = t_coords

            texture = pv.read_texture(self.scene.image_filename(idx, 'color'))
            texture.wrap = pv.Texture.WrapType.CLAMP_TO_BORDER
            view_planes.append(ViewPlane(plane, texture))
        poses = np.asarray(poses)
        title = f"Scene: {self.scan_id}"

        self.viewer = ImguiPlotter(title=title, imgui_backend='imgui_bundle')
        self.viewer.enable_depth_peeling() # allow transparency
        self.viewer.set_background("white")
        self.viewer.enable_pivot_style()

        # add actors to the viewer to show the data
        self.axes_actor = self.viewer.add_axes() # axes 
        self.labels_mesh_actor = self.viewer.add_mesh(self.labels_mesh, culling='back', rgb=True) # the mesh with the instance labels

        centers_arr = np.asarray([centers[id] for id in labels.keys()], dtype=np.float32)
        labels_list = list(labels.values())

        self.node_labels_actor = self.viewer.add_point_labels(centers_arr, labels_list, font_size=15, show_points=False, shape_color="red", text_color="black", always_visible=True, fill_shape=True) # labels of the instances

        self._add_relationship_labels()

        poses_spline = pv.Spline(poses, n_points=poses.shape[0] * 10)
        poses_points = pv.PolyData(poses)
        self.poses_spline_actor = self.viewer.add_mesh(poses_spline, line_width=2, show_scalar_bar=False, color='green')
        self.poses_points_actor = self.viewer.add_mesh(poses_points, color='green', render_points_as_spheres=True, point_size=10)
        self.frustums_actor, mapper = self.viewer.add_composite(frustums, color='red', style="wireframe")

        self.view_planes_actors = [self.viewer.add_mesh(vp.plane, texture=vp.texture, opacity=vp.opacity, smooth_shading=False) for vp in view_planes]
        
        # set all frustums and view planes to invisible initially
        for i, vpa in enumerate(self.view_planes_actors):
            self.frustums_actor.mapper.block_attr[i+1].visible = False
            vpa.visibility = False
        self.active_frustum = 0
        self.frustums_visible = False
        self.frustum_visibility = 0.75
    
    def _add_relationship_labels(self, shown_labels=None):
        graph = self.relationship_graph

        if shown_labels:
            def filter_func(n1, n2):
                return any(l in shown_labels for l in self.relationship_graph[n1][n2].get("label"))
            graph = netx.subgraph_view(self.relationship_graph, filter_edge=filter_func)
        else:
            shown_labels = self.relationship_types
        relationship_centers = np.array([center for _, _, center in graph.edges(data="center")])
        relationship_lines = np.array([line for _, _, line in graph.edges(data="line")]).reshape(-1, 3)
        relationship_labels = np.array(["\n".join((f"{n1}:{l}:{n2}" for l in lbl if l in shown_labels)) for n1, n2, lbl in graph.edges(data="label")])
        # add the filtered lines
        self.viewer.remove_actor(("relationship_labels", "relationship_lines"))
        self.relationship_labels_actor = self.viewer.add_point_labels(relationship_centers, relationship_labels, name="relationship_labels", font_size=15, show_points=False, shape_color="gray", text_color="black", fill_shape=True, always_visible=True) # relationships betwen the instances
        self.relationship_lines_actor = self.viewer.add_lines(relationship_lines, name="relationship_lines", width=1, color='white')

    def _gui(self):
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)
        vec = imgui.get_main_viewport().pos
        imgui.set_next_window_pos(vec, imgui.Cond_.once)
        imgui.set_next_window_size(imgui.get_main_viewport().size)
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin("Vtk View", flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move)
        self.viewer.render_imgui()
        imgui.end()

        imgui.begin("Options")

        if imgui.collapsing_header("Mesh"):
            imgui_default_actor(self.labels_mesh_actor, id="mesh")
        if imgui.collapsing_header("Node Labels"):
            imgui_labels_actor(self.node_labels_actor, id="labels")
        if imgui.collapsing_header("Relationships"):
            imgui_relationships_actors(self.relationship_labels_actor, self.relationship_lines_actor, id="relationships")
            if imgui.collapsing_header("Relationships Filter"):
                imgui.text("Click on the items to enable/disable this relationship.\nHold ctrl to select multiple items.")
                changed = False
                for l in self.relationship_types:
                    clicked, _ = imgui.selectable(l, l in self.active_relationship_types)
                    if clicked:
                        changed = True
                        if not imgui.get_io().key_ctrl: # clear selection unless the ctrl key is held
                            self.active_relationship_types.clear()
                        self.active_relationship_types.add(l) if l not in self.active_relationship_types \
                        else self.active_relationship_types.remove(l)
                if changed:
                    self._add_relationship_labels(self.active_relationship_types)
        if imgui.collapsing_header("Poses"):
            imgui.push_id("Poses")
            _, visibility = imgui.checkbox("Trajectory", self.poses_points_actor.GetVisibility())
            self.poses_points_actor.SetVisibility(visibility)
            self.poses_spline_actor.SetVisibility(visibility)
            _, active_frustum = imgui.slider_int("Pose", self.active_frustum, 0, len(self.view_planes_actors)-1)
            imgui.same_line()
            _, frustums_visible = imgui.checkbox("Visible", self.frustums_visible)
            _, self.frustum_visibility = imgui.slider_float("Opacity", self.frustum_visibility, 0.0, 1.0, format="%.1f")

            if not frustums_visible:
                #if self.active_frustum >= 0:
                for i, vpa in enumerate(self.view_planes_actors):
                    self.frustums_actor.mapper.block_attr[i+1].visible = False
                    vpa.visibility = False
            else:
                self.frustums_actor.mapper.block_attr[self.active_frustum+1].visible = False
                self.view_planes_actors[self.active_frustum].visibility = False
                self.frustums_actor.mapper.block_attr[active_frustum+1].visible = True
                self.view_planes_actors[active_frustum].visibility = True
                self.view_planes_actors[active_frustum].prop.opacity = self.frustum_visibility

            self.active_frustum = active_frustum
            self.frustums_visible = frustums_visible

            imgui.pop_id()
        imgui.end()

    def show(self):
        if self.viewer._closed:
         raise ValueError("Attempting to re-open a closed viewer")
        
        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = self.viewer.title
        runner_params.app_window_params.window_geometry.size = (1920, 1080)
        runner_params.imgui_window_params.show_status_bar = True

        runner_params.callbacks.show_gui = self._gui
        runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
        immapp.run(runner_params=runner_params)

            
def main():
    script_path = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description='Visualize a scene from the dataset')
    parser.add_argument('--dataset_path', required=True, help="The path to the dataset.")
    parser.add_argument('--id', required=True, help='specific scan id to render. By default all scans will be rendered')
    args = parser.parse_args()
    dataset = DatasetInterface3DSSG(args.dataset_path)
    print("dataset", dataset)
    try:
        scene = dataset.scan(int(args.id))
    except:
        scene = dataset.scan(args.id)
    print("scene", scene)
    visualizer = SceneVisualizer(scene)
    visualizer.show()

if __name__ == "__main__":
    main()