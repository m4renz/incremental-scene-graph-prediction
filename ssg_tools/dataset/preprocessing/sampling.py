import numpy as np
import trimesh
from typing import Optional
from math import ceil

__all__ = ['sample_mesh_points']

def _sample_mesh_points(mesh: trimesh.Trimesh, 
                       npoints: Optional[int] = None, 
                       density: Optional[float] = None, 
                       sample_color: bool = False,
                       seed: Optional[int] = None) -> np.ndarray:
    mesh_areas = trimesh.triangles.area(mesh.triangles, sum=False)
    mesh_area = np.sum(mesh_areas)

    if npoints:
        density = npoints / mesh_area
    elif density:
        npoints = ceil(mesh_area * density)

    # seed the random number generator as requested
    if seed is None:
        random = np.random.random
    else:
        random = np.random.default_rng(seed).random

    points_to_sample = mesh_areas * density

    points_to_sample += 1.0 # add one to sample each triangle at least once

    # handle the fractionals by probabilistic sampling of another point
    points_to_sample_fractionals, points_to_sample = np.modf(points_to_sample)
    samples = np.random.random_sample(points_to_sample_fractionals.shape[0])
    points_to_sample[samples < points_to_sample_fractionals] += 1

    points_to_sample = points_to_sample.astype(np.int32)

    face_indices = np.repeat(np.arange(points_to_sample.shape[0]), points_to_sample)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_indices]
    tri_vectors = tri_vectors[face_indices]

    if sample_color and hasattr(mesh.visual, "uv"):
        uv_origins = mesh.visual.uv[mesh.faces[:, 0]]
        uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()
        uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
        uv_vectors -= uv_origins_tile
        uv_origins = uv_origins[face_indices]
        uv_vectors = uv_vectors[face_indices]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if sample_color:
        if hasattr(mesh.visual, "uv"):
            sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)
            uv_samples = sample_uv_vector + uv_origins
            texture = mesh.visual.material.image
            colors = trimesh.visual.uv_to_interpolated_color(uv_samples, texture)
        else:
            colors = mesh.visual.face_colors[face_indices]

        return samples, face_indices, colors

    
    return samples, face_indices

def _interpolate_normals(mesh: trimesh.Trimesh, sample_points: np.ndarray, sample_face_indices: np.ndarray) -> np.ndarray:
    # compute the barycentric coordinates of each sample point
    bary = trimesh.triangles.points_to_barycentric(
        triangles=mesh.triangles[sample_face_indices], points=sample_points)
    # invalid values are treated as centered
    is_nan = np.any(~np.isfinite(bary), axis=1)
    bary[is_nan] = 1 / 3
    # interpolate vertex normals from barycentric coordinates
    sample_normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[sample_face_indices]] *
                                     trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
    
    return sample_normals, bary
    

def sample_mesh_points(mesh: trimesh.Trimesh, 
                       labels: Optional[np.ndarray] = None,
                       npoints: Optional[int] = None, 
                       density: Optional[float] = None, 
                       sample_color: bool = False,
                       seed: Optional[int] = None) -> trimesh.Trimesh:
    if sample_color:
        samples, face_indices, colors = _sample_mesh_points(mesh, npoints=npoints, density=density, sample_color=sample_color, seed=seed)
    else:
        samples, face_indices = _sample_mesh_points(mesh, npoints=npoints, density=density, sample_color=sample_color, seed=seed)
        colors = None

    normals, bary = _interpolate_normals(mesh, samples, face_indices)

    output = trimesh.Trimesh(vertices=samples, vertex_normals=normals, vertex_colors=colors, process=False)

    if labels is not None:
        labels_faces = mesh.faces[face_indices]
        labels_per_face = labels[labels_faces]
        nearest_vertex = np.argmax(bary, axis=1)
        labels_per_face = np.take_along_axis(labels_per_face, nearest_vertex[:, np.newaxis], axis=1).squeeze()

        return output, labels_per_face

    return output
    
