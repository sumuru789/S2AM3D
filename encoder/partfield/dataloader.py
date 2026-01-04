import torch
import boto3
import json
from os import path as osp
# from botocore.config import Config
# from botocore.exceptions import ClientError
import h5py
import io
import numpy as np
import skimage
import trimesh
import os
from scipy.spatial import KDTree
import gc
from plyfile import PlyData

## For remeshing
import mesh2sdf
import tetgen
import vtk
import math
import tempfile

### For mesh processing
import pymeshlab

from partfield.utils import *

#########################
## To handle quad inputs
#########################
def quad_to_triangle_mesh(F):
    """
    Converts a quad-dominant mesh into a pure triangle mesh by splitting quads into two triangles.

    Parameters:
        quad_mesh (trimesh.Trimesh): Input mesh with quad faces.

    Returns:
        trimesh.Trimesh: A new mesh with only triangle faces.
    """
    faces = F

    ### If already a triangle mesh -- skip
    if len(faces[0]) == 3:
        return F

    new_faces = []

    for face in faces:
        if len(face) == 4:  # Quad face
            # Split into two triangles
            new_faces.append([face[0], face[1], face[2]])  # Triangle 1
            new_faces.append([face[0], face[2], face[3]])  # Triangle 2
        else:
            print(f"Warning: Skipping non-triangle/non-quad face {face}")

    new_faces = np.array(new_faces)

    return new_faces
#########################

class Demo_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.data_path = cfg.dataset.data_path
        self.is_pc = cfg.is_pc

        all_files = os.listdir(self.data_path)
        selected = []
        self.meta_json_files = []
        for f in all_files:
            if ".ply" in f and self.is_pc:
                selected.append(f)
            elif f.endswith('.json') and self.is_pc:
                self.meta_json_files.append(f)
            elif (".obj" in f or ".glb" in f or ".off" in f) and not self.is_pc:
                selected.append(f)

        self.meta_data = []
        for meta_file in self.meta_json_files:
            meta_path = os.path.join(self.data_path, meta_file)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if isinstance(meta, list):
                    self.meta_data.extend(meta)
                else:
                    self.meta_data.append(meta)

        self.data_list = []
        for f in selected:
            self.data_list.append({'type': 'ply', 'file': f})
        for meta in self.meta_data:
            self.data_list.append({'type': 'meta', 'meta': meta})
        self.pc_num_pts = 100000
        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name
        print("val dataset len:", len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)

    def load_ply_to_numpy(self, filename):
        """
        Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

        Parameters:
            filename (str): Path to the PLY file.

        Returns:
            numpy.ndarray: Point cloud array of shape (N, 3).
        """
        ply_data = PlyData.read(filename)

        # Extract vertex data
        vertex_data = ply_data["vertex"]
        
        # Convert to NumPy array (x, y, z)
        points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T

        return points

    def get_model(self, item):
        if item['type'] == 'ply':
            ply_file = item['file']
            uid = ply_file.split(".")[-2].replace("/", "_")
            ply_file_read = os.path.join(self.data_path, ply_file)
            pc = self.load_ply_to_numpy(ply_file_read)
        elif item['type'] == 'meta':
            meta = item['meta']
            uid = meta.get('id', 'unknown')
            npy_path = meta.get('data_path', None)
            if npy_path is None or not os.path.exists(npy_path):
                raise FileNotFoundError(f"Missing npy file: {npy_path}")
            npy_data = np.load(npy_path, allow_pickle=True).item()
            pc = npy_data['coord']  # (N, 3)
        else:
            raise ValueError('Unknown data type')

        bbmin = pc.min(0)
        bbmax = pc.max(0)
        center = (bbmin + bbmax) * 0.5
        extent = (bbmax - bbmin).max()
        if extent < 1e-8:
            pc = pc - center
            noise = np.random.normal(scale=1e-4, size=pc.shape).astype(pc.dtype)
            pc = pc + noise
            extent = 1.0
        scale = 2.0 * 0.9 / extent
        pc = (pc - center) * scale
        pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            'uid': uid
        }
        result['pc'] = torch.tensor(pc, dtype=torch.float32)
        return result

    def __getitem__(self, index):
        gc.collect()
        return self.get_model(self.data_list[index])

##############

###############################
class Demo_Remesh_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.data_path = cfg.dataset.data_path

        all_files = os.listdir(self.data_path)

        selected = []
        for f in all_files:
            if (".obj" in f or ".glb" in f):
                selected.append(f)

        self.data_list = selected
        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)


    def get_model(self, ply_file):

        uid = ply_file.split(".")[-2]

        ####
        obj_path = os.path.join(self.data_path, ply_file)
        mesh =  load_mesh_util(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces

        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        extent = (bbmax - bbmin).max()
        if extent < 1e-8:
            vertices = vertices - center
            noise = np.random.normal(scale=1e-4, size=vertices.shape).astype(vertices.dtype)
            vertices = vertices + noise
            extent = 1.0
        scale = 2.0 * 0.9 / extent
        vertices = (vertices - center) * scale
        vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)
        mesh.vertices = vertices

        ### Pre-process mesh
        if self.preprocess_mesh:
            # Create a PyMeshLab mesh directly from vertices and faces
            ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

            # Create a MeshSet and add your mesh
            ms = pymeshlab.MeshSet()
            ms.add_mesh(ml_mesh, "from_trimesh")

            # Apply filters
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_duplicate_vertices')
            percentageMerge = pymeshlab.PercentageValue(0.5)
            ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
            ms.apply_filter('meshing_remove_unreferenced_vertices')


            # Save or extract mesh
            processed = ms.current_mesh()
            mesh.vertices = processed.vertex_matrix()
            mesh.faces = processed.face_matrix()               

            print("after preprocessing...")
            print(mesh.vertices.shape)
            print(mesh.faces.shape)

        ### Save input
        save_dir = f"exp_results/{self.result_name}"
        os.makedirs(save_dir, exist_ok=True)
        view_id = 0            
        mesh.export(f'{save_dir}/input_{uid}_{view_id}.ply')   

        try:
            ###### Remesh ######
            size= 256
            level = 2 / size

            sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
            # NOTE: the negative value is not reliable if the mesh is not watertight
            udf = np.abs(sdf)
            vertices, faces, _, _ = skimage.measure.marching_cubes(udf, level)

            #### Only use SDF mesh ###
            # new_mesh = trimesh.Trimesh(vertices, faces)
            ##########################

            #### Make tet #####
            components = trimesh.Trimesh(vertices, faces).split(only_watertight=False)
            new_mesh = [] #trimesh.Trimesh()
            if len(components) > 100000:
                raise NotImplementedError
            for i, c in enumerate(components):
                c.fix_normals()
                new_mesh.append(c) #trimesh.util.concatenate(new_mesh, c)
            new_mesh = trimesh.util.concatenate(new_mesh)

            # generate tet mesh
            tet = tetgen.TetGen(new_mesh.vertices, new_mesh.faces)
            tet.tetrahedralize(plc=True, nobisect=1., quality=True, fixedvolume=True, maxvolume=math.sqrt(2) / 12 * (2 / size) ** 3)
            tmp_vtk = tempfile.NamedTemporaryFile(suffix='.vtk', delete=True)
            tet.grid.save(tmp_vtk.name)

            # extract surface mesh from tet mesh
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(tmp_vtk.name)
            reader.Update()
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(reader.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()
            writer = vtk.vtkOBJWriter()
            tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=True)
            writer.SetFileName(tmp_obj.name)
            writer.SetInputData(polydata)
            writer.Update()
            new_mesh =  load_mesh_util(tmp_obj.name)
            ##########################

            new_mesh.vertices = new_mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]

            mesh = new_mesh
        ####################

        except:
            print("Error in tet.")
            mesh = mesh 

        pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 

        result = {
                    'uid': uid
                }

        result['pc'] = torch.tensor(pc, dtype=torch.float32)
        result['vertices'] = mesh.vertices
        result['faces'] = mesh.faces

        return result

    def __getitem__(self, index):
        
        gc.collect()

        return self.get_model(self.data_list[index])


class Correspondence_Demo_Dataset(Demo_Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.data_path = cfg.dataset.data_path
        self.is_pc = cfg.is_pc

        self.data_list = cfg.dataset.all_files

        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))
    

class PartNetDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, num_points=100000, is_train=True, source_name=None):
        """PartNet dataset loader."""
        super().__init__()
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        total_samples = len(self.metadata)
        if is_train:
            train_size = int(total_samples)
            self.metadata = self.metadata[:train_size]
        else:
            self.metadata = self.metadata
            
        self.num_points = num_points
        if source_name is not None:
            self.source = source_name
        else:
            lower_path = metadata_path.lower()
            if 'partnet' in lower_path:
                self.source = 'partnet'
            elif 'objaverse' in lower_path:
                self.source = 'objaverse'
            else:
                self.source = 'unknown'
        split = 'train' if is_train else 'test'
        print(f"{split} dataset size: {len(self.metadata)}")
        
    def __len__(self):
        return len(self.metadata)
    
    def normalize_point_cloud(self, points):
        """Normalize points to [-1, 1] with stability safeguards."""
        bbmin = points.min(0)
        bbmax = points.max(0)
        center = (bbmin + bbmax) * 0.5
        extent = (bbmax - bbmin).max()
        if extent < 1e-8:
            points = points - center
            noise = np.random.normal(scale=1e-4, size=points.shape).astype(points.dtype)
            points = points + noise
            extent = 1.0
        scale = 2.0 * 0.9 / extent
        points = (points - center) * scale
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        return points
    
    def __getitem__(self, index):
        """Return a dict with pc/uid/label info."""
        sample = self.metadata[index]
        model_id = sample['id']
        data_path = sample['data_path']
        
        data = np.load(data_path, allow_pickle=True).item()
        
        points = data['coord']  # (N, 3)
        label = data['label']
        
        color = data.get('color', None)
        if color is None:
            color = points.copy()
        else:
            if isinstance(color, torch.Tensor):
                color = torch.clamp(color.clone().detach().float(), -1.0, 1.0)
            else:
                color = torch.clamp(torch.tensor(color, dtype=torch.float32), -1.0, 1.0)
            
        
        
        points = self.normalize_point_cloud(points)
        
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            color = color[indices]
            label = label[indices]
        
        result = {
            'pc': torch.tensor(points, dtype=torch.float32),  # (N, 3)
            'color': color if isinstance(color, torch.Tensor) else torch.tensor(color, dtype=torch.float32),  # (N, 3)
            'uid': model_id,
            'label': torch.tensor(label, dtype=torch.int32),
            'source': self.source
        }
        
        return result    