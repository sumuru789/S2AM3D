import trimesh

def load_mesh_util(input_fname):
    mesh = trimesh.load(input_fname, force='mesh', process=False)
    return mesh