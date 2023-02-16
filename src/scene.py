from io import TextIOWrapper
from collections import namedtuple
from typing import List
import tinyobjloader

Light = namedtuple("Light", ["vertices", "normals", "vertex_indices", "normal_indices", "emission"])
Model = namedtuple("Model",
                   ["vertices", "normals", "vertex_indices", "normal_indices", "sigma_a", "sigma_s", "g", "eta"])


def parse_light(file: TextIOWrapper) -> Light:
    filename = file.readline().strip().split(' ')[-1]
    reader = tinyobjloader.ObjReader()
    if not reader.ParseFromFile(filename):
        print(f"Warning: Error occurred when parsing file '{filename}'")
        exit(-1)
    attrib = reader.GetAttrib()
    vertices = [attrib.vertices[i:i + 3] for i in range(0, len(attrib.vertices), 3)]
    normals = [attrib.normals[i:i + 3] for i in range(0, len(attrib.normals), 3)]
    shapes = reader.GetShapes()
    vertex_indices, normal_indices = [], []
    for shape in shapes:
        print(shape.name)
        indices = shape.mesh.indices
        vertex_indices.extend([index.vertex_index for index in indices])
        normal_indices.extend([index.normal_index for index in indices])
    emission = tuple(float(channel) for channel in file.readline().strip().split(' ')[1:])
    return Light(vertices, normals, vertex_indices, normal_indices, emission)


def parse_model(file: TextIOWrapper) -> Model:
    filename = file.readline().strip()
    reader = tinyobjloader.ObjReader()
    if not reader.ParseFromFile(filename):
        print(f"Warning: Error occurred when parsing file '{filename}'")
        exit(-1)
    attrib = reader.GetAttrib()
    vertices = [attrib.vertices[i:i + 3] for i in range(0, len(attrib.vertices), 3)]
    normals = [attrib.normals[i:i + 3] for i in range(0, len(attrib.normals), 3)]
    shapes = reader.GetShapes()
    vertex_indices, normal_indices = [], []
    for shape in shapes:
        print(shape.name)
        indices = shape.mesh.indices
        vertex_indices.extend([index.vertex_index for index in indices])
        normal_indices.extend([index.normal_index for index in indices])
    properties = {}
    for _ in range(4):
        line = file.readline().strip()
        if line.startswith("Sigma_a"):
            properties["Sigma_a"] = tuple(float(channel) for channel in line.split(' ')[1:])
        elif line.startswith("Sigma_s"):
            properties["Sigma_s"] = tuple(float(channel) for channel in line.split(' ')[1:])
        elif line.startswith("g"):
            properties["g"] = float(line.split(' ')[-1])
        elif line.startswith("eta"):
            properties["eta"] = float(line.split(' ')[-1])
    return Model(vertices, normals, vertex_indices, normal_indices, properties["Sigma_a"], properties["Sigma_s"],
                 properties["g"], properties["eta"])


# Scene = namedtuple("Scene", ["model", "lights"])
Mesh = namedtuple("Mesh", ["vertex_indices", "normal_indices"])
Material = namedtuple("Material", ["sigma_a", "sigma_s", "g", "eta"])


class Scene(object):
    def __init__(self, model: Model, lights: List[Light]) -> None:
        self.material = Material(model.sigma_a, model.sigma_s, model.g, model.eta)
        self.emissions = [light.emission for light in lights]
        self.vertex_arr = model.vertices + [vertex for light in lights for vertex in light.vertices]
        self.normal_arr = model.normals + [normal for light in lights for normal in light.normals]
        self.meshes: List[Mesh] = []
        self.meshes.append(Mesh(model.vertex_indices, model.normal_indices))
        v_offset, n_offset = len(model.vertices), len(model.normals)
        for light in lights:
            mesh = Mesh([item + v_offset for item in light.vertex_indices],
                        [item + n_offset for item in light.normal_indices])
            self.meshes.append(mesh)
            v_offset += len(light.vertices)
            n_offset += len(light.normals)


def parse_scene(filename: str) -> Scene:
    model, lights = None, []
    with open(filename, 'r') as f:
        while line := f.readline():
            if line.startswith("Lights"):
                print(line.strip())
                lights.append(parse_light(f))
            elif line.startswith("Objects"):
                print(line.strip())
                model = parse_model(f)
    return Scene(model, lights)
