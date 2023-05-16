from __future__ import annotations

import json
import math
import sys

import luisa
import numba
import numpy as np
import tinyobjloader
from luisa import RandomSampler
from luisa.accel import offset_ray_origin
from luisa.mathtypes import *
from matplotlib import pyplot as plt

DeviceDirectionLight = luisa.StructType(direction=float3, emission=float3)
DevicePointLight = luisa.StructType(position=float3, emission=float3)
DeviceSurfaceLight = luisa.StructType(emission=float3)
Onb = luisa.StructType(tangent=float3, binormal=float3, normal=float3)


@luisa.func
def to_world(self, v):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal


@luisa.func
def rotate(self, phi):
    tangent = self.tangent * cos(phi) - self.binormal * sin(phi)
    binormal = self.tangent * sin(phi) + self.binormal * cos(phi)
    self.tangent = tangent
    self.binormal = binormal


Onb.add_method(to_world, "to_world")
Onb.add_method(rotate, "rotate")


@luisa.func
def make_onb(normal):
    binormal = normalize(select(
        make_float3(0.0, -normal.z, normal.y),
        make_float3(-normal.y, normal.x, 0.0),
        abs(normal.x) > abs(normal.z)))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    result.tangent = tangent
    result.binormal = binormal
    result.normal = normal
    return result


@luisa.func
def cosine_sample_hemisphere(u):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))


@luisa.func
def uniform_sample_hemisphere(u):
    r = sqrt(1.0 - u.x * u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), u.x)


@luisa.func
def uniform_sample_sphere(u):
    cos_theta = 1.0 - 2.0 * u.x
    sin_theta = sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)


luisa.init()
model_vertex_count: int = 0
vertex_buffer: luisa.Buffer = None
normal_buffer: luisa.Buffer = None
surface_light_buffer: luisa.Buffer = None
point_light_buffer: luisa.Buffer = None
direction_light_buffer: luisa.Buffer = None
surface_light_count: int = 0
point_light_count: int = 0
direction_light_count: int = 0
heap = luisa.BindlessArray()
accel: luisa.Accel = luisa.Accel()
sigma_a: float3 = make_float3(1.0)
sigma_s: float3 = make_float3(1.0)
eta: float = 1.0
g: float = 0.0
spp: int = 2400
res = make_int2(1000, 1000)


@numba.jit
def normal_array_kernel(triangle_array, normal_index_array, normal_vectors, vertex_num):
    normal_array = np.empty((vertex_num, 3), dtype=np.float32)
    for normal_index, vertex_index in zip(normal_index_array, triangle_array):
        normal_array[vertex_index] = normal_vectors[normal_index]
    return normal_array


@numba.jit
def ccw_check_kernel(triangle_array, vertex_array, normal_array):
    for i in range(0, len(triangle_array), 3):
        i0, i1, i2 = triangle_array[i:i + 3]
        p0, p1, p2 = vertex_array[i0], vertex_array[i1], vertex_array[i2]
        n = np.cross(p1 - p0, p2 - p0)
        if np.dot(n, normal_array[i0]) < 0:
            triangle_array[i], triangle_array[i + 1] = i1, i0


def parse_scene(filename: str):
    with open(filename, 'r') as file:
        scene_data = json.load(file)
        vertex_arrays = []
        normal_arrays = []
        triangle_arrays = []
        reader = tinyobjloader.ObjReader()
        # parse the model to be rendered, add model data to array lists
        offset = 0
        if not reader.ParseFromFile(scene_data["render_model"]):
            print("Failed to parse model")
            exit(0)
        else:
            print("Received model data...")
        attrib = reader.GetAttrib()
        global model_vertex_count
        vertex_array = np.array(attrib.vertices, dtype=np.float32).reshape(-1, 3)
        model_vertex_count = vertex_array.shape[0]
        normal_vectors = np.array(attrib.normals, dtype=np.float32).reshape(-1, 3)
        normal_index_array = np.array([
            index.normal_index for shape in reader.GetShapes() for index in shape.mesh.indices
        ], dtype=np.int32)
        triangle_array = np.array([
            index.vertex_index for shape in reader.GetShapes() for index in shape.mesh.indices
        ], dtype=np.int32)
        normal_array = normal_array_kernel(triangle_array, normal_index_array, normal_vectors, vertex_array.shape[0])
        normal_array /= np.linalg.norm(normal_array, axis=1).reshape((-1, 1))
        print("Model data parsing done...")
        # ccw_check_kernel(triangle_array, vertex_array, normal_array)
        vertex_arrays.append(vertex_array)
        normal_arrays.append(normal_array)
        triangle_arrays.append(triangle_array)
        # parse and upload light information
        # add surface light data to array lists
        offset += vertex_arrays[-1].shape[0]
        global surface_light_buffer, direction_light_buffer, point_light_buffer
        global surface_light_count, direction_light_count, point_light_count
        surface_lights = []
        for light in scene_data["surface_lights"]:
            print("Surface light with emission: ", light["emission"])
            reader.ParseFromFile(light["model"])
            attrib = reader.GetAttrib()
            vertex_array = np.array(attrib.vertices, dtype=np.float32).reshape(-1, 3)
            normal_vectors = np.array(attrib.normals, dtype=np.float32).reshape(-1, 3)
            normal_vectors /= np.linalg.norm(normal_vectors, axis=1).reshape(-1, 1)
            normal_array = np.empty_like(vertex_array, dtype=np.float32)
            for shape in reader.GetShapes():
                for index in shape.mesh.indices:
                    normal_array[index.vertex_index] = normal_vectors[index.normal_index]
            surface_lights.append(DeviceSurfaceLight(emission=make_float3(*light["emission"])))
            vertex_arrays.append(vertex_array)
            normal_arrays.append(normal_array)
            triangle_array = np.array([
                index.vertex_index for shape in reader.GetShapes() for index in shape.mesh.indices
            ]) + offset
            print(triangle_array)
            triangle_arrays.append(triangle_array)
            offset += vertex_arrays[-1].shape[0]
        surface_light_buffer = luisa.Buffer.empty(max(len(surface_lights), 1), dtype=DeviceSurfaceLight)
        if surface_lights:
            surface_light_buffer.copy_from_list(surface_lights)
            surface_light_count = len(surface_lights)
        assert len(
            surface_lights) <= 1, "Restrict surface light number to no more than 1 to support light importance sampling"
        direction_lights = \
            [DeviceDirectionLight(
                direction=make_float3(*light["direction"]),
                emission=make_float3(*light["emission"])
            ) for light in scene_data["direction_lights"]]
        direction_light_buffer = luisa.Buffer.empty(max(len(direction_lights), 1), dtype=DeviceDirectionLight)
        if direction_lights:
            direction_light_buffer.copy_from_list(direction_lights)
            direction_light_count = len(direction_lights)
        point_lights = \
            [DevicePointLight(
                position=make_float3(*light["position"]),
                emission=make_float3(*light["emission"])
            ) for light in scene_data["point_lights"]]
        point_light_buffer = luisa.Buffer.empty(max(len(point_lights), 1), dtype=DevicePointLight)
        if point_lights:
            point_light_buffer.copy_from_list(point_lights)
            point_light_count = len(point_lights)
        # combine and upload vertex && normal array lists
        global vertex_buffer, normal_buffer
        vertices = np.concatenate(vertex_arrays)
        normals = np.concatenate(normal_arrays)
        vertex_buffer = luisa.Buffer.empty(len(vertices), float3)
        normal_buffer = luisa.Buffer.empty(len(normals), float3)
        vertex_buffer.copy_from_array(np.hstack((vertices, np.zeros((vertices.shape[0], 1), dtype=np.float32))))
        normal_buffer.copy_from_array(np.hstack((normals, np.zeros((normals.shape[0], 1), dtype=np.float32))))
        # upload array lists to heap
        global heap, accel
        mesh_cnt = len(triangle_arrays)
        for i in range(mesh_cnt):
            triangle_array = triangle_arrays[i]
            triangle_buffer = luisa.Buffer(len(triangle_array), dtype=int)
            triangle_buffer.copy_from_array(triangle_array)
            heap.emplace(i, triangle_buffer)
            mesh = luisa.Mesh(vertex_buffer, triangle_buffer)
            accel.add(mesh)
        accel.update()
        heap.update()
        # parse parameters
        global sigma_a, sigma_s, eta, g
        sigma_a = make_float3(*scene_data["sigma_a"])
        sigma_s = make_float3(*scene_data["sigma_s"])
        eta = scene_data["eta"]
        g = scene_data["g"]
        global spp
        spp = scene_data["spp"]


dmfp: float3 = make_float3(0.0)
albedo: float3 = make_float3(0.0)
transmittance: float = 1.0


def calculate_parameters():
    sigma_s_prime = sigma_s * (1.0 - g)
    sigma_t_prime = sigma_s_prime + sigma_a
    alpha_prime = sigma_s_prime / sigma_t_prime
    fresnel = -1.440 / eta / eta + 0.710 / eta + 0.668 + 0.0636 * eta
    a = (1.0 + fresnel) / (1.0 - fresnel)
    global albedo
    albedo = 0.5 * alpha_prime * (1.0 + make_float3(
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.x))),
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.y))),
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.z)))
    )) / (1.0 + make_float3(
        math.sqrt(3.0 * (1.0 - alpha_prime.x)),
        math.sqrt(3.0 * (1.0 - alpha_prime.y)),
        math.sqrt(3.0 * (1.0 - alpha_prime.z)))
          )
    sigma_tr = make_float3(
        math.sqrt(3.0 * (1.0 - alpha_prime.x)),
        math.sqrt(3.0 * (1.0 - alpha_prime.y)),
        math.sqrt(3.0 * (1.0 - alpha_prime.z))
    ) * sigma_t_prime
    s = albedo - 0.33
    s *= s
    s = 3.5 + 100 * s * s
    global dmfp
    dmfp = 1.0 / (s * sigma_tr)
    reflectance = (1.0 - eta) / (1.0 + eta)
    global transmittance
    transmittance = 1.0 - reflectance * reflectance


@luisa.func
def generate_ray(frag_coord):
    fov = 20.0 * 0.5 * math.pi / 180
    origin = make_float3(0.0, 1.0, 8.0)
    look_at = make_float3(0.0, 0.5, 0.0)
    forward = normalize(look_at - origin)
    right = normalize(cross(forward, make_float3(0.0, 1.0, 0.0)))
    up = normalize(cross(right, forward))
    direction = normalize((frag_coord.x * right * res.x / res.y + frag_coord.y * up) * tan(fov) + forward)
    return make_ray(origin, direction, 1e-2, 1e10)


@luisa.func
def aces_tone_mapping(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


@luisa.func
def linear_to_srgb(x):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                        12.92 * x,
                        x <= 0.00031308),
                 0.0, 1.0)


@luisa.func
def normalized_bssrdf(r):
    a = exp(-r / (3. * dmfp))
    return (a + a * a * a) / (8. * math.pi * r * dmfp)


@luisa.func
def pdf_disk(r):
    a = exp(-r / (3. * dmfp))
    a = (a + a * a * a) / (8. * math.pi * r * dmfp)
    return (a.x + a.y + a.z) / 3.


@luisa.func
def cdf_disk(r):
    a = exp(-r / (3. * dmfp))
    return 1. - 0.25 * a * a * a - 0.75 * a


@luisa.func
def sample_disk(u):
    u = 3. * u
    if u > 2.:
        r = dmfp.x
        u -= 2.
    elif u > 1.:
        r = dmfp.y
        u -= 1.
    else:
        r = dmfp.z
    q = 4. * (u - 1.)
    x = pow(-0.5 * q + sqrt(0.25 * q * q + 1.), 1 / 3) - pow(0.5 * q + sqrt(0.25 * q * q + 1.), 1 / 3)
    return -3. * log(x) * r


@luisa.func
def balanced_heuristic(pdf_a, pdf_b):
    k = 1.
    t = pow(make_float2(pdf_a, pdf_b), k)
    return t.x / max(t.x + t.y, 1e-4)


@luisa.func
def fresnel_schlick(c):
    f0 = (eta - 1.) / (eta + 1.)
    f0 *= f0
    return 1. - lerp(f0, 1., pow(max(0., 1. - c), 5.))


@luisa.func
def collect_direct_illumination(p, n):
    acc = make_float3(0.)
    for i in range(point_light_count):
        point_light = point_light_buffer.read(i)
        light_direction = normalize(point_light.position - p)
        ray = make_ray(p, light_direction, 1e-2, length(point_light.position - p))
        if not accel.trace_any(ray):
            cos_wi = max(dot(n, light_direction), 0.)
            acc += cos_wi * point_light.emission * fresnel_schlick(cos_wi)
    for i in range(direction_light_count):
        direction_light = direction_light_buffer.read(i)
        ray = make_ray(p, direction_light.direction, 1e-2, 1e10)
        if not accel.trace_any(ray):
            cos_wi = max(dot(n, light_direction), 0.)
            acc += cos_wi * direction_light.emission * fresnel_schlick(cos_wi)
    return acc


@luisa.func
def path_tracing_kernel(canvas):
    acc = make_float3(0.0)
    for idx in range(spp):
        sampler = RandomSampler(make_int3(dispatch_id().xy, idx))
        contrib = make_float3(0.0)
        amp = make_float3(1.0)
        frag_coord = make_float2(dispatch_id().x + sampler.next(), dispatch_id().y + sampler.next()) / res * 2. - 1.
        ray = generate_ray(frag_coord)
        pdf_light = 0.0
        pdf_bsdf = 1e30
        for depth in range(8):
            hit = accel.trace_closest(ray)
            if hit.miss():
                break
            i0 = heap.buffer_read(int, hit.inst, hit.prim * 3)
            i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
            i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
            p0 = vertex_buffer.read(i0)
            p1 = vertex_buffer.read(i1)
            p2 = vertex_buffer.read(i2)
            p = hit.interpolate(p0, p1, p2)
            n0 = normal_buffer.read(i0)
            n1 = normal_buffer.read(i1)
            n2 = normal_buffer.read(i2)
            n = normalize(hit.interpolate(n0, n1, n2))
            cos_wo = abs(dot(n, ray.get_dir()))
            if cos_wo < 1e-4:
                break
            onb = make_onb(n)
            if hit.inst == 0:
                amp *= fresnel_schlick(cos_wo)
                ux, uy = sampler.next(), sampler.next()
                uy *= 2
                onb.rotate(sampler.next() * 2. * math.pi)
                if uy > 1.:
                    uy -= 1.
                    axis = n
                elif uy > 1.:
                    uy -= 1.
                    axis = onb.tangent
                else:
                    axis = onb.binormal
                if uy > 0.5:
                    axis = -axis
                    uy = 2. * uy - 1.
                else:
                    uy = 2. * uy
                prev_onb = onb
                uz = sampler.next()
                r_max = sample_disk(uz)
                pdf_xy = -log(ux * uz)
                radius = sample_disk(ux * uz)
                theta = math.pi * uy * 2.
                height = sqrt(r_max * r_max - radius * radius)
                local_origin = make_float3(cos(theta) * radius, sin(theta) * radius,
                                           height)
                onb = make_onb(axis)
                ray = make_ray(p + onb.to_world(local_origin), -axis, 0.0, 2. * height)
                hit = accel.trace_closest(ray)
                if hit.miss() or hit.inst != 0:
                    break
                i0 = heap.buffer_read(int, hit.inst, hit.prim * 3)
                i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
                i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
                p0 = vertex_buffer.read(i0)
                p1 = vertex_buffer.read(i1)
                p2 = vertex_buffer.read(i2)
                offset = hit.interpolate(p0, p1, p2) - p
                p = p + offset
                n0 = normal_buffer.read(i0)
                n1 = normal_buffer.read(i1)
                n2 = normal_buffer.read(i2)
                n = normalize(hit.interpolate(n0, n1, n2))
                onb = make_onb(n)
                p = offset_ray_origin(p, n)
                bssrdf = normalized_bssrdf(length(offset))
                pdf_bssrdf = pdf_disk(length(cross(prev_onb.normal, offset))) * abs(dot(prev_onb.normal, n)) * 0.5 + (
                            pdf_disk(length(cross(prev_onb.tangent, offset))) * abs(
                        dot(prev_onb.tangent, n)) + pdf_disk(length(cross(prev_onb.binormal, offset))) * abs(
                        dot(prev_onb.binormal, n))) * 0.25
                amp *= albedo * bssrdf / (pdf_bssrdf * pdf_xy * math.pi)
                contrib += amp * collect_direct_illumination(p, n)
                ray = make_ray(p, onb.to_world(cosine_sample_hemisphere(make_float2(sampler.next(), sampler.next()))),
                               1e-3, 1e10)
                if surface_light_count != 0:
                    i0 = heap.buffer_read(int, 1, 0)
                    i1 = heap.buffer_read(int, 1, 1)
                    i2 = heap.buffer_read(int, 1, 2)
                    p0 = vertex_buffer.read(i0)
                    p1 = vertex_buffer.read(i1)
                    p2 = vertex_buffer.read(i2)
                    ux, uy = sampler.next(), sampler.next()
                    pp = ux * (p2 - p1) + uy * (p1 - p0) + p0
                    pn = normalize(cross(p1 - p0, p2 - p1))
                    light_direction = normalize(pp - p)
                    cos_light = dot(light_direction, pn)
                    if cos_light > 0:
                        pn = -pn
                    else:
                        cos_light = -cos_light
                    pp = offset_ray_origin(pp, pn)
                    cos_wi_light = abs(dot(light_direction, n))
                    if cos_wi_light > 1e-2:
                        p_ray = make_ray(p, light_direction, 1e-2, length(pp - p))
                        if not accel.trace_any(p_ray):
                            area = length(cross(p1 - p0, p2 - p0))
                            pdf_light = length_squared(pp - p) / (area * cos_light)
                            pdf_bsdf = cos_wi_light / math.pi
                            surface_light = surface_light_buffer.read(0)
                            beta = amp * fresnel_schlick(cos_wi_light)
                            mis_weight = balanced_heuristic(pdf_light, pdf_bsdf)
                            contrib += surface_light.emission * beta * (
                                    mis_weight * cos_wi_light * cos_light / pdf_light)
                cos_wi = dot(n, ray.get_dir())
                if cos_wi < 1e-2:
                    break
                pdf_bsdf = cos_wi / math.pi
                amp *= fresnel_schlick(cos_wi) * cos_wi / pdf_bsdf
            else:
                if depth == 0:
                    ray.set_origin(offset_ray_origin(p, ray.get_dir()))
                    continue
                else:
                    surface_light = surface_light_buffer.read(hit.inst - 1)
                    cos_light = abs(dot(n, ray.get_dir()))
                    area = length(cross(p1 - p0, p2 - p0))
                    pdf_light = length_squared(p - ray.get_origin()) / (area * cos_light)
                    mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
                    contrib += surface_light.emission * amp * (mis_weight * cos_light)
                    break
        if any(isnan(contrib)):
            contrib = make_float3(0.)
        acc += contrib
    acc *= 1. / spp
    color = linear_to_srgb(acc)
    canvas.write(dispatch_id().y * res.x + dispatch_id().x, color)


def main():
    print("Starting to parse scene data...")
    parse_scene(sys.argv[1])
    print("Preparing canvas...")
    canvas = luisa.Buffer(res.x * res.y, dtype=float3)
    print("Calculating parameters...")
    calculate_parameters()
    print("Starting path tracing...")
    path_tracing_kernel(canvas, dispatch_size=(res.x, res.y))
    buffer = np.array([[c.x, c.y, c.z] for c in canvas.to_list()], dtype=np.float32)
    postfix = sys.argv[1].split('/')[-1].split('\\')[-1].split('.')[0]
    plt.imsave(f"result-{postfix}.png", buffer.reshape(res.x, res.y, -1)[::-1, ::-1])


if __name__ == "__main__":
    main()
