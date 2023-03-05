import math
import time

import dearpygui.dearpygui as dpg
import numpy as np
from matplotlib import pyplot as plt

import luisa
from luisa.accel import make_ray, offset_ray_origin
from luisa.framerate import FrameRate
from luisa.mathtypes import *
from luisa.util import RandomSampler
from luisa.window import Window
from scene import parse_scene

render_scene = parse_scene("./test.sc")

SIGMA_A = make_float3(*render_scene.material.sigma_a)
SIGMA_S = make_float3(*render_scene.material.sigma_s)
G = render_scene.material.g
ETA = render_scene.material.eta
print(f"sigma_a: {SIGMA_A}")
print(f"sigma_s: {SIGMA_S}")
print(f"g: {G}")
print(f"eta: {ETA}")
EFFECTIVE_SIGMA_S = SIGMA_S * (1.0 - G)
REDUCED_SIGMA_T = EFFECTIVE_SIGMA_S + SIGMA_A
EFFECTIVE_ALPHA = EFFECTIVE_SIGMA_S / REDUCED_SIGMA_T
FRESNEL_DIFFUSE = -1.440 / (ETA * ETA) + 0.710 / ETA + 0.668 + 0.0636 * ETA
A = (1.0 + FRESNEL_DIFFUSE) / (1.0 - FRESNEL_DIFFUSE)
print(f"A: {A}")
DIFFUSE_ALBEDO = 0.5 * EFFECTIVE_ALPHA * (1.0 + make_float3(
    math.exp(-4.0 / 3.0 * A * math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.x))),
    math.exp(-4.0 / 3.0 * A * math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.y))),
    math.exp(-4.0 / 3.0 * A * math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.z))))
                                          ) * make_float3(
    math.exp(-math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.x))),
    math.exp(-math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.y))),
    math.exp(-math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.z)))
)
print(f"diffuse albedo: {DIFFUSE_ALBEDO}")
SIGMA_TR = make_float3(
    math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.x)),
    math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.y)),
    math.sqrt(3.0 * (1.0 - EFFECTIVE_ALPHA.z))
) * REDUCED_SIGMA_T
L_DIFFUSE = 1.0 / SIGMA_TR
print(f"l diffuse: {L_DIFFUSE}")
D = L_DIFFUSE / (3.5 + 100 * make_float3(
    (DIFFUSE_ALBEDO.x - 0.33) ** 4,
    (DIFFUSE_ALBEDO.y - 0.33) ** 4,
    (DIFFUSE_ALBEDO.z - 0.33) ** 4
))
print(f"d: {D}")
CAMERA_ORIGIN = make_float3(0.0, 1.0, 8.0)
CAMERA_LOOK_AT = make_float3(0.0, 0.5, 0.0)
CAMERA_FOV = 20.0

MAX_DEPTH = 24

luisa.init()
heap = luisa.BindlessArray()
vertex_buffer = luisa.Buffer(len(render_scene.vertex_arr), float3)
vertex_arr = np.array([[*vertex, 0.0]
                       for vertex in render_scene.vertex_arr], dtype=np.float32)
vertex_buffer.copy_from(vertex_arr)
normal_buffer = luisa.Buffer(len(render_scene.normal_arr), float3)
normal_arr = np.array([[*normal, 0.0]
                       for normal in render_scene.normal_arr], dtype=np.float32)
normal_buffer.copy_from(normal_arr)
emission_buffer = luisa.Buffer(len(render_scene.emissions), float3)
emission_arr = np.array([[*emission, 0.0]
                         for emission in render_scene.emissions], dtype=np.float32)
emission_buffer.copy_from(emission_arr)

accel = luisa.Accel()
mesh_num = len(render_scene.meshes)
for mesh_cnt, mesh in enumerate(render_scene.meshes):
    vertex_index_buffer = luisa.Buffer(len(mesh.vertex_indices), int)
    vertex_index_buffer.copy_from(
        np.array(mesh.vertex_indices, dtype=np.int32))
    heap.emplace(mesh_cnt, vertex_index_buffer)
    accel.add(luisa.Mesh(vertex_buffer, vertex_index_buffer))
    normal_index_buffer = luisa.Buffer(len(mesh.normal_indices), int)
    normal_index_buffer.copy_from(
        np.array(mesh.normal_indices, dtype=np.int32))
    heap.emplace(mesh_cnt + mesh_num, normal_index_buffer)
accel.update()
heap.update()

Onb = luisa.StructType(tangent=float3, binormal=float3, normal=float3)


@luisa.func
def to_world(self, v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal


Onb.add_method(to_world, "to_world")


@luisa.func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                        12.92 * x,
                        x <= 0.00031308),
                 0.0, 1.0)


@luisa.func
def make_onb(normal: float3):
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
def generate_ray(p, resolution):
    forward = normalize(CAMERA_LOOK_AT - CAMERA_ORIGIN)
    right = normalize(cross(forward, make_float3(0.0, 1.0, 0.0)))
    up = cross(right, forward)
    fov = CAMERA_FOV * 3.1415926 / 180
    height = 2.0 * tan(0.5 * fov)
    width = resolution.x / resolution.y * height
    direction = normalize(forward + (0.5 - p.x) * width * right + (0.5 - p.y) * height * up)
    return make_ray(CAMERA_ORIGIN, direction, 0.0, 1e30)


@luisa.func
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2.0 * math.pi * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))


@luisa.func
def normalized_bssrdf(d: float, r: float) -> float:
    exponent = exp(-r / (3.0 * d))
    return (exponent + exponent * exponent * exponent) / (8 * math.pi * r * d)


@luisa.func
def pdf_disk(d: float, r: float) -> float:
    c = cdf_disk(d, r)
    exponent = exp(-r / (3.0 * d))
    return (exponent + exponent * exponent * exponent) * abs(c * log(c)) / (8 * math.pi * r * d)


@luisa.func
def cdf_disk(d: float, r: float) -> float:
    exponent = exp(-r / (3.0 * d))
    return 1.0 - 0.25 * exponent * exponent * exponent - 0.75 * exponent


@luisa.func
def invert_cdf_disk(d: float, x: float) -> float:
    q = 4.0 * (x - 1.0)
    x = pow(-0.5 * q + sqrt(0.25 * q * q + 1), 1 / 3) - pow(0.5 * q + sqrt(0.25 * q * q + 1), 1 / 3)
    return -3.0 * log(x) * d


@luisa.func
def fresnel_moment(eta: float) -> float:
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    if eta < 1:
        return 0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3 + 2.49277 * eta4 - 0.68441 * eta5
    else:
        return -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3 - 1.27198 * eta4 + 0.12746 * eta5


@luisa.func
def fresnel_reflectance(cos_wi: float, eta1: float, eta2: float) -> float:
    sin_wi = sqrt(max(0.0, 1.0 - cos_wi * cos_wi))
    sin_wt = eta1 / eta2 * sin_wi
    cos_wt = sqrt(max(0.0, 1.0 - sin_wt * sin_wt))
    r_parallel = (eta2 * cos_wi - eta1 * cos_wt) / (eta2 * cos_wi + eta1 * cos_wt)
    r_perpendicular = (eta1 * cos_wi - eta2 * cos_wt) / (eta1 * cos_wi + eta2 * cos_wt)
    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) * 0.5


@luisa.func
def fresnel_transmittance(eta: float, cos_w: float):
    return 1.0 - fresnel_reflectance(cos_w, 1.0, eta)


@luisa.func
def raytracing_kernel(image, accel, resolution, frame_index):
    set_block_size(8, 8, 1)
    coord = dispatch_id().xy
    sampler = RandomSampler(make_int3(coord, frame_index))
    rx = sampler.next()
    ry = sampler.next()
    pixel = (make_float2(coord) + make_float2(rx, ry)) / make_float2(resolution)
    ray = generate_ray(pixel, resolution)
    radiance = make_float3(0.0)
    beta = make_float3(1.0)
    pdf_bsdf = 1e30
    pdf_light = 0.0
    for depth in range(MAX_DEPTH):
        # trace
        hit = accel.trace_closest(ray)
        if hit.miss():
            break
        i0 = heap.buffer_read(int, hit.inst, hit.prim * 3)
        i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
        p0, p1, p2 = vertex_buffer.read(i0), vertex_buffer.read(i1), vertex_buffer.read(i2)
        p = hit.interpolate(p0, p1, p2)
        i0 = heap.buffer_read(int, hit.inst + mesh_num, hit.prim * 3)
        i1 = heap.buffer_read(int, hit.inst + mesh_num, hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, hit.inst + mesh_num, hit.prim * 3 + 2)
        n0, n1, n2 = normal_buffer.read(i0), normal_buffer.read(i1), normal_buffer.read(i2)
        n = normalize(hit.interpolate(n0, n1, n2))
        onb = make_onb(n)
        cos_wo = dot(-ray.get_dir(), n)
        if cos_wo < 0.0:
            n = -1.0 * n
            cos_wo *= -1.0
        if cos_wo < 1e-4: break
        if hit.inst != 0:
            if depth == 0:
                radiance += emission_buffer.read(hit.inst - 1)
            else:
                radiance += beta * emission_buffer.read(hit.inst - 1) / max(1e-4, pdf_bsdf)
                # light_area = length(cross(p1 - p0, p2 - p0))
                # pdf_light = length_squared(p - ray.get_origin()) / (light_area * cos_wo * (mesh_num - 1))
                # mis_weight = pdf_bsdf / (pdf_light + pdf_bsdf)
                # radiance += mis_weight * beta * emission_buffer.read(hit.inst - 1) / max(1e-4, pdf_bsdf)
            break
        # Sample Incident Point Using Disk CDF
        ux, uy = 3.0 * sampler.next(), sampler.next()
        d = D.x
        pdf_channel = 1.0 / 3.0
        if ux > 2.0:
            d = D.z
            ux -= 2.0
        elif ux > 1.0:
            d = D.y
            ux -= 1.0
        x = onb.tangent
        y = onb.binormal
        z = n
        pdf_axis = 0.5
        if ux > 0.75:
            pdf_axis = 0.25
            ux = 4.0 * (ux - 0.75)
            x = onb.binormal
            y = onb.normal
            z = onb.tangent
        elif ux > 0.5:
            pdf_axis = 0.25
            ux = 4.0 * (ux - 0.5)
            x = onb.normal
            y = onb.tangent
            z = onb.binormal
        else:
            ux = 2.0 * ux
        if ux < 0.5:
            z *= -1.0
            ux *= 2.0
        else:
            ux = 2.0 * (ux - 0.5)
        uz = sampler.next()
        r_max = invert_cdf_disk(d, uz)
        r = invert_cdf_disk(d, ux * uz)
        theta = uy * 2.0 * math.pi
        h = sqrt(max(0.0, r_max * r_max - r * r))
        proj = p + z * h + x * r * cos(theta) + y * r * sin(theta)
        proj_dir = -z
        proj_ray = make_ray(proj, proj_dir, 0.0, 2.0 * h)
        proj_hit = accel.trace_closest(proj_ray)
        if proj_hit.miss() or proj_hit.inst != 0: break
        i0 = heap.buffer_read(int, 0, proj_hit.prim * 3)
        i1 = heap.buffer_read(int, 0, proj_hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, 0, proj_hit.prim * 3 + 2)
        p0, p1, p2 = vertex_buffer.read(i0), vertex_buffer.read(i1), vertex_buffer.read(i2)
        proj_p = proj_hit.interpolate(p0, p1, p2)
        i0 = heap.buffer_read(int, mesh_num, proj_hit.prim * 3)
        i1 = heap.buffer_read(int, mesh_num, proj_hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, mesh_num, proj_hit.prim * 3 + 2)
        n0, n1, n2 = normal_buffer.read(i0), normal_buffer.read(i1), normal_buffer.read(i2)
        proj_n = normalize(proj_hit.interpolate(n0, n1, n2))
        pp = offset_ray_origin(proj_p, proj_n)
        proj_onb = make_onb(proj_n)
        pdf_bssrdf = 0.0
        offset = proj_p - p
        radius = length(cross(offset, onb.normal))
        pdf_bssrdf += (pdf_disk(D.x, radius) + pdf_disk(D.y, radius) + pdf_disk(D.z, radius)) * abs(
            dot(proj_n, onb.normal))
        radius = length(cross(offset, onb.tangent))
        pdf_bssrdf += (pdf_disk(D.x, radius) + pdf_disk(D.y, radius) + pdf_disk(D.z, radius)) * abs(
            dot(proj_n, onb.tangent))
        radius = length(cross(offset, onb.binormal))
        pdf_bssrdf += (pdf_disk(D.x, radius) + pdf_disk(D.y, radius) + pdf_disk(D.z, radius)) * abs(
            dot(proj_n, onb.binormal))
        beta *= fresnel_transmittance(ETA, cos_wo) * DIFFUSE_ALBEDO * normalized_bssrdf(d, length(offset)) / (
                    math.pi * (1.0 - 2.0 * fresnel_moment(1.0 / ETA)) * pdf_bssrdf * pdf_axis * pdf_channel)
        """
        # Sample Light
        ux_light, uy_light = sampler.next(), sampler.next()
        inst_light = min(mesh_num - 2, int((mesh_num - 1) * ux_light))
        ux_light = ux_light * (mesh_num - 1) - inst_light
        prim_light = 0
        if ux_light > 0.5:
            prim_light = 1
            ux_light = 2.0 * (ux_light - 0.5)
        else:
            ux_light *= 2.0
        if ux_light + uy_light > 1.0:
            ux_light = 1.0 - ux_light
            uy_light = 1.0 - uy_light
        i0 = heap.buffer_read(int, inst_light + 1, prim_light * 3)
        i1 = heap.buffer_read(int, inst_light + 1, prim_light * 3 + 1)
        i2 = heap.buffer_read(int, inst_light + 1, prim_light * 3 + 2)
        p0, p1, p2 = vertex_buffer.read(i0), vertex_buffer.read(i1), vertex_buffer.read(i2)
        p_light = p0 + ux_light * (p1 - p0) + uy_light * (p2 - p0)
        i0 = heap.buffer_read(int, inst_light + mesh_num + 1, prim_light * 3)
        i1 = heap.buffer_read(int, inst_light + mesh_num + 1, prim_light * 3 + 1)
        i2 = heap.buffer_read(int, inst_light + mesh_num + 1, prim_light * 3 + 2)
        n0, n1, n2 = normal_buffer.read(i0), normal_buffer.read(i1), normal_buffer.read(i2)
        n_light = normalize(n0 + ux_light * (n1 - n0) + uy_light * (n2 - n0))
        pp_light = offset_ray_origin(p_light, n_light)
        wi_light = normalize(pp_light - pp)
        d_light = length(pp_light - pp)
        shadow_ray = make_ray(pp, wi_light, 0.0, d_light)
        occluded = accel.trace_any(shadow_ray)
        cos_wi_light = dot(proj_n, wi_light)
        cos_light = -dot(n_light, wi_light)
        if ((not occluded and cos_wi_light > 1e-4) and cos_light > 1e-4):
            light_area = length(cross(p1 - p0, p2 - p0))
            pdf_light = (d_light * d_light) / (light_area * cos_light * (mesh_num - 1))
            pdf_bsdf = cos_wi_light / math.pi
            mis_weight = pdf_light / (pdf_light + pdf_bsdf)
            bsdf = fresnel_transmittance(ETA, cos_wi_light)
            emission = emission_buffer.read(inst_light)
            radiance += beta * bsdf * mis_weight * emission * cos_wi_light / max(pdf_light, 1e-4)
        """
        # Sample Next Event Using Cosine Weight
        new_direction = proj_onb.to_world(cosine_sample_hemisphere(make_float2(sampler.next(), sampler.next())))
        cos_wi = abs(dot(proj_n, new_direction))
        pdf_bsdf = cos_wi / math.pi
        beta *= fresnel_transmittance(ETA, cos_wi) * cos_wi
        ray = make_ray(pp, new_direction, 0.0, 1e30)
        # Russian Roulette
        if depth > MAX_DEPTH / 2:
            l = dot(make_float3(0.212671, 0.715160, 0.072169), beta)
            if l == 0.0:
                break
            q = max(l, 0.05)
            r = sampler.next()
            if r >= q:
                break
            beta *= 1.0 / q
    if any(isnan(radiance)):
        radiance = make_float3(0.0)
    image.write(dispatch_id().xy, make_float4(clamp(radiance, 0.0, 30.0), 1.0))


@luisa.func
def accumulate_kernel(accum_image, curr_image):
    p = dispatch_id().xy
    accum = accum_image.read(p)
    curr = curr_image.read(p).xyz
    t = 1.0 / (accum.w + 1.0)
    accum_image.write(p, make_float4(lerp(accum.xyz, curr, t), accum.w + 1.0))


@luisa.func
def aces_tonemapping(x: float3):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


@luisa.func
def clear_kernel(image):
    image.write(dispatch_id().xy, make_float4(0.0))


@luisa.func
def hdr2ldr_kernel(hdr_image, ldr_image, scale: float):
    coord = dispatch_id().xy
    hdr = hdr_image.read(coord)
    ldr = linear_to_srgb(hdr.xyz * scale)
    ldr_image.write(coord, make_float4(ldr, 1.0))


res = 1024, 1024
image = luisa.Texture2D(*res, 4, float)
accum_image = luisa.Texture2D(*res, 4, float)
ldr_image = luisa.Texture2D(*res, 4, float)
arr = np.zeros([*res, 4], dtype=np.float32)
frame_rate = FrameRate(10)
w = Window("Normalized BSSRDF", res, resizable=False, frame_rate=True)
w.set_background(arr, res)
dpg.draw_image("background", (0, 0), res, parent="viewport_draw")
clear_kernel(accum_image, dispatch_size=[*res, 1])
frame_index = 0
sample_per_pass = 32


def update():
    global frame_index, arr
    for i in range(sample_per_pass):
        raytracing_kernel(image, accel, make_int2(
            *res), frame_index, dispatch_size=(*res, 1))
        accumulate_kernel(accum_image, image, dispatch_size=[*res, 1])
        frame_index += 1
    hdr2ldr_kernel(accum_image, ldr_image, 1.0, dispatch_size=[*res, 1])
    ldr_image.copy_to(arr)
    frame_rate.record(sample_per_pass)
    w.update_frame_rate(frame_rate.report())


w.run(update)
plt.imsave("output.png", arr)
