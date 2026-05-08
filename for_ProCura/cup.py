# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Adapted from original implementation by Lingjie Mei
# Modifications made by Tomasz Sadowy in 2025

import json
import os

import bpy
import numpy as np
from numpy.random import uniform


from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import text
from infinigen.assets.objects.tableware.base import TablewareFactory
from infinigen.assets.utils.decorate import (
    read_co,
    remove_vertices,
    subsurf,
    write_attribute,
)
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class CupFactory(TablewareFactory):
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.x_end = log_uniform(0.18, 0.32)
            self.is_short = True  # forced: guards only exist on short cups
            if self.is_short:
                self.is_profile_straight = uniform(0, 1) < 0.2
                self.x_lowest = log_uniform(0.6, 0.9)
                self.depth = log_uniform(0.25, 0.5)
                self.has_guard = True  # forced: always generate a handle
            else:
                self.is_profile_straight = True
                self.x_lowest = log_uniform(0.9, 1.0)
                self.depth = log_uniform(0.5, 1.0)
                self.has_guard = True
            if self.is_profile_straight:
                self.handle_location = uniform(0.45, 0.65)
            else:
                self.handle_location = uniform(-0.1, 0.3)
            self.handle_type = "shear" if uniform(0, 1) < 0.5 else "round"
            self.handle_radius = self.depth * uniform(0.2, 0.4)
            self.handle_inner_radius = self.handle_radius * log_uniform(0.2, 0.3)
            self.handle_taper_x = uniform(0, 2)
            self.handle_taper_y = uniform(0, 2)
            self.x_lower_ratio = log_uniform(0.8, 1.0)
            self.thickness = log_uniform(0.01, 0.04)
            self.bevel_width_pct = uniform(10, 50)
            self.profile_mid_z_ratio = uniform(0.35, 0.70)
            self.top_radius_ratio = uniform(0.75, 1.0)
            self.bottom_radius_ratio = uniform(0.45, 0.95)
            self.handle_angle_alpha = uniform(0, 1)
            self.handle_angle = None
            self.handle_side_angle = uniform(-np.pi, np.pi)
            self.handle_attach_width = self.handle_inner_radius * uniform(0.0, 0.8)
            self.handle_vertical_span = self.depth * uniform(0.25, 0.65)
            self.has_wrap = uniform() < 0.3
            self.wrap_margin = uniform(0.1, 0.2)

            self.wrap_surface = weighted_sample(material_assignments.graphicdesign)()()
            if self.wrap_surface == text.Text:
                self.wrap_surface = text.Text(self.factory_seed, False)

            self.has_inside = uniform(0, 1) < 0.5
            self.scale = log_uniform(0.15, 0.3)

        self.params = {
                "is_short": self.is_short,
                "is_profile_straight": self.is_profile_straight,
                "x_end": self.x_end,
                "x_lowest": self.x_lowest,
                "depth": self.depth,
                "has_guard": self.has_guard,
                "handle_location": self.handle_location,
                "handle_type": self.handle_type,
                "handle_radius": self.handle_radius,
                "handle_inner_radius": self.handle_inner_radius,
                "handle_taper_x": self.handle_taper_x,
                "handle_taper_y": self.handle_taper_y,
                "handle_angle_alpha": self.handle_angle_alpha,
                "handle_angle": self.handle_angle,
                "handle_side_angle": self.handle_side_angle,
                "handle_attach_width": self.handle_attach_width,
                "handle_vertical_span": self.handle_vertical_span,
                "x_lower_ratio": self.x_lower_ratio,
                "bevel_width_pct": self.bevel_width_pct,
                "profile_mid_z_ratio": self.profile_mid_z_ratio,
                "top_radius_ratio": self.top_radius_ratio,
                "bottom_radius_ratio": self.bottom_radius_ratio,
                "thickness": self.thickness,
                "has_wrap": self.has_wrap,
                "wrap_margin": self.wrap_margin,
                "has_inside": self.has_inside,
                "scale": self.scale,
            }


    def _save_params(self):
        output_root = os.environ.get("CUP_OUTPUT_DIR")
        if not output_root:
            return
        asset_dir = os.path.join(output_root, f"CupFactory_{self.factory_seed:03d}")
        os.makedirs(asset_dir, exist_ok=True)
        out_path = os.path.join(asset_dir, "params.json")

        serialisable = {}
        for k, v in self.params.items():
            if isinstance(v, (bool, int, float, str)):
                serialisable[k] = v
            elif hasattr(v, "item"):
                serialisable[k] = v.item()
            else:
                serialisable[k] = type(v).__name__
        serialisable["factory_seed"] = self.factory_seed
        with open(out_path, "w") as f:
                json.dump(serialisable, f, indent=2)

    def create_asset(self, **params) -> bpy.types.Object:
        bottom_radius = self.x_end * self.bottom_radius_ratio
        top_radius = self.x_end * self.top_radius_ratio
        if self.is_profile_straight:
            x_anchors = 0, bottom_radius, top_radius
            z_anchors = 0, 0, self.depth
        else:
            low_profile_radius = self.x_end * self.x_lowest
            mid_radius = (
                low_profile_radius
                + self.x_lower_ratio * (top_radius - low_profile_radius)
            )
            x_anchors = (
                0,
                bottom_radius,
                mid_radius,
                top_radius,
            )
            z_anchors = 0, 0, self.depth * self.profile_mid_z_ratio, self.depth
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [1])
        obj.scale = [1 / self.scale] * 3
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj,
            "BEVEL",
            True,
            offset_type="PERCENT",
            width_pct=self.bevel_width_pct,
            segments=8,
        )
        if self.has_wrap:
            wrap = self.make_wrap(obj)
        else:
            wrap = None
        self.solidify_with_inside(obj, self.thickness)
        subsurf(obj, 2)
        handle_radial_location = max(
            0.0,
            x_anchors[-2] * (1 - self.handle_location)
            + x_anchors[-1] * self.handle_location
            - self.handle_attach_width,
        )
        handle_location = (
            handle_radial_location * np.cos(self.handle_side_angle),
            handle_radial_location * np.sin(self.handle_side_angle),
            z_anchors[-2] * (1 - self.handle_location)
            + z_anchors[-1] * self.handle_location,
        )
        angle_low = np.arctan(
            (x_anchors[-1] - x_anchors[-2]) / (z_anchors[-1] - z_anchors[-2])
        )
        angle_height = np.arctan(
            (x_anchors[2] - x_anchors[1]) / (z_anchors[2] - z_anchors[1])
        )
        handle_angle = (
            angle_low * (1 - self.handle_angle_alpha)
            + (angle_height + 1e-3) * self.handle_angle_alpha
        )
        self.handle_angle = handle_angle
        self.params["handle_angle"] = handle_angle
        if self.has_guard:
            obj = self.add_handle(obj, handle_location, handle_angle)
        if self.has_wrap:
            butil.select_none()
            obj = join_objects([obj, wrap])
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        self._save_params()
        return obj

    def add_handle(self, obj, handle_location, handle_angle):
        bpy.ops.mesh.primitive_torus_add(
            location=handle_location,
            major_radius=self.handle_radius,
            minor_radius=self.handle_inner_radius,
        )
        handle = bpy.context.active_object
        handle.rotation_euler = np.pi / 2, handle_angle, self.handle_side_angle
        vertical_scale = self.handle_vertical_span / max(2 * self.handle_radius, 1e-6)
        handle.scale[1] = vertical_scale
        butil.apply_transform(handle, True)
        butil.modify_mesh(
            handle,
            "SIMPLE_DEFORM",
            deform_method="TAPER",
            angle=self.handle_taper_x,
            deform_axis="X",
        )
        butil.modify_mesh(
            handle,
            "SIMPLE_DEFORM",
            deform_method="TAPER",
            angle=self.handle_taper_y,
            deform_axis="Y",
        )
        butil.modify_mesh(handle, "BOOLEAN", object=obj, operation="DIFFERENCE")
        butil.select_none()
        objs = butil.split_object(handle)
        nonempty = []
        for o in objs:
            co = read_co(o)
            if co.size:
                nonempty.append((o, co))
        if not nonempty:
            butil.delete(objs)
            write_attribute(obj, lambda nw: 0, "guard", "FACE")
            return obj
        radial = np.array([np.cos(self.handle_side_angle), np.sin(self.handle_side_angle), 0])
        i = np.argmax([np.max(co @ radial) for _, co in nonempty])
        objs = [o for o, _ in nonempty]
        handle = objs[i]
        objs.remove(handle)
        butil.delete(objs)
        subsurf(handle, 1)
        write_attribute(handle, lambda nw: 1, "guard", "FACE")
        return join_objects([obj, handle])

    def make_wrap(self, obj):
        butil.select_none()
        obj = deep_clone_obj(obj)
        remove_vertices(
            obj,
            lambda x, y, z: (z / self.depth < self.wrap_margin)
            | (z / self.depth > 1 - self.wrap_margin + uniform(0.0, 0.1))
            | (np.abs(np.arctan2(y, x)) < np.pi * self.wrap_margin),
        )
        obj.scale = 1 + 1e-2, 1 + 1e-2, 1
        butil.apply_transform(obj)
        write_attribute(obj, lambda nw: 1, "text", "FACE")
        return obj

    def finalize_assets(self, assets):
        super().finalize_assets(assets)
        if self.has_wrap:
            for obj in assets if isinstance(assets, list) else [assets]:
                wrap_sides(obj, self.wrap_surface, "u", "v", "z", selection="text")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
