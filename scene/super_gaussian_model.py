import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.mirror_activation = torch.sigmoid

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._quadrant = torch.empty(0)
        self._mirror_quadrant = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._opacity_for_mirror = torch.empty(0)

        self._mirror_opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._quadrant,
            self._mirror_quadrant,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_for_mirror,
            self._mirror_opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._quadrant,
         self._mirror_quadrant,

         self._scaling,
         self._rotation,
         self._opacity,
         self._opacity_for_mirror,
         self._mirror_opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # .clamp(max=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        quadrant = self._quadrant
        return torch.cat((features_dc, features_rest, quadrant), dim=1)
    @property
    def get_ori_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_mirror_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        quadrant = self._mirror_quadrant
        return torch.cat((features_dc, features_rest, quadrant), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_for_mirror(self):
        return self.opacity_activation(self._opacity_for_mirror)

    @property
    def get_mirror_opacity(self):
        return self.mirror_activation(self._mirror_opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_empty(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        gsnum = 1
        fused_point_cloud = torch.zeros(gsnum, 3).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray([[0.1, 0.1, 0.1]])).float().cuda())
        features = torch.zeros((gsnum, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        colors = torch.ones((gsnum, 8, 3)).float().cuda()
        colors = colors * 0.0
        colors[:, 4, 0] = -1.0
        colors[:, 4, 1] = -1.0
        colors[:, 4, 2] = 0.1
        colors[:, 5, 0] = 1.0
        colors[:, 5, 1] = -1.0
        colors[:, 5, 2] = 0.1
        colors[:, 6, 0] = 1.0
        colors[:, 6, 1] = 1.0
        colors[:, 6, 2] = 0.1
        colors[:, 7, 0] = -1.0
        colors[:, 7, 1] = 1.0
        colors[:, 7, 2] = 0.1

        # mir
        mirror_colors = torch.ones((gsnum, 8, 3)).float().cuda()
        mirror_colors = mirror_colors * 0.0
        mirror_colors[:, 4, 0] = -1.0
        mirror_colors[:, 4, 1] = -1.0
        mirror_colors[:, 4, 2] = 0.1
        mirror_colors[:, 5, 0] = 1.0
        mirror_colors[:, 5, 1] = -1.0
        mirror_colors[:, 5, 2] = 0.1
        mirror_colors[:, 6, 0] = 1.0
        mirror_colors[:, 6, 1] = 1.0
        mirror_colors[:, 6, 2] = 0.1
        mirror_colors[:, 7, 0] = -1.0
        mirror_colors[:, 7, 1] = 1.0
        mirror_colors[:, 7, 2] = 0.1

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.ones((fused_point_cloud.shape[0]), device="cuda")
        scales = scales * 0.01
        scales = torch.log(torch.sqrt(scales))[..., None].repeat(1, 2)
        # scales =scales * 10.0
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[0,0] = 0.1486
        # rots[0,1] = 0.6923
        # rots[0,2] = 0.1550
        # rots[0,3] = 0.6889

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 4), dtype=torch.float, device="cuda"))
        opacities_for_mirror = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))

        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._quadrant = nn.Parameter(colors.requires_grad_(True))

        self._mirror_quadrant = nn.Parameter(mirror_colors.requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._opacity_for_mirror = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        colors = torch.zeros((fused_color.shape[0], 8, 3)).float().cuda()
        colors[:, 4, 0] = -1.0
        colors[:, 4, 1] = -1.0
        colors[:, 4, 2] = 0.1
        colors[:, 5, 0] = 1.0
        colors[:, 5, 1] = -1.0
        colors[:, 5, 2] = 0.1
        colors[:, 6, 0] = 1.0
        colors[:, 6, 1] = 1.0
        colors[:, 6, 2] = 0.1
        colors[:, 7, 0] = -1.0
        colors[:, 7, 1] = 1.0
        colors[:, 7, 2] = 0.1

        # mir
        mirror_colors = torch.zeros((fused_color.shape[0], 8, 3)).float().cuda()

        # mirror_colors = torch.ones((fused_color, 8, 3)).float().cuda()
        mirror_colors = mirror_colors * 0.0
        mirror_colors[:, 4, 0] = -1.0
        mirror_colors[:, 4, 1] = -1.0
        mirror_colors[:, 4, 2] = 0.1
        mirror_colors[:, 5, 0] = 1.0
        mirror_colors[:, 5, 1] = -1.0
        mirror_colors[:, 5, 2] = 0.1
        mirror_colors[:, 6, 0] = 1.0
        mirror_colors[:, 6, 1] = 1.0
        mirror_colors[:, 6, 2] = 0.1
        mirror_colors[:, 7, 0] = -1.0
        mirror_colors[:, 7, 1] = 1.0
        mirror_colors[:, 7, 2] = 0.1

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # dist2 = dist2 * 2.0 # scale 
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        # scales = scales * 2.0
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 4), dtype=torch.float, device="cuda"))
        opacities_for_mirror = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # mir
        mirror_opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._quadrant = nn.Parameter(colors.requires_grad_(True))
        # mir
        self._mirror_quadrant = nn.Parameter(mirror_colors.requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # mir
        self._opacity_for_mirror = nn.Parameter(opacities_for_mirror.requires_grad_(True))

        self._mirror_opacity = nn.Parameter(mirror_opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._quadrant], 'lr': training_args.feature_lr / 2.0, "name": "quadrant"},  # new
            {'params': [self._mirror_quadrant], 'lr': training_args.feature_lr / 2, "name": "mirror_quadrant"},  # new

            {'params': [self._opacity], 'lr': training_args.opacity_lr / 1.0, "name": "opacity"},
            {'params': [self._opacity_for_mirror], 'lr': training_args.opacity_lr / 1.0, "name": "opacity_for_mirror"},

            {'params': [self._mirror_opacity], 'lr': training_args.opacity_lr, "name": "mirror_opacity"},

            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._quadrant.shape[1] * self._quadrant.shape[2]):
            l.append('quadrant_{}'.format(i))
        for i in range(self._mirror_quadrant.shape[1] * self._mirror_quadrant.shape[2]):
            l.append('mirror_quadrant_{}'.format(i))
        for i in range(self._opacity.shape[1]):
            l.append('opacity_{}'.format(i))
        for i in range(self._opacity_for_mirror.shape[1]):
            l.append('opacity_for_mirror_{}'.format(i))

        for i in range(self._mirror_opacity.shape[1]):
            l.append('mirror_opacity_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        quadrants = self._quadrant.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mirror_quadrants = self._mirror_quadrant.detach().transpose(1, 2).flatten(
            start_dim=1).contiguous().cpu().numpy()

        # indices = torch.arange(0, self._features_dc.size(0), 4)

        opacities = self._opacity.detach().cpu().numpy()
        opacities_for_mirror = self._opacity_for_mirror.detach().cpu().numpy()

        mirror_opacities = self._mirror_opacity.detach().cpu().numpy()

        # opacities = np.mean(opacities, axis=1, keepdims=True)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, quadrants, opacities, scale, rotation), axis=1)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, quadrants, mirror_quadrants, opacities, opacities_for_mirror, mirror_opacities,
             scale, rotation),
            axis=1)

        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self ,   super_render:bool):
        if super_render:
            opacities_new = self.inverse_opacity_activation(
                torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        else:

            opacities_new_for_mirror = self.inverse_opacity_activation(
                torch.min(self.get_opacity_for_mirror, torch.ones_like(self.get_opacity_for_mirror) * 0.01))
            optimizable_tensors_for_mirror = self.replace_tensor_to_optimizer(opacities_new_for_mirror,
                                                                              "opacity_for_mirror")
            self._opacity_for_mirror = optimizable_tensors_for_mirror["opacity_for_mirror"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("quadrant_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        quadrants = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            quadrants[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        quadrants = quadrants.reshape((quadrants.shape[0], 3, 8))

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mirror_quadrant_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        mirror_quadrants = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            mirror_quadrants[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        mirror_quadrants = mirror_quadrants.reshape((mirror_quadrants.shape[0], 3, 8))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opa_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_")]
        opa_names = sorted(opa_names, key=lambda x: int(x.split('_')[-1]))
        opacities = np.zeros((xyz.shape[0], len(opa_names)))
        for idx, attr_name in enumerate(opa_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opa_names_mirror = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_for_mirror_")]
        opa_names_mirror = sorted(opa_names_mirror, key=lambda x: int(x.split('_')[-1]))
        opacities_for_mirror = np.zeros((xyz.shape[0], len(opa_names_mirror)))
        for idx, attr_name in enumerate(opa_names_mirror):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])

        mirror_opa_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mirror_opacity_")]
        mirror_opa_names = sorted(mirror_opa_names, key=lambda x: int(x.split('_')[-1]))
        mirror_opacities = np.zeros((xyz.shape[0], len(mirror_opa_names)))
        for idx, attr_name in enumerate(mirror_opa_names):
            mirror_opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._quadrant = nn.Parameter(
            torch.tensor(quadrants, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mirror_quadrant = nn.Parameter(
            torch.tensor(mirror_quadrants, dtype=torch.float, device="cuda").transpose(1,
                                                                                       2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_for_mirror = nn.Parameter(
            torch.tensor(opacities_for_mirror, dtype=torch.float, device="cuda").requires_grad_(True))

        self._mirror_opacity = nn.Parameter(
            torch.tensor(mirror_opacities, dtype=torch.float, device="cuda").requires_grad_(True))

        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            thismask = mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][thismask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][thismask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][thismask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][thismask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._quadrant = optimizable_tensors["quadrant"]
        self._mirror_quadrant = optimizable_tensors["mirror_quadrant"]

        self._opacity = optimizable_tensors["opacity"]
        self._opacity_for_mirror = optimizable_tensors["opacity_for_mirror"]

        self._mirror_opacity = optimizable_tensors["mirror_opacity"]

        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_quadrant, new_mirror_quadrant,
                              new_opacities, new_opacities_for_mirror, new_mirror_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "quadrant": new_quadrant,
             "mirror_quadrant": new_mirror_quadrant,

             "opacity": new_opacities,
             "opacity_for_mirror": new_opacities_for_mirror,
             "mirror_opacity": new_mirror_opacities,

             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._quadrant = optimizable_tensors["quadrant"]

        self._mirror_quadrant = optimizable_tensors["mirror_quadrant"]
        self._opacity = optimizable_tensors["opacity"]
        self._opacity_for_mirror = optimizable_tensors["opacity_for_mirror"]

        self._mirror_opacity = optimizable_tensors["mirror_opacity"]

        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, max_gs, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        if n_init_points < max_gs:
            if selected_pts_mask.sum() + n_init_points > max_gs:
                limited_num = max_gs - n_init_points
                padded_grad[~selected_pts_mask] = 0
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grad, (1.0 - ratio))
                selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            if selected_pts_mask.sum() > 0:
                stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
                stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
                means = torch.zeros_like(stds)
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,
                                                                                                                      1)
                new_scaling = self.scaling_inverse_activation(
                    self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
                new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

                new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
                new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)

                new_quadrant = self._quadrant[selected_pts_mask].repeat(N, 1, 1)
                new_mirror_quadrant = self._mirror_quadrant[selected_pts_mask].repeat(N, 1, 1)

                new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
                new_opacity_for_mirror = self._opacity_for_mirror[selected_pts_mask].repeat(N, 1)

                new_mirror_opacity = self._mirror_opacity[selected_pts_mask].repeat(N, 1)

                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_quadrant,
                                           new_mirror_quadrant, new_opacity, new_opacity_for_mirror,
                                           new_mirror_opacity, new_scaling, new_rotation)
                prune_filter = torch.cat(
                    (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
                self.prune_points(prune_filter)

    def densify_and_split_by_scale(self, max_gs, scale_threshold=1.0, scene_extent=1.0, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # padded_grad = torch.zeros((n_init_points), device="cuda")
        # padded_grad[:grads.shape[0]] = grads.squeeze()
        scales = self.get_scaling
        scales = torch.max(scales, dim=1).values

        selected_pts_mask = torch.where(scales >= scale_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        if n_init_points < max_gs:
            if selected_pts_mask.sum() + n_init_points > max_gs:
                limited_num = max_gs - n_init_points
                scales[~selected_pts_mask] = 0
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(scales, (1.0 - ratio))
                selected_pts_mask = torch.where(scales > threshold, True, False)
            if selected_pts_mask.sum() > 0:
                stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
                stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
                means = torch.zeros_like(stds)
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,
                                                                                                                      1)
                new_scaling = self.scaling_inverse_activation(
                    self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
                new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

                new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
                new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)

                new_quadrant = self._quadrant[selected_pts_mask].repeat(N, 1, 1)

                new_mirror_quadrant = self._mirror_quadrant[selected_pts_mask].repeat(N, 1, 1)

                new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
                new_opacity_for_mirror = self._opacity_for_mirror[selected_pts_mask].repeat(N, 1)

                new_mirror_opacity = self._mirror_opacity[selected_pts_mask].repeat(N, 1)

                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_quadrant,
                                           new_mirror_quadrant, new_opacity, new_opacity_for_mirror,new_mirror_opacity, new_scaling,
                                           new_rotation)
                prune_filter = torch.cat(
                    (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
                self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, max_gs):
        # Extract points that satisfy the gradient condition
        n_init_points = self.get_xyz.shape[0]
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        if n_init_points < max_gs:
            if selected_pts_mask.sum() + n_init_points > max_gs:
                limited_num = max_gs - n_init_points
                grads_tmp = grads.squeeze().clone()
                grads_tmp[~selected_pts_mask] = 0
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(grads_tmp, (1.0 - ratio))
                selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

            if selected_pts_mask.sum() > 0:
                new_xyz = self._xyz[selected_pts_mask]
                new_features_dc = self._features_dc[selected_pts_mask]
                new_features_rest = self._features_rest[selected_pts_mask]
                new_quadrant = self._quadrant[selected_pts_mask]
                new_mirror_quadrant = self._mirror_quadrant[selected_pts_mask]

                new_opacities = self._opacity[selected_pts_mask]
                new_opacities_for_mirror = self._opacity_for_mirror[selected_pts_mask]

                new_scaling = self._scaling[selected_pts_mask]
                new_rotation = self._rotation[selected_pts_mask]
                new_mirror_opacity = self._mirror_opacity[selected_pts_mask]

                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_quadrant,
                                           new_mirror_quadrant, new_opacities, new_opacities_for_mirror,
                                           new_mirror_opacity, new_scaling,
                                           new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_gs=1e9,super_render = False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, max_gs)
        self.densify_and_split(grads, max_grad, extent, max_gs)

        self.densify_and_split_by_scale(max_gs, 1.0, extent)


        opacities = self.get_opacity if super_render else self.get_opacity_for_mirror

        opacities = torch.max(opacities, dim=1).values.unsqueeze(-1)
        # mirror_opacities = self.get_mirror_opacity
        #
        # mirror_opacities = torch.max(opacities, dim=1).values.unsqueeze(-1)

        prune_mask = (opacities < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def super_train_iit(self):
        self._opacity = self._opacity_for_mirror.repeat(1, 1, 4)

    @torch.no_grad()
    def compute_mirror_plane(self, min_opacity, sansac_threshold=0.01):
        # filter mirror points

        # opacity = self.get_opacity.mean(-1).unsqueeze(-1)
        # opacity = self.get_opacity.mean(-1).unsqueeze(-

        valid_points_mask = (self.get_mirror_opacity > min_opacity).squeeze() & (
                self.get_opacity_for_mirror > min_opacity).squeeze()
        # self.get_opacity > min_opacity).squeeze()
        mirror_xyz = self._xyz[valid_points_mask]

        # compute plane parameters
        # points = mirror_xyz.detach()
        # center = points.mean(0)
        # covariance_matrix = points - center
        # covariance_matrix = torch.matmul(covariance_matrix.transpose(0, 1), covariance_matrix)
        # eig_value, eig_vector = torch.linalg.eigh(covariance_matrix)
        # normal = eig_vector[:, 0]
        # a, b, c = normal[0].item(), normal[1].item(), normal[2].item()
        # d = -torch.matmul(normal, center).item()
        from utils import ransac
        self.mirror_equ, mirror_pts_ids = ransac.Plane(mirror_xyz.detach().cpu().numpy(), sansac_threshold)

        # mirror_transform
        a, b, c, d = self.mirror_equ[0], self.mirror_equ[1], self.mirror_equ[2], self.mirror_equ[3]
        mirror_transform = np.array([
            1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d,
            -2 * a * b, 1 - 2 * b * b, -2 * b * c, -2 * b * d,
            -2 * a * c, -2 * b * c, 1 - 2 * c * c, -2 * c * d,
            0, 0, 0, 1
        ]).reshape(4, 4)
        mirror_transform = torch.as_tensor(mirror_transform, dtype=torch.float, device="cuda")

        #
        # dist = ((
        #     mirror_xyz[:, 0] * a + mirror_xyz[:, 1] * b + mirror_xyz[:, 2] * c + d
        #     ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)).abs().detach()
        # outlier_mask =  dist > (dist.min() * 0.7 + dist.max() * 0.3)
        # self._opacity[valid_points_mask][outlier_mask] = -100
        # trimesh.points.PointCloud(mirror_xyz[outlier_mask.logical_not()].detach().cpu().numpy()).export("filter.ply")

        return mirror_transform

    def get_plane_error(self, save_mirror_path=None, min_opacity=0.5):
        """enforcing the mirror points close to the plane"""
        opacity = self.get_opacity.mean(-1).unsqueeze(-1)
        opacity = self.get_opacity_for_mirror
        valid_points_mask = (self.get_mirror_opacity > min_opacity).squeeze() & (
                opacity > min_opacity).squeeze()
        # self.get_opacity > min_opacity).squeeze()
        mirror_xyz = self._xyz[valid_points_mask]
        import trimesh

        mirror_opacity = self.get_mirror_opacity[valid_points_mask]

        if save_mirror_path is not None:
            trimesh.points.PointCloud(mirror_xyz.detach().cpu().numpy(),
                                      mirror_opacity.repeat(1, 4).detach().cpu().numpy()).export(save_mirror_path)

        # if save_mirror_path is not None:
        #     trimesh.points.PointCloud(mirror_xyz.detach().cpu().numpy()).export(save_mirror_path)

        a, b, c, d = self.mirror_equ[0], self.mirror_equ[1], self.mirror_equ[2], self.mirror_equ[3]
        dist = ((
                        mirror_xyz[:, 0] * a + mirror_xyz[:, 1] * b + mirror_xyz[:, 2] * c + d
                ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)).abs()

        # outlier_mask =  dist > (dist.min() * 0.7 + dist.max() * 0.3)
        # self._opacity[valid_points_mask][outlier_mask.detach()] *= -100

        return dist
