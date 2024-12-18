#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from random import randint
import numpy as np
import torch
import math
# from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from  diff_gaussian_rasterization_camera_grad import GaussianRasterizationSettings, GaussianRasterizer
import trimesh
from scene.origin_gaussian_model import GaussianModel
from utils.loss_utils import l1_loss
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
import open3d as o3d
from torchvision.utils import save_image
import copy
from torch import nn

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
mirror_transform=None, render_mirror_mask=False, remove_mirror=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # viewmatrix = viewpoint_camera.world_view_transform if mirror_transform is   None else viewpoint_camera.world_view_transform_mirror
    # projmatrix = viewpoint_camera.full_proj_transform if mirror_transform is   None else viewpoint_camera.full_proj_transform_mirror
    # campos = viewpoint_camera.camera_center if mirror_transform is   None else viewpoint_camera.camera_center_mirror
    viewmatrix = viewpoint_camera.world_view_transform
    projmatrix = viewpoint_camera.full_proj_transform
    campos = viewpoint_camera.camera_center


    if mirror_transform is not None:
        mirror_transform = pc.mirror_transform_model.get_cur_mirror_transform()
        w2c = viewmatrix.transpose(0, 1) # Q_o
        viewmatrix = torch.matmul(w2c, mirror_transform.inverse()).transpose(0, 1)
        projmatrix = (viewmatrix.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))).squeeze(0)
        campos = viewmatrix.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        # antialiasing=False,

        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity


        # opacity = opacity * (pc.get_mirror_opacity < 0.5)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    try:
        means3D.retain_grad()
    except:
        pass

    if mirror_transform is not None or remove_mirror:
        opacity = opacity * (1 - pc.get_mirror_opacity)
        means3D = means3D[pc.scene_point_mask]
        means2D = means2D[pc.scene_point_mask]
        shs = shs[pc.scene_point_mask]
        opacity = opacity[pc.scene_point_mask]
        scales = scales[pc.scene_point_mask]
        rotations = rotations[pc.scene_point_mask]
        if cov3D_precomp:
            cov3D_precomp = cov3D_precomp[cov3D_precomp]



    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        theta=viewpoint_camera.cam_rot_delta if mirror_transform is   None else viewpoint_camera.cam_rot_delta_mirror,
        rho=viewpoint_camera.cam_trans_delta if mirror_transform is   None else viewpoint_camera.cam_trans_delta_mirror,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii
    }

    if render_mirror_mask:
        colors_precomp = pc.get_mirror_opacity.repeat(1, 3)
        if mirror_transform:
            colors_precomp = colors_precomp[cov3D_precomp]
        mirror_mask, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        rets["mirror_mask"] = mirror_mask

    # additional regularizations
    # render_alpha = allmap[1:2]

    # # get normal map
    # render_normal = allmap[2:5]
    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    # # get median depth map
    # render_depth_median = allmap[5:6]
    # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # # get expected depth map
    # render_depth_expected = allmap[0:1]
    # render_depth_expected = (render_depth_expected / render_alpha)
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # # get depth distortion map
    # render_dist = allmap[6:7]

    # # psedo surface attributes
    # # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    # surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal, surf_point = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = surf_normal.permute(2,0,1)
    # surf_point = surf_point.permute(2,0,1)
    # # remember to multiply with accum_alpha since render_normal is unnormalized.
    # surf_normal = surf_normal * (render_alpha).detach()


    # rets.update({
    #         'rend_alpha': render_alpha,
    #         'rend_normal': render_normal,
    #         'rend_dist': render_dist,
    #         'surf_depth': surf_depth,
    #         'surf_normal': surf_normal,
    #         'surf_point': surf_point,
    # })

    return rets

def get_mirrot_points(viewpoint_stack_,bg_color,pc):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    shs = pc.get_features
    scales = pc.get_scaling
    rotations = pc.get_rotation
    # colors = pc.get_mirror_opacity.repeat(1, 3).clone()
    # colors = colors.detach().requires_grad_(True)
    colors = pc.get_mirror_opacity.repeat(1, 3)
    colors = colors.detach().requires_grad_(True)
    # colors= torch.zeros_like(pc.get_xyz,device=pc.get_xyz.device,requires_grad=True)


    # colors.
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device, requires_grad=False)
    viewpoint_stack =  copy.deepcopy(viewpoint_stack_)
    viewpoint_stack2 = copy.deepcopy(viewpoint_stack_)

    for i in range(len(viewpoint_stack)):
        viewpoint_camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        campos = viewpoint_camera.camera_center

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.active_sh_degree,
            campos=campos,
            prefiltered=False,
            # antialiasing=False,
            debug=True
            # pipe.debug

        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        grad, radii= rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )



        mask = viewpoint_camera.gt_alpha_mask>0.5
        mask = mask.repeat(3,1,1)
        target = grad *  mask .cuda().float()

        loss = 1 * target.sum()


        loss.backward(retain_graph=True)


        mins = torch.min(colors.grad, dim=-1).values
        maxes = torch.max(colors.grad, dim=-1).values
        assert torch.allclose(mins , maxes), "Something is wrong with gradient calculation"
        gaussian_grads += (colors.grad).norm(dim=[1])
        colors.grad.zero_()

        mask_inverted = ~mask
        target = grad * mask_inverted.cuda()
        loss = 1 * target.sum()
        loss.backward(retain_graph=True)
        gaussian_grads -= (colors.grad).norm(dim=[1])
        colors.grad.zero_()





    mask_3d = gaussian_grads > 0

    #vis
    means3Dn = pc.get_xyz[~mask_3d]
    means2Dn = screenspace_points[~mask_3d]
    opacityn = pc.get_opacity[~mask_3d]
    shsn = pc.get_features[~mask_3d]
    scalesn = pc.get_scaling[~mask_3d]
    rotationsn = pc.get_rotation[~mask_3d]
    for i in range(len(viewpoint_stack2)):
        viewpoint_camera = viewpoint_stack2.pop(randint(0, len(viewpoint_stack2) - 1))
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        campos = viewpoint_camera.camera_center
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.active_sh_degree,
            campos=campos,
            prefiltered=False,
            # antialiasing=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        img, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,

        )
        save_image(img, "outtemp/" + str(i) + ".png")

        # for nomask
        img, _  = rasterizer(
            means3D=means3Dn,
            means2D=means2Dn,
            shs=shsn,
            colors_precomp=None,
            opacities=opacityn,
            scales=scalesn,
            rotations=rotationsn,
            cov3D_precomp=None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        save_image(img, "outtemp/" + str(i) + "no.png")
    return mask_3d ,~mask_3d



def remove_fly_points(points:torch.Tensor,vis= False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
    inlier_cloud = pcd.select_by_index(ind)
    if vis:
        o3d.visualization.draw_geometries([inlier_cloud])

    return torch.asarray(np.asarray( inlier_cloud.points))


def calculate_mirror_transform(viewpoint_stack,pc:GaussianModel,pipe,bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    mirror_points_mask,scene_points_mask = get_mirrot_points(viewpoint_stack,bg_color,pc)

    mirror_points = remove_fly_points(pc.get_xyz[mirror_points_mask],vis=False)
    mirror_transform = pc.calculate_plane(mirror_points.numpy())
    # mirror_transform = calculate_plane(mirror_points.numpy(), pc)

    # apply mask
    pc.scene_point_mask = scene_points_mask
    pc.mirror_points_mask = mirror_points_mask

    return mirror_transform

    # for i in range(len(viewpoint_stack)):
    #     viewpoint_camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    #     render_pkg = render(viewpoint_camera, pc, pipe, bg_color)
    #
    #     # mirror_render_pkg = super_render(viewpoint_camera, pc, pipe, bg_color, mirror_transform=mirror_transform)
    #     mirror_render_pkg = render(viewpoint_camera, pc, pipe, bg_color,
    #                                      mirror_transform=mirror_transform)
    #     mirror_image = mirror_render_pkg["render"]
    #     image = render_pkg["render"]
    #     gt_mirror_mask = viewpoint_camera.gt_alpha_mask.repeat(3, 1, 1)
    #     super_image = image * (1 - gt_mirror_mask) + mirror_image * gt_mirror_mask
    #     save_image(super_image, "outtemp/" + str(i) + "mirror.png")
    #     save_image(viewpoint_camera.original_image, "outtemp/" + str(i) + "gt.png")
    #
    # sdf = 9


def calculate_plane(points,pc:GaussianModel):

    from utils import ransac
    pc.mirror_equ, mirror_pts_ids = ransac.Plane(points, 0.05)
    trimesh.points.PointCloud(points).export("outtemp/1.ply")
    pc.mirror_equ = torch.tensor(pc.mirror_equ).float().cuda()
    pc.mirror_equ_params = nn.Parameter(pc.mirror_equ.requires_grad_(True))

    # mirror_transform
    a, b, c, d = pc.mirror_equ[0], pc.mirror_equ[1], pc.mirror_equ[2], pc.mirror_equ[3]
    mirror_transform = torch.asarray([
        1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d,
        -2 * a * b, 1 - 2 * b * b, -2 * b * c, -2 * b * d,
        -2 * a * c, -2 * b * c, 1 - 2 * c * c, -2 * c * d,
        0, 0, 0, 1
    ]).reshape(4, 4)
    # mirror_transform = torch.as_tensor(mirror_transform, dtype=torch.float, device="cuda")
    # mirror_transform.requires_grad_()

    return mirror_transform



