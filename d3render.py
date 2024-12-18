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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from origin_gaussian_renderer import render
import torchvision

from scene.mirror_transform_model import MirrorTransformModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
# from gaussian_renderer import GaussianModel
from origin_gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipe, bg_color, train_test_exp, mirror_transform,separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    for idx, viewpoint_camera in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        render_pkg = render(viewpoint_camera, gaussians, pipe, bg_color)
        mirror_render_pkg = render(viewpoint_camera, gaussians, pipe, bg_color,
                                             mirror_transform=mirror_transform)

        mirror_image = mirror_render_pkg["render"]
        image = render_pkg["render"]
        gt_mirror_mask = viewpoint_camera.gt_alpha_mask.repeat(3, 1, 1)
        super_image = image * (1 - gt_mirror_mask) + mirror_image * gt_mirror_mask
        gt = viewpoint_camera.original_image[0:3, :, :]

        # if args.train_test_exp:
        #     rendering = rendering[..., rendering.shape[-1] // 2:]
        #     gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(super_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,  opt,separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        checkpoints_path = os.path.join(scene.model_path,"chkpnt"+str(scene.loaded_iter)+'.pth')
        (model_params,mirror_transform, scene_mask, first_iter) = torch.load(checkpoints_path)
        scene.generate_mirror_camera_transform(mirror_transform)
        gaussians.restore(model_params, opt)
        gaussians.scene_point_mask = scene_mask
        gaussians.mirror_transform_model = MirrorTransformModel(0)
        gaussians.mirror_transform_model.pre_mirror_transform = mirror_transform

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 0, mirror_transform,separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, 0, mirror_transform,separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,op,SPARSE_ADAM_AVAILABLE)