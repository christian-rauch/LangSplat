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
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # views = [views[0]]

    # print("################################")
    # for v in views:
    #     print("view TR", v.T, v.R)
    #     print("view P", v.full_proj_transform)
    #     print("view V", v.world_view_transform)
    #     print("view cc", v.camera_center)
    # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")

    # for v in views:
    #     v.T = [1, 1, 1]
    #     v.camera_center = torch.from_numpy(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    #     v.world_view_transform = torch.from_numpy(np.eye(4, dtype=np.float32))
    #     v.full_proj_transform = torch.from_numpy(np.eye(4, dtype=np.float32))

    # fake camera views
    # T = [-2.619411333182333, 0.059014285207531554, -5.916798437340467]
    # R = [[0.9236408705691647, 0.1167078348753332, -0.36505728796032477],
    #      [-0.0651929082280865, 0.986469505922722, 0.15042539214289907],
    #      [0.37767370431498126, -0.1151398938825837, 0.9187520764089229]]

    # OK
    # T = [ 4.65786186, -0.43376951,  4.47095841]
    # R = [[ 0.92364087,  0.11670783, -0.36505729],
    #      [-0.06519291,  0.98646951,  0.15042539],
    #      [ 0.3776737 , -0.11513989,  0.91875208],]

    # T = [-2.97674486, -0.79132456,  1.98750789]
    # R =  [[ 0.12041174,  0.05589921, -0.99114898],
    #         [ 0.05876815 , 0.996261  ,  0.06332709,],
    #         [ 0.99098301, -0.06587332,  0.11667643],]

    # T = [ 4.65786186, -0.43376951,  5.47095841]
    # R = [[ 0.92364087,  0.11670783, -0.36505729],
    #      [-0.06519291,  0.98646951,  0.15042539],
    #      [ 0.3776737 , -0.11513989,  0.91875208],]

    # T = [ 1.65786186, -0.43376951,  7.47095841]
    # R = [[ 0.92364087,  0.11670783, -0.36505729],
    #      [-0.06519291,  0.98646951,  0.15042539],
    #      [ 0.3776737 , -0.11513989,  0.91875208],]

    # frame 155
    # T = [1.584053018600151, -0.13880305753090616, 0.19339471116605858]
    # T = [1.584053018600151, -0.13880305753090616, 10.19339471116605858]
    T = [1.584053018600151, -0.13880305753090616, 15.19339471116605858]
    R = [[0.7369558120174383, -0.2088297631842908, 0.642873440999168],
         [0.2707562460948057, 0.962645135879144, 0.002323266853641984],
         [-0.6193441582305972, 0.1723498545883264, 0.7659688905490978],]

    target_size = [600, 1000]

    views2 = []
    for v in views:
        v2 = Camera(
            colmap_id=0,
            R=np.array(R, dtype=np.float32),
            T=torch.from_numpy(np.array(T, dtype=np.float32)),
            FoVx=np.deg2rad(60),
            FoVy=np.deg2rad(60) * target_size[0] / target_size[1],
            image=v.original_image,
            gt_alpha_mask=v.original_image,
            image_name=v.image_name,
            uid=None,
        )
        v2.image_width = target_size[1]
        v2.image_height = target_size[0]
        views2.append(v2)

    views = views2

    # print("#### AFTER")
    # for v in views:
    #     print("view TR", v.T, v.R)
    #     print("view P", v.full_proj_transform)
    #     print("view V", v.world_view_transform)
    #     print("view cc", v.camera_center)
    # print("YYYYY AFTER")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        rendering_colour = output["render"]
        rendering_feat = output["language_feature_image"]

        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + "_overview.npy"),rendering_feat.permute(1,2,0).cpu().numpy())
        # np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + "_overview.npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering_feat, os.path.join(render_path, '{0:05d}'.format(idx) + "_overview.png"))
        torchvision.utils.save_image(rendering_colour, os.path.join(render_path, '{0:05d}'.format(idx) + "_rgb_overview.png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + "_overview.png"))

        break

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train_overview", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test_overview", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.include_feature = True

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)