""" Video super-resolution with Cog"""
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import tempfile

from cog import BasePredictor, Input, Path

# add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
sys.path.append(os.path.join(os.path.dirname(__file__), "/models/modules/DCNv2"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import glob
import logging
import os
import os.path as osp
import re
import sys
from shutil import rmtree

import cv2
import numpy as np
import torch

import data.util as data_util
import models.modules.Sakuya_arch as Sakuya_arch
import utils.util as util


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Installing DCNv2 module......")
        import subprocess
        subprocess.run(
            ["python", "setup.py", "install"], cwd="codes/models/modules/DCNv2/"
        )

        print("Loading model checkpoint......")
        model_path = "./experiments/pretrained_models/xiang2020zooming.pth"
        self.model = Sakuya_arch.LunaTokis(64, 3, 8, 5, 40)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(
        self,
        video: Path = Input(description="Input video"),
        fps: int = Input(
            description="Specify fps of output video. Default: 24.", default=24
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        ffmpeg_dir = ""
        output_path = "output.mp4"  # output video name
        video = str(video)
        fps = int(fps)

        scale = 4
        N_ot = 3
        N_in = 1 + N_ot // 2

        # extract the input video to temporary folder
        save_folder = osp.join(osp.dirname(output_path), ".delme")
        save_out_folder = osp.join(osp.dirname(output_path), ".hr_delme")
        if os.path.isdir(save_folder):
            rmtree(save_folder)
        util.mkdirs(save_folder)
        if os.path.isdir(save_out_folder):
            rmtree(save_out_folder)
        util.mkdirs(save_out_folder)

        print("Extracting frames to images......")
        error = util.extract_frames(ffmpeg_dir, video, save_folder)
        if error:
            print(error)
            exit(1)

        # temporal padding mode
        padding = "replicate"
        save_imgs = True

        ############################################################################

        def single_forward(model, imgs_in):
            with torch.no_grad():
                # print(imgs_in.size()) # [1,5,3,270,480]
                b, n, c, h, w = imgs_in.size()
                h_n = int(4 * np.ceil(h / 4))
                w_n = int(4 * np.ceil(w / 4))
                imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
                imgs_temp[:, :, :, 0:h, 0:w] = imgs_in
                model_output = model.cuda()(imgs_temp)
                model_output = model_output[:, :, :, 0 : scale * h, 0 : scale * w]
                if isinstance(model_output, list) or isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
            return output

        # zsm images
        img_path_l = glob.glob(save_folder + "/*")
        img_path_l.sort(
            key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group())
        )
        select_idx_list = util.test_index_generation(False, N_ot, len(img_path_l))

        print("Performing model inference.......")
        for select_idxs in select_idx_list:
            # get input images
            select_idx = select_idxs[0]
            imgs_in = (
                util.read_seq_imgs_by_list([img_path_l[x] for x in select_idx])
                .unsqueeze(0)
                .to(self.device)
            )
            output = single_forward(self.model, imgs_in)
            outputs = output.data.float().cpu().squeeze(0)
            # save imgs
            out_idx = select_idxs[1]
            for idx, name_idx in enumerate(out_idx):
                output_f = outputs[idx, ...].squeeze(0)
                if save_imgs:
                    output = util.tensor2img(output_f)
                    cv2.imwrite(
                        osp.join(save_out_folder, "{:06d}.png".format(name_idx)), output
                    )

        print("Saving frames to video......")
        output_path = Path(tempfile.mkdtemp()) / output_path
        combine_frames(save_out_folder, str(output_path), fps)

        # remove tmp folder
        rmtree(save_folder)
        rmtree(save_out_folder)

        print(f"Saved output video to {str(output_path)}")
        return output_path


# combine frames to a video
def combine_frames(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(
        pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    for i in range(len(files)):
        filename = os.path.join(pathIn, files[i])
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()