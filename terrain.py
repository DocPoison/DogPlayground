# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

import random

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(0.2, 0.9)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                difficulty += np.random.uniform(-0.1, 0.1)
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                # terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.length_per_env_pixels,
                              length=self.width_per_env_pixels, # 令人感叹
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1):
        terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels, # 令人感叹
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty

        pit_width = 0.6 * difficulty

        # if choice < self.proportions[0]:
        #     if choice < self.proportions[0]/ 2:
        #         slope *= -1
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # elif choice < self.proportions[1]:
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        # elif choice < self.proportions[3]:
        #     if choice<self.proportions[2]:
        #         step_height *= -1
        #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        # elif choice < self.proportions[4]:
        #     num_rectangles = 20
        #     rectangle_min_size = 1.
        #     rectangle_max_size = 2.
        #     terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        # elif choice < self.proportions[5]:
        #     terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        # elif choice < self.proportions[6]:
        #     gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        # else:
        #     pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        if choice < self.proportions[0]:
            parallel_pit(terrain, pit_width)
            # self.add_roughness(terrain)
        elif choice < self.proportions[1]:
            parallel_pit(terrain, pit_width) # elliptical_pit(terrain,pit_width)
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            parallel_pit(terrain, pit_width)# semicircle_pit(terrain, pit_width)
            # self.add_roughness(terrain)
        elif choice < self.proportions[3]:
            parallel_pit(terrain, pit_width)
            # self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            flat_ground(terrain, pit_width)
            # self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        # y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        # y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

# 地上有一个坑，各种（5种）类型的坑

# def parallel_pit(terrain, pit_width):
#     """
#     在地形上生成三个平行坑（矩形截面），深度 4m。
#     坑的排布：前 10m 无坑，之后每隔 15m 一个坑，共 3 个。
#     同时，在平地（坑外区域）上添加随机 terrain。
#     """
#     depth_m = 4.0  # 坑深 4m
#     depth_pix = int(depth_m / terrain.vertical_scale)  # 转换为像素高度
    
#     # 计算 pit_width 对应的像素长度（虽然本函数中没有直接使用 pit_width_pix，但保留此变量以便参考）
#     pit_width_pix = int(pit_width / terrain.horizontal_scale)
    
#     # 先在整个地形上添加随机的高度变化，生成随机平地
#     terrain_utils.random_uniform_terrain(terrain, 
#                                            min_height=-0.05, 
#                                            max_height=0.05, 
#                                            step=0.005, 
#                                            downsampled_scale=0.2)
    
#     # 生成 3 个坑
#     for i in range(3):
#         # 每个坑的 x 范围（单位：米）
#         pit_start_m = 10.0 + i * 5.0
#         pit_end_m   = pit_start_m + pit_width
        
#         # 转换为像素坐标（x 方向）
#         pit_start_x = int(pit_start_m / terrain.horizontal_scale)
#         pit_end_x   = int(pit_end_m   / terrain.horizontal_scale)
        
#         # 在 [pit_start_x, pit_end_x] × [0, terrain.width-1] 范围内挖坑
#         # 注意数组切片上限是非包含，所以可以直接赋值坑区域
#         terrain.height_field_raw[pit_start_x:pit_end_x, 0:terrain.width] = -depth_pix
        
#         # # 记录边沿检测点（此处仅记录 x 坐标范围，实际可根据需要增加更多信息）
#         # terrain.edgecheck.append({
#         #     "type": "parallel",
#         #     "pit_index": i,
#         #     "start_x": pit_start_x,
#         #     "end_x": pit_end_x,
#         #     "depth_pix": depth_pix
#         # })
def parallel_pit(terrain, pit_width):
    """
    在地形上生成三个平行坑（矩形截面），深度 4m。
    坑的排布：前 10m 无坑，之后每隔 5m 一个坑，共 3 个。
    同时，在平地（坑外区域）上添加随机 terrain，以及在该地块上边和下边添加墙体：
      - 墙宽 0.05m
      - 墙高 1m
    """
    depth_m = 4.0  # 坑深 4m
    depth_pix = int(depth_m / terrain.vertical_scale)  # 转换为像素高度
    
    # 计算 pit_width 对应的像素长度（供参考）
    pit_width_pix = int(pit_width / terrain.horizontal_scale)
    
    # 先在整个地形上添加随机高度变化，生成随机平地
    terrain_utils.random_uniform_terrain(terrain, 
                                           min_height=-0.05, 
                                           max_height=0.05, 
                                           step=0.005, 
                                           downsampled_scale=0.2)
    
    # 生成 3 个坑
    for i in range(3):
        # 每个坑的 x 范围（单位：米）
        pit_start_m = 10.0 + i * 5.0
        pit_end_m   = pit_start_m + pit_width
        
        # 转换为像素坐标（x 方向）
        pit_start_x = int(pit_start_m / terrain.horizontal_scale)
        pit_end_x   = int(pit_end_m   / terrain.horizontal_scale)
        
        # 在 [pit_start_x, pit_end_x] × [0, terrain.width-1] 范围内挖坑，坑深为 -depth_pix
        terrain.height_field_raw[pit_start_x:pit_end_x, 0:terrain.width] = -depth_pix
        
        # # 记录边沿检测点（这里只记录了 x 方向起止范围，实际可根据需要扩展更多信息）
        # terrain.edgecheck.append({
        #     "type": "parallel",
        #     "pit_index": i,
        #     "start_x": pit_start_x,
        #     "end_x": pit_end_x,
        #     "depth_pix": depth_pix
        # })
    
    # 添加上边和下边的墙
    wall_width_pix = int(0.1 / terrain.horizontal_scale)  # 墙宽转换为像素
    wall_height_pix = int(1.0 / terrain.vertical_scale)      # 墙高转换为像素

    # 上边墙：在所有 x 行，y 方向前 wall_width_pix 列设置为 wall_height_pix
    terrain.height_field_raw[:, 0:wall_width_pix] = wall_height_pix
    # 下边墙：在所有 x 行，y 方向后 wall_width_pix 列设置为 wall_height_pix
    terrain.height_field_raw[:, terrain.width - wall_width_pix:terrain.width] = wall_height_pix


# def parallel_pit(terrain, pit_width):
#     """
#     在地形上生成三个平行坑（矩形截面），深度 4m。
#     坑的排布：前 10m 无坑，之后每隔 15m 一个坑，共 3 个。
#     """
#     depth_m = 4.0  # 前三种坑深度固定为 4m
#     depth_pix = int(depth_m / terrain.vertical_scale)  # 转为像素高度
    
#     # 计算 pit_width 对应的像素长度
#     pit_width_pix = int(pit_width / terrain.horizontal_scale)
    
#     # 生成 3 个坑
#     for i in range(3):
#         # 每个坑的 x 范围（米）
#         pit_start_m = 10.0 + i * 15.0
#         pit_end_m   = pit_start_m + pit_width
        
#         # 转为像素坐标
#         pit_start_x = int(pit_start_m / terrain.horizontal_scale)
#         pit_end_x   = int(pit_end_m   / terrain.horizontal_scale)
        
#         # 在 [pit_start_x, pit_end_x] × [0, terrain.width-1] 范围内挖坑
#         # 注意：数组切片上限是非包含，所以用 pit_end_x+1 更稳妥
#         terrain.height_field_raw[pit_start_x:pit_end_x, 0:terrain.width] = -depth_pix
        
#         # 记录边沿检测点 (示例仅记录 x 坐标)
#         # terrain.edgecheck.append({
#         #     "type": "parallel",
#         #     "pit_index": i,
#         #     "start_x": pit_start_x,
#         #     "end_x": pit_end_x,
#         #     "depth_pix": depth_pix
#         # })

def elliptical_pit(terrain, pit_width):
    """
    在地形上生成三个椭圆坑，深度 4m。
    长轴方向横穿整个宽度，短轴方向为 pit_width。
    """
    depth_m = 4.0
    depth_pix = int(depth_m / terrain.vertical_scale)
    pit_width_pix = int(pit_width / terrain.horizontal_scale)
    
    center_y = terrain.width // 2
    # 椭圆长轴半径 (major radius) 和短轴半径 (minor radius)
    a = terrain.width / 2.0   # y 方向半径
    # x 方向半径先按 pit_width_pix / 2
    # 注意要 float 类型参与计算
    b = pit_width_pix / 2.0   
    
    for i in range(3):
        pit_start_m = 10.0 + i * 15.0
        pit_end_m   = pit_start_m + pit_width
        pit_start_x = int(pit_start_m / terrain.horizontal_scale)
        pit_end_x   = int(pit_end_m   / terrain.horizontal_scale)
        center_x    = (pit_start_x + pit_end_x) / 2.0
        
        # 遍历 [pit_start_x, pit_end_x], [0, terrain.width]，判断是否在椭圆内
        for x in range(pit_start_x, pit_end_x):
            for y in range(terrain.width):
                # 椭圆方程: ((x - cx)^2 / b^2) + ((y - cy)^2 / a^2) <= 1
                dx = x - center_x
                dy = y - center_y
                if (dx*dx)/(b*b) + (dy*dy)/(a*a) <= 1.0:
                    terrain.height_field_raw[x, y] = -depth_pix
        
        # 记录边沿
        # terrain.edgecheck.append({
        #     "type": "elliptical",
        #     "pit_index": i,
        #     "start_x": pit_start_x,
        #     "end_x": pit_end_x,
        #     "depth_pix": depth_pix
        # })

def semicircle_pit(terrain, pit_width):
    """
    在地形上生成三个半圆形坑，深度 4m。
    假设半圆在 y 方向居中，直径 = 0.7 * pit_width。
    x 方向坑宽度 = pit_width。
    """
    depth_m = 4.0
    depth_pix = int(depth_m / terrain.vertical_scale)
    pit_width_pix = int(pit_width / terrain.horizontal_scale)
    
    # 半圆直径的像素
    diameter_pix = 0.7 * pit_width_pix
    radius = diameter_pix / 2.0
    
    center_y = terrain.width // 2  # 半圆中心在 y 方向
    
    for i in range(3):
        pit_start_m = 10.0 + i * 15.0
        pit_end_m   = pit_start_m + pit_width
        pit_start_x = int(pit_start_m / terrain.horizontal_scale)
        pit_end_x   = int(pit_end_m   / terrain.horizontal_scale)
        
        # 半圆中心 x 坐标（前后方向）
        # 注意：这里为了简化，让半圆只影响 y 方向的分布，x 方向范围还是矩形区
        # 也可以把 x 方向也变成某种弧线，但这里演示简单做法
        for x in range(pit_start_x, pit_end_x):
            for y in range(terrain.width):
                dy = y - center_y
                # 半圆方程(只保留下半圆或上半圆)，此处示例保留"上半圆"
                # (dy)^2 <= radius^2  => |dy| <= radius
                # 我们可以让 [center_y - radius, center_y + radius] 之间挖坑
                if abs(dy) <= radius:
                    # 让挖坑深度与半圆形状一致（如果想做更平滑的弧度）
                    # 这里只是简单地把半圆范围内都挖到同一深度
                    terrain.height_field_raw[x, y] = -depth_pix
        
        # terrain.edgecheck.append({
        #     "type": "semicircle",
        #     "pit_index": i,
        #     "start_x": pit_start_x,
        #     "end_x": pit_end_x,
        #     "depth_pix": depth_pix
        # })

def slope_pit(terrain, pit_width):
    """
    在地形上生成三个有斜坡的坑，深度 1m。
    坑口(顶部)宽度 = pit_width, 坑底宽度 = 0.4 * pit_width。
    通过在 y 方向做线性插值，构造一个梯形截面。
    """
    depth_m = 1.0
    depth_pix = int(depth_m / terrain.vertical_scale)
    
    top_width_pix = int(pit_width / terrain.horizontal_scale)
    bottom_width_pix = int(0.4 * pit_width / terrain.horizontal_scale)
    
    center_y = terrain.width // 2
    
    # 两侧各留一点余量，以便在中间挖坑
    # 顶部区间 [y_top_left, y_top_right]
    y_top_left  = center_y - top_width_pix // 2
    y_top_right = center_y + top_width_pix // 2
    
    # 底部区间 [y_bottom_left, y_bottom_right]
    # 注意底部更窄
    y_bottom_left  = center_y - bottom_width_pix // 2
    y_bottom_right = center_y + bottom_width_pix // 2
    
    for i in range(3):
        pit_start_m = 10.0 + i * 15.0
        pit_end_m   = pit_start_m + pit_width
        pit_start_x = int(pit_start_m / terrain.horizontal_scale)
        pit_end_x   = int(pit_end_m   / terrain.horizontal_scale)
        
        # 沿 x 方向，从坑口到坑底线性收窄
        for x in range(pit_start_x, pit_end_x):
            alpha = (x - pit_start_x) / float(pit_end_x - pit_start_x)  # 0~1
            # 当前截面的左右 y 边界
            curr_left  = int((1 - alpha) * y_top_left  + alpha * y_bottom_left)
            curr_right = int((1 - alpha) * y_top_right + alpha * y_bottom_right)
            
            # 挖坑
            terrain.height_field_raw[x, curr_left:curr_right] = -depth_pix
        
        # terrain.edgecheck.append({
        #     "type": "slope",
        #     "pit_index": i,
        #     "start_x": pit_start_x,
        #     "end_x": pit_end_x,
        #     "depth_pix": depth_pix,
        #     "top_width_pix": top_width_pix,
        #     "bottom_width_pix": bottom_width_pix
        # })

# def flat_ground(terrain, pit_width):
#     """
#     不生成任何坑，也不记录边沿信息。
#     """
#     # 这里什么也不做
#     # terrain.edgecheck = []  # 清空或保持为空
def flat_ground(terrain, pit_width):
    """
    不生成任何坑，也不记录边沿信息，同时在平地上添加上边和下边墙：
      - 墙宽 0.05m
      - 墙高 1m
    """
    # 如果需要平地保持完全平整，可根据需求保持现有高度值不变，
    # 或者可以调用 terrain_utils.random_uniform_terrain 添加随机扰动（此处保持平整）
    
    # 添加上边和下边的墙
    wall_width_pix = int(0.1 / terrain.horizontal_scale)  # 墙宽转换为像素
    wall_height_pix = int(1.0 / terrain.vertical_scale)      # 墙高转换为像素

    # 注意：这里假设 terrain.width 表示 height_field_raw 的列数
    # 上边墙：在所有行，y 方向前 wall_width_pix 列设置为 wall_height_pix
    terrain.height_field_raw[:, 0:wall_width_pix] = wall_height_pix
    # 下边墙：在所有行，y 方向后 wall_width_pix 列设置为 wall_height_pix
    terrain.height_field_raw[:, terrain.width - wall_width_pix:terrain.width] = wall_height_pix
