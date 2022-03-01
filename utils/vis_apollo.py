import pwd
from unicodedata import category
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image
from pyparsing import col, line
from tqdm import tqdm
from collections import OrderedDict
import subprocess
from smoke.modeling.smoke_coder import encode_label
from correspondece_constraint import *
import math
from math import sin, cos
import pickle
import torch
from pycocotools import mask
import csv
import imageio
import shutil


def write_video(image_path, save_dir, save_name='out.mp4'):
    writer = imageio.get_writer(os.path.join(save_dir, save_name), fps=int(args.frame_rate))

    for image in sorted(os.listdir(image_path)):
        if image.endswith(('.jpg', '.png')):
            img = imageio.imread(os.path.join(image_path, image))
            writer.append_data(img)
    writer.close()

def get_xzgrid(zx_dim=(128, 101), zrange=30.0):
    """
    BEV grids when transfer RF images to cart coordinates
    :param zx_dim: dimension of output BEV image
    :param zrange: largest range value in z axis
    """
    origin = np.array([0, int(zx_dim[1] / 2)])
    zline, zreso = np.linspace(0, zrange, num=zx_dim[0], endpoint=False, retstep=True)
    xmax = zreso * (origin[1] + 1)
    xline = np.linspace(0, xmax, num=origin[1] + 1, endpoint=False)
    xline = np.concatenate([np.flip(-xline[1:]), xline])
    return xline, zline


def xz2idx_interpolate(x, z, x_grid, z_grid):
    """get interpolated XZ indices in float"""
    xids = np.arange(x_grid.shape[0])
    zids = np.arange(z_grid.shape[0])
    x_id = np.interp(x, x_grid, xids)
    z_id = np.interp(z, z_grid, zids)
    return x_id, z_id


def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale * scale_xyz[0]
    y = npline[11] * scale * scale_xyz[1]
    z = npline[12] * scale * scale_xyz[2]
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2


    x_corners += -l / 2
    z_corners += -w / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:])) # (9, 2): includes the first corner at the end



def draw_birdeyes(ax2, line_gt, line_p, shape, scale):
    # shape = 900

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

    if line_gt is not None:
        gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)
        codes = [Path.LINETO] * gt_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(gt_corners_2d, codes)
        p = patches.PathPatch(pth, fill=False, color='orange', label='ground truth')
        ax2.add_patch(p)


def draw_on_chirp(ax3, line_p, x_grid, z_grid, color):
    # shape = 900
    obj = detectionInfo(line_p)

    R = np.array([[np.cos(obj.rot_global), np.sin(obj.rot_global)],
                  [-np.sin(obj.rot_global), np.cos(obj.rot_global)]])
    t = np.array([obj.tx * scale_xyz[0], obj.tz * scale_xyz[2]]).reshape(1, 2).T

    x_corners = [0, obj.l, obj.l, 0]  # -l/2
    z_corners = [obj.w, obj.w, 0, 0]  # -w/2

    x_corners += -np.float64(obj.l) / 2
    z_corners += -np.float64(obj.w) / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = corners_2D + t

    x_coors, z_coors = xz2idx_interpolate(corners_2D[0, :], corners_2D[1, :], x_grid, z_grid)
    corners_2D = np.concatenate((x_coors[:, None], z_coors[:, None]), axis=1).astype(np.int16)
    corners_2D = np.vstack((corners_2D, corners_2D[0, :]))

    codes = [Path.LINETO] * corners_2D.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(corners_2D, codes)
    p = patches.PathPatch(pth, fill=False, color=color, label='auxiliary')
    ax3.add_patch(p)

    # draw center of cars
    x = np.float64(obj.tx) * scale_xyz[0]
    z = np.float64(obj.tz) * scale_xyz[2]
    x_centers, z_centers = xz2idx_interpolate(x, z, x_grid, z_grid)
    ax3.scatter(x=x_centers, y=z_centers, c=color, alpha=.5)

# gt_bb, pred_bb: (4, ), (4, )
def GIOU(gt_bb, pred_bb):
    x_g1, x_g2, y_g1, y_g2 = gt_bb[0], gt_bb[2], gt_bb[1], gt_bb[3]
    x_p1, x_p2, y_p1, y_p2 = pred_bb[0], pred_bb[2], pred_bb[1], pred_bb[3]

    A_g = (x_g2 - x_g1) * (y_g2 - y_g1)
    A_p = (x_p2 - x_p1) * (y_p2 - y_p1)
    x_I1, x_I2, y_I1, y_I2 = np.maximum(x_p1, x_g1), np.minimum(x_p2, x_g2), np.maximum(y_p1, y_g1), np.minimum(y_p2, y_g2) # intersection
    A_I = np.clip((x_I2 - x_I1) * (y_I2 - y_I1), 0, None) 
    A_U = A_g + A_p - A_I

    # area of the smallest enclosing box
    min_box = np.minimum(gt_bb, pred_bb)
    max_box = np.maximum(gt_bb, pred_bb)
    A_C = (max_box[2] - min_box[0]) * (max_box[3] - min_box[1])

    iou = A_I / A_U
    giou = iou - (A_C - A_U) / A_C
    return float(giou)

def draw_2Dbox(ax, P2, line, color, draw_gt, pitch):
    K = P2[:, :3]
    obj = detectionInfo(line)
    # todo: incoporate pitch to encode_label()
    _, pred_box, _ = encode_label(K, obj.rot_global, (obj.l, obj.h, obj.w), locs=(obj.tx * scale_xyz[0], obj.ty * scale_xyz[1], obj.tz * scale_xyz[2]), rx=pitch)
    xmin = int(np.clip(pred_box[0], 0., float(w_h[0])))
    ymin = int(np.clip(pred_box[1], 0., float(w_h[1])))
    xmax = int(np.clip(pred_box[2], 0., float(w_h[0])))
    ymax = int(np.clip(pred_box[3], 0., float(w_h[1])))
    width = xmax - xmin
    height = ymax - ymin
    box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color=color, linewidth='1')
    ax.add_patch(box_2d)

    if draw_gt:
        xmin_gt = int(float(line[16]))
        xmax_gt = int(float(line[18]))
        ymin_gt = int(float(line[17]))
        ymax_gt = int(float(line[19]))
        width = xmax_gt - xmin_gt
        height = ymax_gt - ymin_gt
        box_2d = patches.Rectangle((xmin_gt, ymin_gt), width, height, fill=False, color='r', linewidth='1')
        ax.add_patch(box_2d)
        # add GIOU text
        gt, pred = np.array([xmin_gt, ymin_gt, xmax_gt, ymax_gt]), np.array([xmin, ymin, xmax, ymax])
        giou = GIOU(gt, pred)
        bb_min = np.minimum(gt, pred)
        ax.text(int((xmin_gt + xmax_gt)/2), int(bb_min[1]) -5, f'GIoU:{giou:.2f}', color='lime', size='x-small') 
        
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(R, np.dot(P, Y))

def compute_3Dbox(P2, obj):
    R = euler_to_Rot(obj['yaw'], obj['pitch'], obj['roll'])

    x_corners = [0, obj['w'], obj['w'], obj['w'], obj['w'], 0, 0, 0]
    y_corners = [0, 0, obj['h'], obj['h'], 0, 0, obj['h'], obj['h']]
    z_corners = [0, 0, 0, obj['l'], obj['l'], obj['l'], obj['l'], 0] 

    x_corners = [i - obj['w'] / 2 for i in x_corners]
    y_corners = [i - obj['h'] / 2 for i in y_corners]
    z_corners = [i - obj['l'] / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D =  R.dot(corners_3D)
    corners_3D += np.array([obj['tx'] * scale_xyz[0], obj['ty'] * scale_xyz[1], obj['tz'] * scale_xyz[2]]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(ax, P2, obj, color):

    corners_2D = compute_3Dbox(P2, obj)
    # print(corners_2D)
    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    # put a mask on the front
    ax.plot([corners_2D[0, 5], corners_2D[0, 3]],[corners_2D[1, 5], corners_2D[1, 3]], color='r',lw=1)
    ax.plot([corners_2D[0, 4], corners_2D[0, 6]],[corners_2D[1, 4], corners_2D[1, 6]], color='r',lw=1)
    ax.add_patch(p)
    # ax.add_patch(front_fill)

# l13, l23(# of sampled points, 3(x, y1 or y2, y3))
def draw_sampled_points(ax, l13, l23):
    # todo draw sampled points
    l12_point_idx = (l13[:, 2] < 0).nonzero(as_tuple=True)[0]
    l12_line_idx = (l13[:, 2] >= 0).nonzero(as_tuple=True)[0]
    # draw sampled points on projected 3d bbox with no vertically matched hull mask
    ax.scatter(l13[l12_point_idx, 0], l13[l12_point_idx, 1], s=1, c='green')
    # draw line in matched pair
    for i in l12_line_idx:
        ax.plot([l13[i, 0], l13[i, 0]], [l13[i, 1], l13[i, 2]], '-go', linewidth=1, markersize=2)

    l13_point_idx = (l23[:, 2] < 0).nonzero(as_tuple=True)[0]
    l13_line_idx = (l23[:, 2] >= 0).nonzero(as_tuple=True)[0]

    # draw sampled points on projected 3d bbox with no vertically matched hull mask
    ax.scatter(l23[l13_point_idx, 0], l23[l13_point_idx, 1], s=1, c='red')
    # draw line in matched pair
    for i in l13_line_idx:
        ax.plot([l23[i, 0], l23[i, 0]], [l23[i, 1], l23[i, 2]], '-ro', linewidth=1, markersize=2)

def draw_hull_mask(line, im):
    encoded_rle = {
            'size': [float(line[20]), float(line[21])],
            'counts': line[22]
    }
    decoded_mask = mask.decode(encoded_rle) # (mask_h, mask_w) np array
    color = [0, 0, 255]
    mask_blue = decoded_mask[:, :, None].repeat(3, axis=2) * np.array(color, dtype=np.uint8).reshape((1, 1, 3))
    im =  im * ~decoded_mask.astype(bool)[:, :, None].repeat(3, axis=2) + mask_blue
    return im

def map_seq_to_lines(lines):
    seq_to_lines = {}
    for line in lines:
        line = line.split(' ')
        seq = int(line[0])
        seq = f'{seq:010}'
        new_line = ' '.join(line[1:])
        if not seq in seq_to_lines.keys():
            seq_to_lines[seq] = [new_line]
        else:
            seq_to_lines[seq].append(new_line)
    return seq_to_lines

def cam_matrix(capture_date):
    P2 = None
    if capture_date == '0929':
        P2 = [1189.964744, 0.000000, 735.409035, 0, 0.000000, 1189.824753, 518.136149, 0, 0.000000, 0.000000, 1.000000, 0]
    elif capture_date == '0529':
        P2 = [849.177455, 0.000000, 712.166787, 0, 0.000000, 854.207389, 543.445028, 0, 0.000000, 0.000000, 1.000000, 0]
    elif capture_date=='kitti':
        P2 = [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]
    elif capture_date=='apollo':
        P2 = [2304.5479, 0, 1686.2379, 0, 0, 2305.8757, 1354.9849, 0, 0, 0, 1, 0]
    return P2


def load_prediction(pred_file_path):
    predictions = []
    with open(pred_file_path, 'r') as file:
        for line_id, line in enumerate(file.readlines()):
            line = line.strip().split(' ')
            predictions.append({
                'type': line[0],
                'truncation': float(line[1]),
                'occlusion': float(line[2]),
                'box2d': [float(line[3]), float(line[4]), float(line[5]), float(line[6])],
                'h': float(line[7]),
                'l': float(line[8]),
                'w': float(line[9]),
                'tx': float(line[10]),
                'ty': float(line[11]),
                'tz': float(line[12]),
                'quat': [float(line[13]), float(line[14]), float(line[15]), float(line[16])],
                'pitch': float(line[17]),
                'yaw': float(line[18]),
                'roll': float(line[19]),
                'score': float(line[20])
            })
    return predictions

def visualize_single_file(image_root_dir, pred_file_path, capture_date, save_fig_path):
    # visualize 3D bounding box
    fig = plt.figure(figsize=(20.00, 20), dpi=100)
    ax = fig.add_subplot()
    P2 = cam_matrix(capture_date)
    P2 = np.array(P2, dtype=np.float32).reshape(3, 4) 
    frame_id = pred_file_path.split('/')[-1][:-4]
    image_path = os.path.join(image_root_dir, f'{frame_id}.jpg')
    image = Image.open(image_path).convert('RGB')
    # print(frame_id)
    for obj in load_prediction(pred_file_path):
        # print(obj['box2d'])
        draw_3Dbox(ax, P2, obj, 'green')
        # print('\n')

        


    ax.imshow(image)
    ax.set_xticks([]) #remove axis value
    ax.set_yticks([])
    fig.savefig(os.path.join(save_fig_path, f'{frame_id}_pred.jpg'), bbox_inches='tight', pad_inches=0)
    

def visualization(args, image_path, calib_path, pred_path, sampled_points_path,
                  dataset, VEHICLES, start_frame, end_frame, radar_dir, capture_date, pred_file_all=True, draw_3dbb=False, draw_2dbb=True, pred_file=None, draw_sp=False):

    if calib_path is None:
        P2 = cam_matrix(capture_date)
        P2 = np.array(P2, dtype=np.float32).reshape(3, 4)

    seq_to_lines, row_idx  = {}, 0
    # save_hull = False
# if pred_file_all:
    with open(pred_file) as f:
        seq_to_lines = map_seq_to_lines(f.readlines())
    for index in tqdm(range(start_frame, end_frame)):
        image_file = os.path.join(image_path, dataset[index]+ '.jpg')
        # label_file = os.path.join(label_path, dataset[index] + '.txt')
        # prediction_file = os.path.join(pred_path, dataset[index]+ '.txt')
        if calib_path is not None:
            calibration_file = os.path.join(calib_path, dataset[index] + '.txt')
            for line in open(calibration_file):
                if 'P2' in line:
                    P2 = line.split(' ')
                    P2 = np.asarray([float(i) for i in P2[1:]])
                    P2 = np.reshape(P2, (3, 4))

        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
        # fig2 = plt.figure(figsize=(12.8, 7.68))

        # fig.tight_layout()
        gs = GridSpec(1, 7)
        gs.update(wspace=1)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:5])
        ax3 = fig.add_subplot(gs[0, 5:])
        # ax4 = fig2.add_subplot()

        # draw chirp image
        chirp_cart = np.load(os.path.join(radar_dir, f'{dataset[index][-6:]}_0000.npy'))
        ax3.imshow(chirp_cart, vmin=0, vmax=1, origin='lower')
        xz_grid = get_xzgrid(zx_dim=chirp_cart.shape)
        ax3.set_xticks(np.arange(0, len(xz_grid[0]), 30), xz_grid[0][::30])
        ax3.set_yticks(np.arange(0, len(xz_grid[1]), 20), xz_grid[1][::20])
        ax3.set_xlabel('x(m)')
        ax3.set_ylabel('z(m)')
        ax3.grid(alpha=.2)

        # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
        image = Image.open(image_file).convert('RGB')
        # im = np.array(image)
        shape = 900
        scale = 15 # rotio of pixel / meter
        birdimage = np.zeros((shape, shape, 3), np.uint8)

        if pred_file_all:
            if dataset[index] in seq_to_lines.keys():
                for line_p in seq_to_lines[dataset[index]]:
                    line_p = line_p.strip().split(' ')
                    truncated = np.abs(float(line_p[1]))
                    trunc_level =  255

                    # truncated object in dataset is not observable
                    if line_p[0] in VEHICLES  and truncated < trunc_level:
                        color = 'green'
                        draw_gt = False
                        if len(line_p) == 23:
                            draw_gt = True
                        elif line_p[0] == 'Cyclist':
                            color = 'yellow'
                        elif line_p[0] == 'Pedestrian':
                            color = 'cyan'
                        if draw_3dbb:
                            draw_3Dbox(ax, P2, line_p[:20], color, 0)
                        draw_birdeyes(ax2, None, line_p[:20], shape, scale)
                        draw_on_chirp(ax3, line_p[:20], xz_grid[0], xz_grid[1], color)
                        if draw_2dbb:
                            draw_2Dbox(ax, P2, line_p[:20], color, draw_gt, 0)
                        # if draw_gt and draw_sp:
                        #     l123 = torch.tensor(row_to_l123[row_idx])
                        #     draw_3Dbox(ax4, P2, line_p[:20], color, pitch)
                        #     draw_sampled_points(ax4, l123[:10, :], l123[10:, :])
                        #     ax4.text(10, 10, s=f'Loss green: {loss_l13:.4f}, Loss red: {loss_l23:.4f}, Loss green+red: {loss_l13+loss_l23:.4f}, pitch: {pitch:.4f}', fontsize='x-large', c='blue')
                        #     im = draw_hull_mask(line_p, im)
                        #     save_hull = True
                    row_idx += 1
                    

        # visualize 3D bounding box
        ax.imshow(image)
        ax.set_xticks([]) #remove axis value
        ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')
        
        # visualize bird eye view
        tick_count = 10
        tick_step = int(shape/tick_count)
        ax2.imshow(birdimage, origin='lower')
        ax2.set_xticks([ i for i in range(tick_step, shape, tick_step)])
        ax2.set_yticks([ i for i in range(tick_step, shape, tick_step)])
        ax2.set_xticklabels([ round((i - shape/2) / scale, 1) for i in range(tick_step, shape, tick_step)])
        ax2.set_yticklabels([ round(i / scale, 1) for i in range(tick_step, shape, tick_step)])
        ax2.grid(alpha=0.2)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('z (m)')

        # draw sampled points during pitch optimization
        # if save_hull:
        #     ax4.imshow(im, interpolation=None)
        #     ax4.axis('off')

        # add legend
        legend_elements = []
        class_colors = ['green', 'yellow', 'cyan']
        labels = ['Car', 'Cyclist', 'Pedestrian']
        for i in range(0,len(class_colors)):
            legend_elements.append( Line2D([0], [0], marker='o', color=class_colors[i], label=labels[i],
                          markerfacecolor=class_colors[i], markersize=3, ls='', alpha=0.5))
        ax3.legend(handles=legend_elements, loc='upper right',
                                fontsize='x-small', framealpha=0.2)


        handles, labels = ax2.get_legend_handles_labels()
        if len(handles) > 0:
            legend = ax2.legend([handles[0]], [labels[0]], loc='lower right',
                                fontsize='x-small', framealpha=0.2)
        for text in legend.get_texts():
            plt.setp(text, color='w')

        if args.save == False:
            plt.show()
        else:
            fig.savefig(os.path.join(args.path, dataset[index]), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
            # if save_hull:
            #     file_name = sampled_points_path.split('.')[0]
            #     fig2.savefig(os.path.join(args.path, f'{file_name}_{dataset[index]}.png'), dpi=fig2.dpi, bbox_inches='tight') # {dataset[index]}_hull_sampled
            #     save_hull = False
        
        plt.close(fig)
        # plt.close(fig2)
        # video_writer.write(np.uint8(fig))

def main(args):
    cruw_seq_train_root = '/mnt/disk1/CRUW/ROD2021/sequences/train'
    radar_dir = 'RADAR_XZ_H_MAG'
    image_path = f'/home/andy/SMOKE/datasets/CRUW/train/{seq_name}/IMAGES_0' # f'/home/andy/SMOKE/datasets/CRUW/train/{date}/IMAGES_0'
    calib_path = None
    pred_path = '/home/andy/SMOKE/tools/logs/inference/CRUW4/data'# change here
    pred_file = f'/home/andy/bb_optimize/data/cruw/{seq_name}_filtered_masked.txt' # '/home/andy/bb_optimize/data/cruw/single_car.txt'
    # sampled_points_path = '/home/andy/bb_optimize/sampled_points_500_Epoch1_02082022_01:14:48.pickle'
    
    dataset = [name.split('.')[0] for name in sorted(os.listdir(image_path))]

    VEHICLES = ['Car', 'Cyclist', 'Pedestrian']
    start_frame = 0
    end_frame = len(os.listdir(image_path))
    # end_frame = 
    cam_type = 'apollo' # change here

    # for sampled_points_path in os.listdir(sampled_points_dir):
    visualization(args, image_path, calib_path, pred_path, None,
                dataset, VEHICLES, start_frame, end_frame, os.path.join(cruw_seq_train_root, seq_name, radar_dir), cam_type, pred_file_all=True, pred_file=pred_file, draw_3dbb=True, draw_2dbb=False)

if __name__ == '__main__':
    w_h = (2710, 3384)
    scale_xyz =  (1., 1., 1.) # (1.35, 1, 1.35), (1.3686, 1.1231, 1.3977), (1.3534, 1.1443, 1.3651)
    seq_name = '2019_09_29_ONRD001' # change here
    dataset = 'CRUW/Train'
    parser = argparse.ArgumentParser(description='Visualize 3D bounding box on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--save', type=bool, default=True, help='Save Figure or not')
    parser.add_argument('-p', '--path', type=str, default=f'./results_with_quat/{dataset}/{seq_name}_scale_{scale_xyz}', help='Output Image folder') #change here # ./results/CRUW/train/{date}_scale_{scale_xyz}__pitch_{pitch}
    parser.add_argument('-fr', '--frame_rate', type=str, default='10', help='frame rate of output video')
    parser.add_argument('-sv', '--save_video', type=bool, default=True, help='whether to save video')
    parser.add_argument('-sf', '--seperate_frames', type=bool, default=True, help='visualize frames from seperate prediction files')


    args = parser.parse_args()

 
    if args.seperate_frames:
        # pred_root = '/home/andy/SMOKE/tools/logs_apollo_new2/inference/CRUW_train_2019_09_29_ONRD001/data' #'/home/andy/SMOKE/tools/logs_apollo_new/inference/apollocar3d/data'
        # save_dir = f'/home/andy/3d-bounding-box-estimation-for-autonomous-driving/results/{seq_name}' #{seq_name}
        pred_root = '/home/andy/SMOKE/tools/logs_apollo_new/inference/apollocar3d/data' #'/home/andy/SMOKE/tools/logs_apollo_new/inference/apollocar3d/data'
        save_dir = f'/home/andy/3d-bounding-box-estimation-for-autonomous-driving/results/apollo_dis'
        count = 100
        predfiles = sorted(os.listdir(pred_root))
        random_idx = np.random.choice(np.arange(len(predfiles)), size=count, replace=False).tolist()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        # for predfile in [predfiles[i] for i in random_idx]:
        #     count -= 1
        #     visualize_single_file('/home/andy/LOCNet_Apollo_Unofficial-main/data/Kaggle/pku-autonomous-driving/train_images', os.path.join(pred_root, predfile), 'apollo', save_dir)
        #     if count < 1:
        #         break
        # for predfile in predfiles:
        #     # count -= 1
        #     visualize_single_file(f'/home/andy/SMOKE/datasets/CRUW/train/{seq_name}/IMAGES_0', os.path.join(pred_root, predfile), '0929', save_dir)
        #     # if count < 1:
        #     #     break
        if args.save_video:
            # os.chdir(save_dir)
            # subprocess.run(['ffmpeg', '-framerate', args.frame_rate, '-i', *os.listdir(save_dir) , '-y', f'out.mp4'])
            # write_video(save_dir, save_dir, f'{seq_name}_unscaled.mov')
            write_video(save_dir, save_dir, f'apollo_dis.mov')
    else:
        if not os.path.exists(args.path):
                os.makedirs(args.path)

        main(args)

        if args.save_video:
            # write_video(save_dir, save_dir, f'apollo_dis.mov')
            subprocess.run(['ffmpeg', '-framerate', args.frame_rate, '-i', '%010d.png' , '-y', f'out.mp4']) # -y overwrite #f'{date}_scale_{scale_xyz}_pitch_{global_pitch}_3dbb.mp4', '%03d9_0000000571.png'
