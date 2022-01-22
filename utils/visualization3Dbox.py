import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import subprocess

from correspondece_constraint import *

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


def draw_box(ax2, shape=900):
    # shape = 900
    scale = 15
    offset = np.array([[50, 100]])
    pred_corners_2d = np.array([[0, 0], [0, 100], [100, 100], [100, 0], [0, 0] ], dtype=np.int16) + offset
    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='yellow', label='auxiliary')
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



def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx * scale_xyz[0], obj.ty * scale_xyz[1], obj.tz * scale_xyz[2]]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(ax, P2, line, color):

    corners_2D = compute_3Dbox(P2, line)

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

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)



def visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES, start_frame, end_frame, radar_dir, capture_date):

    if calib_path is None:
        if capture_date == '0929':
            P2 = [1189.964744, 0.000000, 735.409035, 0, 0.000000, 1189.824753, 518.136149, 0, 0.000000, 0.000000, 1.000000, 0]
        elif capture_date == '0529':
            P2 = [849.177455, 0.000000, 712.166787, 0, 0.000000, 854.207389, 543.445028, 0, 0.000000, 0.000000, 1.000000, 0]
        elif 'kitti':
            P2 = [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]

        P2 = np.array(P2, dtype=np.float32).reshape(3, 4)

    radar_mags = sorted(os.listdir(radar_dir))
    assert len(radar_mags) == len(dataset)
    for index in tqdm(range(start_frame, end_frame)):
        image_file = os.path.join(image_path, dataset[index]+ '.jpg')
        # label_file = os.path.join(label_path, dataset[index] + '.txt')
        prediction_file = os.path.join(pred_path, dataset[index]+ '.txt')
        if calib_path is not None:
            calibration_file = os.path.join(calib_path, dataset[index] + '.txt')
            for line in open(calibration_file):
                if 'P2' in line:
                    P2 = line.split(' ')
                    P2 = np.asarray([float(i) for i in P2[1:]])
                    P2 = np.reshape(P2, (3, 4))

        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

        # fig.tight_layout()
        gs = GridSpec(1, 7)
        gs.update(wspace=1)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:5])
        ax3 = fig.add_subplot(gs[0, 5:])

        # draw chirp image
        chirp_cart = np.load(os.path.join(radar_dir, radar_mags[index]))
        ax3.imshow(chirp_cart, vmin=0, vmax=1, origin='lower')
        xz_grid = get_xzgrid(zx_dim=chirp_cart.shape)
        ax3.set_xticks(np.arange(0, len(xz_grid[0]), 30), xz_grid[0][::30])
        ax3.set_yticks(np.arange(0, len(xz_grid[1]), 20), xz_grid[1][::20])
        ax3.set_xlabel('x(m)')
        ax3.set_ylabel('z(m)')
        ax3.grid(alpha=.2)


        # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
        image = Image.open(image_file).convert('RGB')
        shape = 900
        scale = 15 # rotio of pixel / meter
        birdimage = np.zeros((shape, shape, 3), np.uint8)

        # open(label_file) as f1, 
        with open(prediction_file) as f2: 
            for line_p in  f2:
                # line_gt = line_gt.strip().split(' ')
                line_p = line_p.strip().split(' ')
                truncated = np.abs(float(line_p[1]))
                occluded = np.abs(float(line_p[2]))
                # trunc_level = 1 if args.a == 'training' else 255
                trunc_level =  255

            # truncated object in dataset is not observable
                if line_p[0] in VEHICLES  and truncated < trunc_level:
                    color = 'green'
                    if line_p[0] == 'Cyclist':
                        color = 'yellow'
                    elif line_p[0] == 'Pedestrian':
                        color = 'cyan'
                    draw_3Dbox(ax, P2, line_p, color)
                    draw_birdeyes(ax2, None, line_p, shape, scale)
                    draw_on_chirp(ax3, line_p, xz_grid[0], xz_grid[1], color)


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
        # video_writer.write(np.uint8(fig))

def main(args):
    label_path = None
    
    # kitti dataset output example
    # image_path = '/home/andy/SMOKE/datasets/kitti/testing/image_2'
    # calib_path = '/home/andy/SMOKE/datasets/kitti/testing/calib'
    # pred_path = '/home/andy/SMOKE/tools/logs/inference/kitti_test/data'

    cruw_seq_train_root = '/mnt/disk1/CRUW/ROD2021/sequences/train'
    radar_dir = 'RADAR_XZ_H_MAG'
    image_path = f'/home/andy/SMOKE/datasets/CRUW/train/{date}/IMAGES_0'
    calib_path = None
    pred_path = '/home/andy/SMOKE/tools/logs/inference/CRUW4/data'# change here
    
    dataset = [name.split('.')[0] for name in sorted(os.listdir(image_path))]

    VEHICLES = ['Car', 'Cyclist', 'Pedestrian']
    start_frame = 0
    # end_frame = len(os.listdir(image_path))
    end_frame = 3
    cam_type = '0929' # change here

    visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES, start_frame, end_frame, os.path.join(cruw_seq_train_root, date, radar_dir), cam_type)

if __name__ == '__main__':

    scale_xyz = (1.3686, 1.1231, 1.3977) # (1.35, 1, 1.35)
    date = '2019_09_29_ONRD001' # change here
    parser = argparse.ArgumentParser(description='Visualize 3D bounding box on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-a', '-dataset', type=str, default='tracklet', help='training dataset or tracklet')
    parser.add_argument('-s', '--save', type=bool, default=True, help='Save Figure or not')
    parser.add_argument('-p', '--path', type=str, default=f'./results/CRUW/train/{date}_scaled_xyz_new', help='Output Image folder') #change here
    parser.add_argument('-fr', '--frame_rate', type=str, default='10', help='frame rate of output video')
    parser.add_argument('-sv', '--save_video', type=bool, default=True, help='whether to save video')
    

    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    main(args)

    if args.save_video:
        os.chdir(args.path)
        subprocess.run(['ffmpeg', '-framerate', args.frame_rate, '-i', '%010d.png' , f'{date}_scaled_xyz_{scale_xyz}.mp4'])

