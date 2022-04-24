import logging

import pyrealsense as pyrs
from pyrealsense.constants import rs_option

import os
import sys
import torch
import time
import math
import numpy as np
import cv2
from pydub import AudioSegment
from pydub.playback import play

from threading import Thread, Timer
import pyttsx3
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg
from segmentron.data.dataloader.cocostuff import IDNAME, MAPPING

coco_plattet182 = [
    [167, 200, 7],
    [127, 228, 215],
    [26, 135, 248],
    [238, 73, 166],
    [91, 210, 215],
    [122, 20, 236],
    [234, 173, 35],
    [34, 98, 46],
    [115, 11, 206],
    [52, 251, 238],
    [209, 156, 236],
    [239, 10, 0],
    [26, 122, 36],
    [162, 181, 66],
    [26, 64, 22],
    [46, 226, 200],
    [89, 176, 6],
    [103, 36, 32],
    [74, 89, 159],
    [250, 215, 25],
    [57, 246, 82],
    [51, 156, 111],
    [139, 114, 219],
    [65, 208, 253],
    [33, 184, 119],
    [230, 239, 58],
    [176, 141, 158],
    [21, 29, 31],
    [135, 133, 163],
    [152, 241, 248],
    [253, 54, 7],
    [231, 86, 229],
    [179, 220, 46],
    [155, 217, 185],
    [58, 251, 190],
    [40, 201, 63],
    [236, 52, 220],
    [71, 203, 170],
    [96, 56, 41],
    [252, 231, 125],
    [255, 60, 100],
    [11, 172, 184],
    [127, 46, 248],
    [1, 105, 163],
    [191, 218, 95],
    [87, 160, 119],
    [149, 223, 79],
    [216, 180, 245],
    [58, 226, 163],
    [11, 43, 118],
    [20, 23, 100],
    [71, 222, 109],
    [124, 197, 150],
    [38, 106, 43],
    [115, 73, 156],
    [113, 110, 50],
    [94, 2, 184],
    [163, 168, 155],
    [83, 39, 145],
    [150, 169, 81],
    [134, 25, 2],
    [145, 49, 138],
    [46, 27, 209],
    [145, 187, 117],
    [197, 9, 211],
    [179, 12, 118],
    [107, 241, 133],
    [255, 176, 224],
    [49, 56, 217],
    [10, 227, 177],
    [152, 117, 25],
    [139, 76, 23],
    [53, 191, 10],
    [14, 244, 90],
    [247, 94, 189],
    [202, 160, 149],
    [24, 31, 150],
    [164, 236, 24],
    [47, 10, 204],
    [84, 187, 44],
    [17, 153, 55],
    [9, 191, 39],
    [216, 53, 216],
    [54, 13, 26],
    [241, 13, 196],
    [157, 90, 225],
    [99, 195, 27],
    [20, 186, 253],
    [175, 192, 0],
    [81, 11, 238],
    [137, 83, 196],
    [53, 186, 24],
    [231, 20, 101],
    [246, 223, 173],
    [75, 202, 249],
    [9, 188, 201],
    [216, 83, 7],
    [152, 92, 54],
    [137, 192, 79],
    [242, 169, 49],
    [99, 65, 207],
    [178, 112, 1],
    [120, 135, 40],
    [71, 220, 82],
    [180, 83, 172],
    [68, 137, 75],
    [46, 58, 15],
    [0, 80, 68],
    [175, 86, 173],
    [19, 208, 152],
    [215, 235, 142],
    [95, 30, 166],
    [246, 193, 8],
    [222, 19, 72],
    [177, 29, 183],
    [238, 61, 178],
    [246, 136, 87],
    [199, 207, 174],
    [218, 149, 231],
    [98, 179, 168],
    [23, 10, 10],
    [223, 9, 253],
    [206, 114, 95],
    [177, 242, 152],
    [115, 189, 142],
    [254, 105, 107],
    [59, 175, 153],
    [42, 114, 178],
    [50, 121, 91],
    [78, 238, 175],
    [232, 201, 123],
    [61, 39, 248],
    [76, 43, 218],
    [121, 191, 38],
    [13, 164, 242],
    [83, 70, 160],
    [109, 2, 64],
    [252, 81, 105],
    [151, 107, 83],
    [31, 95, 170],
    [7, 238, 218],
    [227, 49, 19],
    [56, 102, 49],
    [152, 241, 48],
    [110, 35, 108],
    [59, 198, 242],
    [186, 189, 39],
    [26, 157, 41],
    [183, 16, 169],
    [114, 26, 104],
    [131, 142, 127],
    [118, 85, 219],
    [203, 84, 210],
    [245, 16, 127],
    [57, 238, 110],
    [223, 225, 154],
    [143, 21, 231],
    [12, 215, 113],
    [117, 58, 3],
    [170, 201, 252],
    [60, 190, 197],
    [38, 22, 24],
    [37, 155, 237],
    [175, 41, 211],
    [188, 151, 129],
    [231, 92, 102],
    [229, 112, 245],
    [157, 182, 40],
    [1, 60, 204],
    [57, 58, 19],
    [156, 199, 180],
    [211, 47, 8],
    [153, 115, 233],
    [172, 117, 198],
    [33, 63, 208],
    [107, 80, 154],
    [217, 164, 13],
    [136, 83, 59],
    [53, 206, 6],
    [95, 127, 75],
    [110, 22, 240],
    [244, 212, 2]
    ]


coco_id_name = {
    0: 'unlabeled', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
    8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter',
    14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant',
    22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie',
    29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
    36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle', 41: 'wine glass',
    42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
    50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair',
    58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse',
    66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink',
    73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
    80: 'toothbrush', 81: 'banner', 82: 'blanket', 83: 'branch', 84: 'bridge', 85: 'building-other', 86: 'bush',
    87: 'cabinet', 88: 'cage', 89: 'cardboard', 90: 'carpet', 91: 'ceiling-other', 92: 'ceiling-tile',
    93: 'cloth', 94: 'clothes', 95: 'clouds', 96: 'counter', 97: 'cupboard', 98: 'curtain', 99: 'desk',
    100: 'dirt', 101: 'door', 102: 'fence', 103: 'floor-marble', 104: 'floor-other', 105: 'floor-stone', 106: 'floor-tile',
    107: 'floor-wood', 108: 'flower', 109: 'fog', 110: 'food-other', 111: 'fruit', 112: 'furniture-other',
    113: 'grass', 114: 'gravel', 115: 'ground-other', 116: 'hill', 117: 'house', 118: 'leaves', 119: 'light',
    120: 'mat', 121: 'metal', 122: 'mirror-stuff', 123: 'moss', 124: 'mountain', 125: 'mud', 126: 'napkin', 127: 'net',
    128: 'paper', 129: 'pavement', 130: 'pillow', 131: 'plant-other', 132: 'plastic', 133: 'platform',
    134: 'playingfield', 135: 'railing', 136: 'railroad', 137: 'river', 138: 'road', 139: 'rock', 140: 'roof',
    141: 'rug', 142: 'salad', 143: 'sand', 144: 'sea', 145: 'shelf', 146: 'sky-other', 147: 'skyscraper', 148: 'snow',
    149: 'solid-other', 150: 'stair', 151: 'stone', 152: 'straw', 153: 'structural-other', 154: 'table', 155: 'tent',
    156: 'textile-other', 157: 'towel', 158: 'tree', 159: 'vegetable', 160: 'wall-brick', 161: 'wall-concrete',
    162: 'wall-other', 163: 'wall-panel', 164: 'wall-stone', 165: 'wall-tile', 166: 'wall-wood', 167: 'water-other',
    168: 'waterdrops', 169: 'window', 170: 'window', 171: 'wood'}


coco_plattet172 = [color for i, color in enumerate(coco_plattet182) if i in coco_id_name.keys()]

walkable = [103,104,105,106,107,
            114,115,125,129,133,134,136,
]

interest = [1, 2, 3, 4, 6, 7, 8,
            16, 17,
            63, 64, 65, 66, 67, 68, 69, 74,
            93, 94, 96, 97, 145,
            99, 101, 57, 61, 154, 169, 170,
            158,
            150
            ]

def convert_z16_to_bgr(frame):
    '''Performs depth histogram normalization
    This raw Python implementation is slow. See here for a fast implementation using Cython:
    https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/cython_methods/methods.pyx
    '''
    hist = np.histogram(frame, bins=0x10000)[0]
    hist = np.cumsum(hist)
    hist -= hist[0]
    rgb_frame = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)

    zeros = frame == 0
    non_zeros = frame != 0

    f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
    rgb_frame[non_zeros, 0] = 255 - f
    rgb_frame[non_zeros, 1] = 0
    rgb_frame[non_zeros, 2] = f
    rgb_frame[zeros, 0] = 20
    rgb_frame[zeros, 1] = 5
    rgb_frame[zeros, 2] = 0
    return rgb_frame

def demo():
    args = parse_args() 
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'vis_result_{}_{}_{}_{}'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()


    if os.path.isdir(args.input_img):
        img_paths = [os.path.join(args.input_img, x) for x in os.listdir(args.input_img)]
    else:
        img_paths = [args.input_img]
    for img_path in img_paths:
        image = Image.open(img_path).convert('RGB')
        size = image.size
        image = image.resize((512, 512))
        images = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, cfg.DATASET.NAME).resize(size)
        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '.png'
        mask.save(os.path.join(output_dir, outname))

from segmentron.models.model_zoo import MODEL_REGISTRY
from segmentron.data.dataloader import datasets

def second_model():
    from segmentron.models.pvt_fpt import PVT_FPT
    model_name = cfg.MODEL.MODEL_NAME
    datasets[cfg.DATASET.NAME].NUM_CLASS = 172
    cfg.MODEL.EMB_CHANNELS = 128
    model = MODEL_REGISTRY.get(model_name)()
    path = 'workdirs/cocostuff/pvt_tiny_FPT128/model_cocostuff.pth'
    model_dic = torch.load(path, map_location='cuda:0')
    if 'state_dict' in model_dic.keys():
        model_dic = model_dic['state_dict']
    msg = model.load_state_dict(model_dic, strict=False)
    logging.info(msg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

class TTS():
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('volume', 0.2)
    def run(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

def saying(text):
    tts = TTS()
    tts.run(text)
    del(tts)

def playfile(file, dB):
    sound = AudioSegment.from_wav(file)
    sound += dB
    play(sound)

def beep(n=2):
    '''play the ascii bell for n times.'''
    # os.system()
    for _ in range(n):
        sys.stdout.write('\a')
        sys.stdout.flush()
        time.sleep(0.02)


n_left, n_middle, n_right = 0, 0, 0
count_frame = 0
n_obstacle = 0
obstacles = list()
obstacle_distance = 1000  #mm
is_save = False

def segment_cam(fps, sleep_sec):
    global n_left, n_middle, n_right, obstacles, count_frame
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    if is_save:
        output_dir_rgb = 'vis_trans_coco/r200_demo_{}/rgb'.format( cfg.TIME_STAMP)
        os.makedirs(output_dir_rgb, exist_ok=True)
        output_dir_depth = 'vis_trans_coco/r200_demo_{}/depth'.format( cfg.TIME_STAMP)
        os.makedirs(output_dir_depth, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])
    with torch.no_grad():

        model = get_segmentation_model().to(args.device)
        model.eval()

        model_coco = second_model()
        model_coco.eval()


        w, h = 320, 240
        depth_stream = pyrs.stream.DepthStream(width=w, height=h, fps=fps)
        dac_stream = pyrs.stream.DACStream(width=w, height=h, fps=fps)
        ir_stream = pyrs.stream.InfraredStream(width=w, height=h, fps=fps)
        color_stream = pyrs.stream.ColorStream(width=w, height=h, fps=fps)
        with pyrs.Service() as serv:
            with serv.Device(streams=(depth_stream, dac_stream, ir_stream, color_stream)) as dev:
                dev.apply_ivcam_preset(0)
                try: 
                    custom_options = [
                                    (rs_option.RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED, 1),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_LR_THRESHOLD, 30),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_SCORE_MAXIMUM_THRESHOLD, 1023),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_MEDIAN_THRESHOLD, 0),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_SCORE_MINIMUM_THRESHOLD, 0),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_SECOND_PEAK_THRESHOLD, 0),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_COUNT_THRESHOLD, 0),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_TEXTURE_DIFFERENCE_THRESHOLD, 0),
                                    (rs_option.RS_OPTION_R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD, 0),
                                    (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
                    dev.set_device_options(*zip(*custom_options))
                except pyrs.RealsenseError:
                    pass 
                cnt = 0
                last = time.time()
                smoothing = 0.9
                fps_smooth = 30

                t1 = time.time()
                orientation_interval = 2
                left_obstacles, middle_obstacles, right_obstacles = [], [], []
                center_objs = []
                while True:

                    time.sleep(sleep_sec)
                    cnt += 1
                    if (cnt % 10) == 0:
                        now = time.time()
                        dt = now - last
                        fps = 10/dt
                        fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                        last = now
                    dev.wait_for_frames()
                    frame = dev.color
                    d = dev.dac
                    image = Image.fromarray(frame)
                    size = image.size
                    image = image.resize((512, 512))
                    images = transform(image).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        output = model(images)
                        output_coco = model_coco(images)

                    pred_trans = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
                    pred_coco = torch.argmax(output_coco[0], 1).squeeze(0).cpu().data.numpy()
                    pred_coco = pred_coco + 1

                    mask_trans = get_color_pallete(pred_trans, cfg.DATASET.NAME).resize(size)
                    mask_trans = cv2.cvtColor(np.asarray(mask_trans.convert('RGB')), cv2.COLOR_RGB2BGR)

                    mask_coco = postprocess(pred_coco, pred_trans)
                    mask_coco = cv2.resize(mask_coco, size)

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # --- orientation
                    t3 = time.time()
                    orientation(pred_coco)
                    max_orient = max([n_left, n_middle, n_right])

                    # --- obstacle
                    obstacle_mask, left_obstacle, middle_obstacle, right_obstacle = obstacle_detect(pred_coco, d)
                    left_obstacles.append(left_obstacle)
                    middle_obstacles.append(middle_obstacle)
                    right_obstacles.append(right_obstacle)

                    # --- interesting object
                    pred_coco_rgb, center_obj, distance_obj = interest_detect(pred_coco, pred_trans, d)
                    pred_coco_rgb = cv2.resize(pred_coco_rgb, size)
                    center_objs.append(center_obj)

                    # obstacles.append(obstacle)
                    if t3 - t1 > orientation_interval:
                        trans_mask, trans_obj = is_transparent_2(pred_trans)

                        is_left_obstacle = np.mean(left_obstacles) < obstacle_distance 
                        is_middle_obstacle = np.mean(middle_obstacles) < obstacle_distance 
                        is_right_obstacle = np.mean(right_obstacles) < obstacle_distance

                        # --- close_obstacle
                        close_obstacle = 1000
                        is_left_close_obstacle = np.mean(left_obstacles) < close_obstacle
                        is_middle_close_obstacle = np.mean(middle_obstacles) < close_obstacle
                        is_right_close_obstacle = np.mean(right_obstacles) < close_obstacle

                        # ---- close depth, obstacle
                        if is_left_close_obstacle or is_middle_close_obstacle or is_right_close_obstacle:
                            Thread(target=playfile, args=('sounds/both_1.wav', -20,)).start()
                        elif center_obj == 'stair':
                            Thread(target=saying, args=(center_obj,)).start()
                        elif distance_obj < 2000 and center_obj:
                            Thread(target=saying, args=(center_obj,)).start()
                            print('too closed object: ', center_obj)
                        elif trans_obj:
                            Thread(target=saying, args=(trans_obj,)).start()
                            print('transparent object:', trans_obj)
                        # ---- walkable path
                        elif max_orient/cnt > 0.4:
                            text = None
                            left_str = 'left'
                            right_str = 'right'
                            middle_str = 'forward'
                            if max_orient == n_left:
                                text = left_str
                                Thread(target=saying, args=(text,)).start()

                            elif max_orient == n_right:
                                text = right_str
                                Thread(target=saying, args=(text,)).start()
                            else:
                                text = middle_str
                                Thread(target=saying, args=(text,)).start()
                            print('walkable: ', text, n_left/cnt, n_middle/cnt, n_right/cnt)
                        # ---- interesting obj
                        elif center_obj:
                            Thread(target=saying, args=(center_obj,)).start()
                            print('interesting object: ', center_obj)
                        # ---- depth, obstacle
                        elif is_left_obstacle or is_middle_obstacle or is_right_obstacle:
                            Thread(target=playfile, args=('sounds/both_1.wav', -20,)).start()

                        n_left, n_middle, n_right = 0, 0, 0
                        left_obstacles, middle_obstacles, right_obstacles = [], [], []
                        t1 = t3
                        obstacles = []
                        center_objs = []
                        cnt = 0
                        count_frame = 0
                    
                    img_numpy = np.concatenate((frame, obstacle_mask, mask_trans, pred_coco_rgb, mask_coco), axis=1)
                    img_numpy = cv2.resize(img_numpy, (320*5, 240))

                    if is_save:
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        outname = os.path.join(output_dir_rgb, '{}.png'.format(timestr))
                        img_numpy = np.concatenate((frame, pred_coco_rgb), axis=1)
                        img_numpy = cv2.resize(img_numpy, (640 * 2, 480))
                        cv2.imwrite(outname, img_numpy)

                        outname = os.path.join(output_dir_depth, '{}.png'.format(timestr))
                        depth_img = Image.fromarray(d)
                        depth_img.save(outname)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

def draw_pallete():
    from segmentron.utils.visualize import stanford2d3dpallete, trans10kv2pallete
    trans_cls = ['Background', 'Shelf', 'Jar or Tank', 'Freezer', 'Window',
                'Glass Door', 'Eyeglass', 'Cup', 'Floor Glass', 'Glass Bow',
                'Water Bottle', 'Storage Box']

    cell_width = 180
    zeros = np.zeros((150, cell_width*10))
    for id, cls in enumerate(trans_cls):
        zeros[:, id*cell_width:((id+1)*cell_width)] = id
    p = get_color_pallete(zeros, 'transparent11')
    rgb = cv2.cvtColor(np.asarray(p.convert('RGB')), cv2.COLOR_RGB2BGR)
    rgb[100:150, :, :] = (255,255,255)
    for id, cls in enumerate(trans_cls):
        c = trans10kv2pallete[id*3:((id+1)*3)]
        c = c[::-1]
        cv2.putText(rgb, cls, (id*cell_width, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    cv2.imwrite('pallete.png', rgb)

def obstacle_detect(pred_coco, d):
    global n_obstacle
    pred_coco_img = pred_coco.astype(np.uint8)
    m = np.isin(cv2.resize(pred_coco_img, (d.shape[1], d.shape[0])), walkable)
    depth = np.where(~m, d, 5000)
    depth = np.where(depth > 1, depth, 5000)
    w = 320 #640
    region_obstacle = depth[:, :] # all region
    left_obstacle = region_obstacle[:, :w//3]
    left_obstacle = left_obstacle.flatten()
    left_obstacle = left_obstacle[left_obstacle!=5000]
    left_obstacle = left_obstacle.mean()
    middle_obstacle = region_obstacle[:, w//3:w-w//3]
    middle_obstacle = middle_obstacle.flatten()
    middle_obstacle = middle_obstacle[middle_obstacle!=5000]
    middle_obstacle = middle_obstacle.mean()
    right_obstacle = region_obstacle[:, w-w//3:]
    right_obstacle = right_obstacle.flatten()
    right_obstacle = right_obstacle[right_obstacle!=5000]
    right_obstacle = right_obstacle.mean()
    obstacle_mask = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    obstacle_mask[:,:,:3][depth < obstacle_distance] = (128, 128, 128)
    obstacle_mask[:,:,:3][depth < 1] = (0, 0, 0)


    obstacle_mask[:, w//3:w//3+3, :3] = (0, 0, 255)
    obstacle_mask[:, w-w//3:w-w//3+3, :3] = (0, 0, 255)
    return obstacle_mask, left_obstacle, middle_obstacle, right_obstacle

def is_transparent(pred_trans):
    ratio_transparent = 0.5
    mask = np.logical_or(pred_trans == 4, pred_trans == 5, pred_trans == 8)
    left, middle, right = mask[:, :170], mask[:, 170:342], mask[:, 342:]
    lw, mw, rw = np.count_nonzero(left)/left.size, np.count_nonzero(middle)/middle.size, np.count_nonzero(right)/right.size
    lw, mw, rw = round(lw, 2), round(mw, 2), round(rw, 2)
    l = lw > ratio_transparent
    m = mw > ratio_transparent
    r = rw > ratio_transparent
    # --- object
    print("transparent ratio (l, m, r):", lw, mw, rw)
    text = None
    if l or m or r:
        windows = np.count_nonzero(pred_trans == 4)
        doors = np.count_nonzero(pred_trans == 5)
        walls = np.count_nonzero(pred_trans == 8)
        max_trans = max(windows, doors, walls)
        if windows == max_trans:
            text = 'window'
        elif doors == max_trans:
            text = 'door'
        else:
            text = 'wall'

    return mask, l, m, r, text

def is_transparent_2(pred_trans):
    ratio_transparent = 0.5
    mask = np.logical_or(pred_trans == 4, pred_trans == 5, pred_trans == 8)
    text = None
    windows = np.count_nonzero(pred_trans == 4) / pred_trans.size
    doors = np.count_nonzero(pred_trans == 5) / pred_trans.size
    walls = np.count_nonzero(pred_trans == 8) / pred_trans.size
    max_trans = max(windows, doors, walls)
    if max_trans > ratio_transparent:
        if windows == max_trans:
            text = 'glass window'
        elif doors == max_trans:
            text = 'glass door'
        else:
            text = 'glass wall'

    return mask, text


def interest_detect(pred_coco, pred_trans, depth):
    masks = {}
    h, w = pred_coco.shape
    d = cv2.resize(depth, pred_coco.shape[:2])
    pred_coco_rgb = np.zeros((pred_coco.shape[0], pred_coco.shape[1], 3), dtype=np.uint8)
    init_distance = max(pred_coco.shape)
    fx, fy = h // 2, w // 2 
    center_obj = None
    distance_obj = 6000
    for i, idx in enumerate(interest):
        if idx in np.unique(pred_coco):
            key = coco_id_name[idx]
            mask = np.zeros_like(pred_coco, dtype=np.uint8)
            mask[pred_coco == idx] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            max_c = max(contours, key=cv2.contourArea)
            max_mask = np.zeros_like(pred_coco, dtype=np.uint8)
            cv2.drawContours(max_mask, [max_c], -1, 1, -1)
            x, y, w, h = cv2.boundingRect(max_c)
            cx, cy = x + w // 2, y + h // 2
            # --- depth < 2m
            dist_obj = np.mean(d[max_mask == 1])
            if dist_obj > 3000 or cx < h//4 or cx > h//4 * 3:
                continue
            masks[key] = max_mask
            rgb = coco_plattet172[idx]
            pred_coco_rgb[:, :, :3][pred_coco == idx] = rgb
            pred_coco_rgb = cv2.rectangle(pred_coco_rgb, (x, y), (x + w, y + h), rgb, 2)
            cv2.putText(pred_coco_rgb, key, (x+10, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.9, rgb, 1, cv2.LINE_AA)

            d2c = math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
            if d2c < init_distance:
                init_distance = d2c
                center_obj = key
                distance_obj = dist_obj
    return pred_coco_rgb, center_obj, distance_obj

def orientation(pred_coco):
    text = None
    left_str = 'left'
    right_str = 'right'
    middle_str = 'forward'
    global n_left, n_middle, n_right
    m = np.isin(pred_coco, walkable)
    left, middle, right = m[:, :170], m[:, 170:342], m[:, 342:]
    lw, mw, rw = np.count_nonzero(left)/left.size, np.count_nonzero(middle)/middle.size, np.count_nonzero(right)/right.size
    lw, mw, rw = round(lw, 2), round(mw, 2), round(rw, 2)
    max_walkable = max([lw, mw, rw])
    if max_walkable > 0.2:
        if lw == max_walkable:
            n_left += 1
            text = left_str
        elif rw == max_walkable:
            n_right += 1
            text = right_str
        else:
            n_middle += 1
            text = middle_str

def postprocess(pred_coco, pred_trans):
    mask_floor = np.zeros((512, 512, 3), dtype=np.uint8)
    m = np.isin(pred_coco, walkable)
    mask_floor[:, :, :3][m] = (152, 251, 152)
    mask_floor[:, 168:170, :3] = (0, 0, 255)
    mask_floor[:, 342:344, :3] = (0, 0, 255)
    return mask_floor


if __name__ == '__main__':
    segment_cam(fps=60, sleep_sec=0.001)