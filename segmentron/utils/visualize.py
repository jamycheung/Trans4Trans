import os
import logging
import numpy as np
import torch

from PIL import Image
#from torchsummary import summary
from thop import profile

__all__ = ['get_color_pallete', 'print_iou', 'set_img_color',
           'show_prediction', 'show_colorful_images', 'save_colorful_images']


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        # lines.append('%-8s: %.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('mean_IU: %.3f%% || mean_IU_no_back: %.3f%% || mean_pixel_acc: %.3f%%' % (
            mean_IU * 100, mean_IU_no_back * 100, mean_pixel_acc * 100))
    else:
        lines.append('mean_IU: %.3f%% || mean_pixel_acc: %.3f%%' % (mean_IU * 100, mean_pixel_acc * 100))
    lines.append('=================================================')
    line = "\n".join(lines)

    print(line)


@torch.no_grad()
def show_flops_params(model, device, input_shape=[1, 3, 512, 512]):
    #summary(model, tuple(input_shape[1:]), device=device)
    input = torch.randn(*input_shape).to(torch.device(device))
    flops, params = profile(model, inputs=(input,), verbose=False)

    logging.info('{} flops: {:.3f}G input shape is {}, params: {:.3f}M'.format(
        model.__class__.__name__, flops / 1000000000, input_shape[1:], params / 1000000))


def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out


def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    im.show()


def save_colorful_images(prediction, filename, output_dir, palettes):
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


def get_color_pallete(npimg, dataset='cityscape'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255

    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'cityscape':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityscapepallete)
        return out_img
    elif dataset in ['trans10kv2', 'transparent11']:
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(trans10kv2pallete)
        return out_img
    elif dataset == 'pascal_voc':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(vocpallete)
        return out_img
    elif dataset == 'stanford2d3d':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(stanford2d3dpallete)
        return out_img
    elif dataset == 'cocostuff':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cocostuffpallete)
        return out_img

def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _getvocpallete(256)

adepallete = [
    0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,
    5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
    143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255,
    7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6,
    10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255,
    20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15,
    20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255,
    31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163,
    0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255,
    0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0,
    31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0,
    194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,
    0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255,
    0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,
    163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
    10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0,
    255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,
    133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]

cityscapepallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

trans10kv2pallete = [
    0, 0, 0,
    120, 120, 70,
    235, 255, 7,
    6, 230, 230,
    204, 255, 4,
    120, 120, 120,
    140, 140, 140,
    255, 51, 7,
    224, 5, 255,
    204, 5, 255,
    150, 5, 61,
    4, 250, 7]

stanford2d3dpallete = [
    0,   0,   0,
    255,   0,  40,
    255,  72,   0,
    255, 185,   0,
    205, 255,   0,
     91, 255,   0,
      0, 255,  21,
      0, 255, 139,
      0, 255, 252,
      0, 143, 255,
      0,  23, 255,
     90,   0, 255,
    204,   0, 255,
    255,   0, 191]

cocostuffpallete = [
    167, 200, 7,
    127, 228, 215,
    26, 135, 248,
    238, 73, 166,
    91, 210, 215,
    122, 20, 236,
    234, 173, 35,
    34, 98, 46,
    115, 11, 206,
    52, 251, 238,
    209, 156, 236,
    239, 10, 0,
    26, 122, 36,
    162, 181, 66,
    26, 64, 22,
    46, 226, 200,
    89, 176, 6,
    103, 36, 32,
    74, 89, 159,
    250, 215, 25,
    57, 246, 82,
    51, 156, 111,
    139, 114, 219,
    65, 208, 253,
    33, 184, 119,
    230, 239, 58,
    176, 141, 158,
    21, 29, 31,
    135, 133, 163,
    152, 241, 248,
    253, 54, 7,
    231, 86, 229,
    179, 220, 46,
    155, 217, 185,
    58, 251, 190,
    40, 201, 63,
    236, 52, 220,
    71, 203, 170,
    96, 56, 41,
    252, 231, 125,
    255, 60, 100,
    11, 172, 184,
    127, 46, 248,
    1, 105, 163,
    191, 218, 95,
    87, 160, 119,
    149, 223, 79,
    216, 180, 245,
    58, 226, 163,
    11, 43, 118,
    20, 23, 100,
    71, 222, 109,
    124, 197, 150,
    38, 106, 43,
    115, 73, 156,
    113, 110, 50,
    94, 2, 184,
    163, 168, 155,
    83, 39, 145,
    150, 169, 81,
    134, 25, 2,
    145, 49, 138,
    46, 27, 209,
    145, 187, 117,
    197, 9, 211,
    179, 12, 118,
    107, 241, 133,
    255, 176, 224,
    49, 56, 217,
    10, 227, 177,
    152, 117, 25,
    139, 76, 23,
    53, 191, 10,
    14, 244, 90,
    247, 94, 189,
    202, 160, 149,
    24, 31, 150,
    164, 236, 24,
    47, 10, 204,
    84, 187, 44,
    17, 153, 55,
    9, 191, 39,
    216, 53, 216,
    54, 13, 26,
    241, 13, 196,
    157, 90, 225,
    99, 195, 27,
    20, 186, 253,
    175, 192, 0,
    81, 11, 238,
    137, 83, 196,
    53, 186, 24,
    231, 20, 101,
    246, 223, 173,
    75, 202, 249,
    9, 188, 201,
    216, 83, 7,
    152, 92, 54,
    137, 192, 79,
    242, 169, 49,
    99, 65, 207,
    178, 112, 1,
    120, 135, 40,
    71, 220, 82,
    180, 83, 172,
    68, 137, 75,
    46, 58, 15,
    0, 80, 68,
    175, 86, 173,
    19, 208, 152,
    215, 235, 142,
    95, 30, 166,
    246, 193, 8,
    222, 19, 72,
    177, 29, 183,
    238, 61, 178,
    246, 136, 87,
    199, 207, 174,
    218, 149, 231,
    98, 179, 168,
    23, 10, 10,
    223, 9, 253,
    206, 114, 95,
    177, 242, 152,
    115, 189, 142,
    254, 105, 107,
    59, 175, 153,
    42, 114, 178,
    50, 121, 91,
    78, 238, 175,
    232, 201, 123,
    61, 39, 248,
    76, 43, 218,
    121, 191, 38,
    13, 164, 242,
    83, 70, 160,
    109, 2, 64,
    252, 81, 105,
    151, 107, 83,
    31, 95, 170,
    7, 238, 218,
    227, 49, 19,
    56, 102, 49,
    152, 241, 48,
    110, 35, 108,
    59, 198, 242,
    186, 189, 39,
    26, 157, 41,
    183, 16, 169,
    114, 26, 104,
    131, 142, 127,
    118, 85, 219,
    203, 84, 210,
    245, 16, 127,
    57, 238, 110,
    223, 225, 154,
    143, 21, 231,
    12, 215, 113,
    117, 58, 3,
    170, 201, 252,
    60, 190, 197,
    38, 22, 24,
    37, 155, 237,
    175, 41, 211,
    188, 151, 129,
    231, 92, 102,
    229, 112, 245,
    157, 182, 40,
    1, 60, 204,
    57, 58, 19,
    156, 199, 180,
    211, 47, 8
]
