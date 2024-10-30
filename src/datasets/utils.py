import torch
import numpy as np


def points2d_to_bbox(point2d, margin=0, square=True):

    x_min = int(min(point2d[:, 0]))
    y_min = int(min(point2d[:, 1]))
    x_max = int(max(point2d[:, 0]))
    y_max = int(max(point2d[:, 1]))

    w = x_max - x_min
    h = y_max - y_min

    if square and (h != w):
        diff = abs(h-w)
        pad = int(diff/2)
        if h > w:
            x_min -= pad if diff % 2 == 0 else pad+1
            x_max += pad
        else:
            y_min -= pad if diff % 2 == 0 else pad+1
            y_max += pad

    bbox = np.array([x_min-margin, y_min-margin, x_max+margin, y_max+margin])

    return bbox


def bbox_to_cropped_bbox(bbox, image_shape):
    # restricting bbox to image boundaries
    # [x1, y1, x2, y2] -> [x1, y1, x2, y2]
    return np.array([max(0, bbox[0]), max(0, bbox[1]), min(image_shape[1], bbox[2]), min(image_shape[0], bbox[3])])


def crop_image(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()


def crop_and_pad_image(image, bounding_box):
    # Get the dimensions of the image
    if len(image.shape) == 3:
        height, width, channels = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    else:
        raise ValueError("Invalid image shape. Expected (h, w, 3) or (h, w), but got shape {}".format(image.shape))

    # Get the bounding box coordinates
    x1, y1, x2, y2 = bounding_box

    # Calculate the cropped region
    start_x = max(0, x1)
    start_y = max(0, y1)
    end_x = min(width, x2)
    end_y = min(height, y2)

    # Calculate the dimensions of the cropped region
    cropped_width = end_x - start_x
    cropped_height = end_y - start_y

    # Initialize the cropped image with zeros
    if channels == 1:
        cropped_image = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    else:
        cropped_image = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)

    # Calculate the region to paste the cropped image in the padded canvas
    paste_start_x = max(0, -x1)
    paste_start_y = max(0, -y1)

    # Copy the cropped region into the padded canvas
    cropped_image[paste_start_y:paste_start_y + cropped_height, paste_start_x:paste_start_x + cropped_width] = \
        image[start_y:end_y, start_x:end_x].copy()

    return cropped_image


def get_visible_joints_2d(joints_2d, input_res):
    joints_vis = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < input_res[1])) & \
                    ((joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < input_res[0]))
    return joints_vis.astype(np.float32)


def generate_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def batch_joints_img_to_cropped_joints(pts, bboxes, image_size=None):
    # pts: [b, 21, 2]
    # bboxes: [b, 4]
    # Subtract bbox top-left corner (x1, y1)
    if image_size is None:
        image_size = 256.
    # Copy tensors
    pts = pts.clone() if isinstance(pts, torch.Tensor) else np.copy(pts)

    pts[:, :, :2] -= bboxes[:, None, :2]

    # Calculate widths and heights from bboxes
    widths = bboxes[:, None, 2] - bboxes[:, None, 0]
    heights = bboxes[:, None, 3] - bboxes[:, None, 1]

    # Scale points by width and height respectively
    pts[:, :, 0] *= image_size / widths
    pts[:, :, 1] *= image_size / heights

    return pts


def batch_cropped_joints_to_joints_img(cropped_pts, bboxes, image_size=None):
    # cropped_pts: [b, 21, 2]
    # bboxes: [b, 4]
    if image_size is None:
        image_size = 256.
    # Copy tensors
    cropped_pts = cropped_pts.clone() if isinstance(cropped_pts, torch.Tensor) else np.copy(cropped_pts)
    # Scale points back to the original bbox size
    widths = bboxes[:, None, 2] - bboxes[:, None, 0]
    heights = bboxes[:, None, 3] - bboxes[:, None, 1]
    cropped_pts[:, :, 0] *= widths / image_size
    cropped_pts[:, :, 1] *= heights / image_size

    # Add the bbox top-left corner (x1, y1)
    cropped_pts[:, :, :2] += bboxes[:, None, :2]

    return cropped_pts


def joints_img_to_cropped_joints(pts, bboxes, image_size=None):
    if image_size is None:
        image_size = 256.
    cropped_pts = np.copy(pts)
    for i in range(pts.shape[0]):
        cropped_pts[i, :, 0] -= bboxes[i][0]
        cropped_pts[i, :, 1] -= bboxes[i][1]

        cropped_pts[i, :, 0] *= image_size/(bboxes[i][2]-bboxes[i][0])
        cropped_pts[i, :, 1] *= image_size/(bboxes[i][3]-bboxes[i][1])
    return cropped_pts


def center_scale_to_box(center, scale):
    """Convert bbox center scale to bbox xyxy

    Args:
        center (np.array): center of the bbox (x, y)
        scale (np.float_): side length of the bbox (bbox must be square)

    Returns:
        list: list of 4 elms, containing bbox' s xmin, ymin, xmax, ymax.
    """
    pixel_std = 1.0
    w = scale * pixel_std
    h = scale * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return bbox


def batch_center_scale_to_box(center, scale):
    """Convert batch of bbox centers and scales to bbox xyxy coordinates.

    Args:
        center (np.array): Array of centers of the bboxes (batch_size, 2) where each row is (x, y).
        scale (np.array): Array of side lengths of the bboxes (batch_size,) where each element is the scale.

    Returns:
        np.array: Array of shape (batch_size, 4) containing bbox's [xmin, ymin, xmax, ymax] for each item in the batch.
    """
    pixel_std = 1.0
    w = scale * pixel_std
    h = scale * pixel_std
    
    xmin = center[:, 0] - w * 0.5
    ymin = center[:, 1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    
    # Stack the results into a batch of bounding boxes
    bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1).astype("int")
    
    return bboxes


def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError('Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError('Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def center_scale_to_box(center, scale):
    """Convert bbox center scale to bbox xyxy

    Args:
        center (np.array): center of the bbox (x, y)
        scale (np.float_): side length of the bbox (bbox must be square)

    Returns:
        list: list of 4 elms, containing bbox' s xmin, ymin, xmax, ymax.
    """
    pixel_std = 1.0
    w = scale * pixel_std
    h = scale * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox
