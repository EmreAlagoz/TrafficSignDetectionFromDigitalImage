import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
import sys


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_img(path, width=None):
    img = cv2.imread(path)
    if width:
        img = image_resize(img, width=width)

    return img


def resize_and_pad_image(
    image,
    min_side=512,
    max_side=1024,
    jitter=[512, 1024],
    stride=128.0
):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def swap_xy(boxes):
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def to_xyxy(bbox):
    return tf.stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 2] + bbox[:, 0], bbox[:, 3] + bbox[:, 1],],
        axis=-1,
    )


def normalize_bbox(bbox, w=1622, h=626):
    return tf.stack([
        bbox[:, 0] / w,
        bbox[:, 1] / h,
        bbox[:, 2] / w,
        bbox[:, 3] / h,
    ], axis=-1)


def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def compute_iou(boxes1, boxes2):
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(
    image, boxes, classes, scores, figsize=(15, 15), linewidth=2, color=[1, 0, 0],
    box_true=None, label_true=None, save_path=''
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)

    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(boxes)):
        box, _cls, score = boxes[i], classes[i], scores[i]

        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    if box_true is not None and label_true is not None:
        for i in range(len(box_true)):
            box_t, cls_t = box_true[i], label_true[i]
            text = "{}: {:.2f}".format(cls_t, 1.0)
            x1, y1, w, h = box_t
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False,
                edgecolor=[1,1,1], linewidth=3
            )
            ax.add_patch(patch)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    return ax


def try_ignore_error(func, *argv):
    try:
        func(*argv)
    except Exception as e:
        print("WARN: ", e)
