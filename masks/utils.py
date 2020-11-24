import cv2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def get_obj_position(obj_mask):
    """returns center position of the object"""
    (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = get_mask_box(obj_mask)
    position = {'x': (box_top_x + box_bottom_x) / 2, 'y': (box_top_y + box_bottom_y) / 2}
    return position


def split_obj_masks(mask, num_objs):
    obj_masks = []
    for obj_idx in range(num_objs):
        obj_masks.append(mask == obj_idx)
    return obj_masks


def get_mask_box(obj_mask):
    height, width = obj_mask.shape
    rows, cols = np.where(obj_mask == True)
    box_top_x, box_top_y = max(0, rows.min() - 1), max(0, cols.min() - 1)
    box_bottom_x, box_bottom_y = min(rows.max() + 1, height - 1), min(cols.max() + 1, width - 1)

    return (box_top_x, box_top_y), (box_bottom_x, box_bottom_y)


def get_mask_box(obj_mask):
    height, width = obj_mask.shape
    rows, cols = np.where(obj_mask == True)
    box_top_x, box_top_y = max(0, rows.min() - 1), max(0, cols.min() - 1)
    box_bottom_x, box_bottom_y = min(rows.max() + 1, height - 1), min(cols.max() + 1, width - 1)
    return (box_top_x, box_top_y), (box_bottom_x, box_bottom_y)


def mask_img(mask, img):
    img_arr = np.asarray(img)
    masked_arr = img_arr * mask[:, :, np.newaxis]
    return Image.fromarray(masked_arr)


def draw_appearance_bars(base_image, frame_objects_info):
    objects = []
    apperance_match = []
    for obj_key, obj_info in frame_objects_info.items():
        if obj_info['visible']:
            objects.append(str(obj_key))
            apperance_match.append(obj_info['appearance'])

    fig = plt.figure(figsize=(base_image.size[0] / 100, base_image.size[1] / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(objects, apperance_match)
    ax.set_ylabel('probability')
    ax.set_title('Object Appearance Matching')

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bars_img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    appearance_img = np.concatenate((np.array(base_image), bars_img), axis=1)
    return Image.fromarray(appearance_img)


def draw_bounding_boxes(base_image, frame_objects_info):
    box_img = np.array(base_image)
    for obj_key, obj_info in frame_objects_info.items():
        if obj_info['visible']:
            (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = get_mask_box(obj_info['mask'])
            box_img = cv2.rectangle(box_img, (box_top_y, box_top_x), (box_bottom_y, box_bottom_x), (255, 255, 0), 2)
            box_img = cv2.putText(box_img, str(obj_key), (box_top_y, box_top_x), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    return Image.fromarray(box_img)
