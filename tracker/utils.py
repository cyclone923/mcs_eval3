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


# def get_mask_box(obj_mask):
#     height, width = obj_mask.shape
#     rows, cols = np.where(obj_mask == True)
#     box_top_x, box_top_y = max(0, rows.min() - 1), max(0, cols.min() - 1)
#     box_bottom_x, box_bottom_y = min(rows.max() + 1, height - 1), min(cols.max() + 1, width - 1)
#     return (box_top_x, box_top_y), (box_bottom_x, box_bottom_y)


def mask_img(mask, img):
    img_arr = np.asarray(img)
    masked_arr = img_arr * mask[:, :, np.newaxis]
    return Image.fromarray(masked_arr)


def draw_appearance_bars(base_image, frame_objects_info):
    appearance_x_labels = []
    appearance_matches = []
    appearance_shape_prob = []
    appearance_shape_prob_labels = []
    appearance_color_prob = []
    appearance_color_prob_labels = []
    appearance_clr_hist_quotient = []

    for obj_key, obj_info in frame_objects_info.items():
        if obj_info['visible']:
            appearance_shape_prob.append(obj_info['appearance']['shape_prob'])
            appearance_shape_prob_labels.append(obj_info['appearance']['shape_prob_labels'])
            appearance_color_prob.append(obj_info['appearance']['color_prob'])
            appearance_color_prob_labels.append(obj_info['appearance']['color_prob_labels'])
            appearance_x_labels.append(obj_key)
            appearance_matches.append(obj_info['appearance']['match'])
            appearance_clr_hist_quotient.append(obj_info['appearance']['color_hist_quotient'])

    # plot shape info
    plot_rows, plot_cols = 3, 3
    fig, ax = plt.subplots(plot_rows, plot_cols, figsize=(base_image.size[0] * plot_cols / 100,
                                                          base_image.size[1] * plot_rows / 100),
                           dpi=100)

    sub_plot_i = 0

    # image
    ax[0, 0].imshow(np.array(base_image))
    sub_plot_i += 1

    # color hist
    ax_i, ax_j = sub_plot_i // plot_cols, sub_plot_i % plot_cols
    ax[ax_i, ax_j].bar(np.arange(len(appearance_clr_hist_quotient)), appearance_clr_hist_quotient)
    ax[ax_i, ax_j].set_ylim([0, 1.2])
    ax[ax_i, ax_j].set_ylabel('probability')
    ax[ax_i, ax_j].set_title('Color Histogram dist')
    ax[ax_i, ax_j].set_xticks(np.arange(len(appearance_clr_hist_quotient)))
    ax[ax_i, ax_j].set_xticklabels(appearance_x_labels)
    sub_plot_i += 1

    # appearance match
    ax_i, ax_j = sub_plot_i // plot_cols, sub_plot_i % plot_cols
    ax[ax_i, ax_j].bar(np.arange(len(appearance_matches)), np.array(appearance_matches).astype(np.int))
    ax[ax_i, ax_j].set_ylabel('match(binary)')
    ax[ax_i, ax_j].set_title('Appearance Match')
    ax[ax_i, ax_j].set_xticks(np.arange(len(appearance_clr_hist_quotient)))
    ax[ax_i, ax_j].set_xticklabels(appearance_x_labels)
    sub_plot_i += 1

    # shape class
    for o_i, object in enumerate(appearance_x_labels):
        ax_i, ax_j = sub_plot_i // plot_cols, sub_plot_i % plot_cols
        ax[ax_i, ax_j].set_ylim([0, 1.2])
        ax[ax_i, ax_j].bar(np.arange(len(appearance_shape_prob[o_i])), appearance_shape_prob[o_i])
        ax[ax_i, ax_j].set_ylabel('probability')
        ax[ax_i, ax_j].set_title('Shape Object:{}'.format(object))
        ax[ax_i, ax_j].set_xticks(np.arange(len(appearance_shape_prob[o_i])))
        ax[ax_i, ax_j].set_xticklabels(appearance_shape_prob_labels[o_i], rotation=20)
        ax[ax_i, ax_j].legend()
        sub_plot_i += 1

    # color class
    for o_i, object in enumerate(appearance_x_labels):
        ax_i, ax_j = sub_plot_i // plot_cols, sub_plot_i % plot_cols
        ax[ax_i, ax_j].set_ylim([0, 1.2])
        ax[ax_i, ax_j].bar(np.arange(len(appearance_color_prob[o_i])), appearance_color_prob[o_i])
        ax[ax_i, ax_j].set_ylabel('probability')
        ax[ax_i, ax_j].set_title('Object Color:{}'.format(object))

        ax[ax_i, ax_j].set_xticks(np.arange(len(appearance_color_prob[o_i])))
        ax[ax_i, ax_j].set_xticklabels(appearance_color_prob_labels[o_i], rotation=20)
        ax[ax_i, ax_j].legend()
        sub_plot_i += 1

    # clr histogram deviation
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bars_img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    # appearance_img = np.concatenate((np.array(base_image), bars_img), axis=1)
    return Image.fromarray(bars_img)


def draw_bounding_boxes(base_image, frame_objects_info):
    box_img = np.array(base_image)
    for obj_key, obj_info in frame_objects_info.items():
        if obj_info['visible']:
            (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = get_mask_box(obj_info['mask'])
            box_img = cv2.rectangle(box_img, (box_top_y, box_top_x), (box_bottom_y, box_bottom_x), (255, 255, 0), 2)
            box_img = cv2.putText(box_img, str(obj_key), (box_top_y, box_top_x), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=3)

    return Image.fromarray(box_img)
