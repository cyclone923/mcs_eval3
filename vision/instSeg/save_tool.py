from collections import defaultdict
import os
import cv2
import numpy as np
from PIL import Image as PilImage
from PIL import ImageDraw as PilImageDraw
from PIL import ImageFont
from fonts.ttf import FredokaOne

'''
pip install fonts
pip install font-fredoka-one
pip install font-amatic-sc
'''

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


class ColorMap(object):
    def __init__(self, cmap_name='pascal', label_num=256):
        if cmap_name == 'pascal':
            self.create_pascal_cmap(label_num)
        elif cmap_name == 'jet':
            self.create_jet_cmap(label_num)
        else:
            print('Sorry, currently we only support color map PASCAL  and JET ')

    def create_pascal_cmap(self, label_num=21):
        self.inv_cmap = defaultdict()
        self.cmap     = np.zeros([label_num, 3], dtype=np.uint8)
        for i in range(label_num):
            k, r,g,b = i, 0,0,0
            for j in range(8):
                r |= (k&1)      << (7-j)
                g |= ((k&2)>>1) << (7-j)
                b |= ((k&4)>>2) << (7-j)
                k = k>>3

            self.cmap[i,:] = [r,g,b]
            inv_key = (((r<<8) + g)<<8) + b
            self.inv_cmap[inv_key] = i


    def convert_label2rgb(self, inI):
        ht, wd = inI.shape
        rgbI = np.zeros([ht, wd, 3], dtype = np.uint8)
        rgbI[:,:,0] = self.cmap[inI, 0]
        rgbI[:,:,1] = self.cmap[inI, 1]
        rgbI[:,:,2] = self.cmap[inI, 2]

        return rgbI

    def get_colormap(self):
        return self.cmap/255.0


    def convert_rgb2label(self, inI):
        ht, wd, _ = inI.shape
        grayI = np.zeros([ht, wd], dtype=np.uint8)

        inI   = inI.astype(np.int)
        keyI  = (((inI[:,:,2]<<8) + inI[:,:,1])<<8) + inI[:,:,0]
        keyList = np.unique(keyI)
        for ele in keyList:
            grayI[keyI==ele] = self.inv_cmap[ele]

        return grayI

    def create_jet_cmap(self, label_num=256):
        if(label_num > 256):
            print('Sorry, now only support jet map with <= 256 colors.')

        full_cmap = [ 0,          0,     0,
                      0,          0,     0.5312,
                      0,          0,     0.5469,
                      0,          0,     0.5625,
                      0,          0,     0.5781,
                      0,          0,     0.5938,
                      0,          0,     0.6094,
                      0,          0,     0.6250,
                      0,          0,     0.6406,
                      0,          0,     0.6562,
                      0,          0,     0.6719,
                      0,          0,     0.6875,
                      0,          0,     0.7031,
                      0,          0,     0.7188,
                      0,          0,     0.7344,
                      0,          0,     0.7500,
                      0,          0,     0.7656,
                      0,          0,     0.7812,
                      0,          0,     0.7969,
                      0,          0,     0.8125,
                      0,          0,     0.8281,
                      0,          0,     0.8438,
                      0,          0,     0.8594,
                      0,          0,     0.8750,
                      0,          0,     0.8906,
                      0,          0,     0.9062,
                      0,          0,     0.9219,
                      0,          0,     0.9375,
                      0,          0,     0.9531,
                      0,          0,     0.9688,
                      0,          0,     0.9844,
                      0,          0,     1.0000,
                      0,     0.0156,     1.0000,
                      0,     0.0312,     1.0000,
                      0,     0.0469,     1.0000,
                      0,     0.0625,     1.0000,
                      0,     0.0781,     1.0000,
                      0,     0.0938,     1.0000,
                      0,     0.1094,     1.0000,
                      0,     0.1250,     1.0000,
                      0,     0.1406,     1.0000,
                      0,     0.1562,     1.0000,
                      0,     0.1719,     1.0000,
                      0,     0.1875,     1.0000,
                      0,     0.2031,     1.0000,
                      0,     0.2188,     1.0000,
                      0,     0.2344,     1.0000,
                      0,     0.2500,     1.0000,
                      0,     0.2656,     1.0000,
                      0,     0.2812,     1.0000,
                      0,     0.2969,     1.0000,
                      0,     0.3125,     1.0000,
                      0,     0.3281,     1.0000,
                      0,     0.3438,     1.0000,
                      0,     0.3594,     1.0000,
                      0,     0.3750,     1.0000,
                      0,     0.3906,     1.0000,
                      0,     0.4062,     1.0000,
                      0,     0.4219,     1.0000,
                      0,     0.4375,     1.0000,
                      0,     0.4531,     1.0000,
                      0,     0.4688,     1.0000,
                      0,     0.4844,     1.0000,
                      0,     0.5000,     1.0000,
                      0,     0.5156,     1.0000,
                      0,     0.5312,     1.0000,
                      0,     0.5469,     1.0000,
                      0,     0.5625,     1.0000,
                      0,     0.5781,     1.0000,
                      0,     0.5938,     1.0000,
                      0,     0.6094,     1.0000,
                      0,     0.6250,     1.0000,
                      0,     0.6406,     1.0000,
                      0,     0.6562,     1.0000,
                      0,     0.6719,     1.0000,
                      0,     0.6875,     1.0000,
                      0,     0.7031,     1.0000,
                      0,     0.7188,     1.0000,
                      0,     0.7344,     1.0000,
                      0,     0.7500,     1.0000,
                      0,     0.7656,     1.0000,
                      0,     0.7812,     1.0000,
                      0,     0.7969,     1.0000,
                      0,     0.8125,     1.0000,
                      0,     0.8281,     1.0000,
                      0,     0.8438,     1.0000,
                      0,     0.8594,     1.0000,
                      0,     0.8750,     1.0000,
                      0,     0.8906,     1.0000,
                      0,     0.9062,     1.0000,
                      0,     0.9219,     1.0000,
                      0,     0.9375,     1.0000,
                      0,     0.9531,     1.0000,
                      0,     0.9688,     1.0000,
                      0,     0.9844,     1.0000,
                      0,     1.0000,     1.0000,
                 0.0156,     1.0000,     0.9844,
                 0.0312,     1.0000,     0.9688,
                 0.0469,     1.0000,     0.9531,
                 0.0625,     1.0000,     0.9375,
                 0.0781,     1.0000,     0.9219,
                 0.0938,     1.0000,     0.9062,
                 0.1094,     1.0000,     0.8906,
                 0.1250,     1.0000,     0.8750,
                 0.1406,     1.0000,     0.8594,
                 0.1562,     1.0000,     0.8438,
                 0.1719,     1.0000,     0.8281,
                 0.1875,     1.0000,     0.8125,
                 0.2031,     1.0000,     0.7969,
                 0.2188,     1.0000,     0.7812,
                 0.2344,     1.0000,     0.7656,
                 0.2500,     1.0000,     0.7500,
                 0.2656,     1.0000,     0.7344,
                 0.2812,     1.0000,     0.7188,
                 0.2969,     1.0000,     0.7031,
                 0.3125,     1.0000,     0.6875,
                 0.3281,     1.0000,     0.6719,
                 0.3438,     1.0000,     0.6562,
                 0.3594,     1.0000,     0.6406,
                 0.3750,     1.0000,     0.6250,
                 0.3906,     1.0000,     0.6094,
                 0.4062,     1.0000,     0.5938,
                 0.4219,     1.0000,     0.5781,
                 0.4375,     1.0000,     0.5625,
                 0.4531,     1.0000,     0.5469,
                 0.4688,     1.0000,     0.5312,
                 0.4844,     1.0000,     0.5156,
                 0.5000,     1.0000,     0.5000,
                 0.5156,     1.0000,     0.4844,
                 0.5312,     1.0000,     0.4688,
                 0.5469,     1.0000,     0.4531,
                 0.5625,     1.0000,     0.4375,
                 0.5781,     1.0000,     0.4219,
                 0.5938,     1.0000,     0.4062,
                 0.6094,     1.0000,     0.3906,
                 0.6250,     1.0000,     0.3750,
                 0.6406,     1.0000,     0.3594,
                 0.6562,     1.0000,     0.3438,
                 0.6719,     1.0000,     0.3281,
                 0.6875,     1.0000,     0.3125,
                 0.7031,     1.0000,     0.2969,
                 0.7188,     1.0000,     0.2812,
                 0.7344,     1.0000,     0.2656,
                 0.7500,     1.0000,     0.2500,
                 0.7656,     1.0000,     0.2344,
                 0.7812,     1.0000,     0.2188,
                 0.7969,     1.0000,     0.2031,
                 0.8125,     1.0000,     0.1875,
                 0.8281,     1.0000,     0.1719,
                 0.8438,     1.0000,     0.1562,
                 0.8594,     1.0000,     0.1406,
                 0.8750,     1.0000,     0.1250,
                 0.8906,     1.0000,     0.1094,
                 0.9062,     1.0000,     0.0938,
                 0.9219,     1.0000,     0.0781,
                 0.9375,     1.0000,     0.0625,
                 0.9531,     1.0000,     0.0469,
                 0.9688,     1.0000,     0.0312,
                 0.9844,     1.0000,     0.0156,
                 1.0000,     1.0000,          0,
                 1.0000,     0.9844,          0,
                 1.0000,     0.9688,          0,
                 1.0000,     0.9531,          0,
                 1.0000,     0.9375,          0,
                 1.0000,     0.9219,          0,
                 1.0000,     0.9062,          0,
                 1.0000,     0.8906,          0,
                 1.0000,     0.8750,          0,
                 1.0000,     0.8594,          0,
                 1.0000,     0.8438,          0,
                 1.0000,     0.8281,          0,
                 1.0000,     0.8125,          0,
                 1.0000,     0.7969,          0,
                 1.0000,     0.7812,          0,
                 1.0000,     0.7656,          0,
                 1.0000,     0.7500,          0,
                 1.0000,     0.7344,          0,
                 1.0000,     0.7188,          0,
                 1.0000,     0.7031,          0,
                 1.0000,     0.6875,          0,
                 1.0000,     0.6719,          0,
                 1.0000,     0.6562,          0,
                 1.0000,     0.6406,          0,
                 1.0000,     0.6250,          0,
                 1.0000,     0.6094,          0,
                 1.0000,     0.5938,          0,
                 1.0000,     0.5781,          0,
                 1.0000,     0.5625,          0,
                 1.0000,     0.5469,          0,
                 1.0000,     0.5312,          0,
                 1.0000,     0.5156,          0,
                 1.0000,     0.5000,          0,
                 1.0000,     0.4844,          0,
                 1.0000,     0.4688,          0,
                 1.0000,     0.4531,          0,
                 1.0000,     0.4375,          0,
                 1.0000,     0.4219,          0,
                 1.0000,     0.4062,          0,
                 1.0000,     0.3906,          0,
                 1.0000,     0.3750,          0,
                 1.0000,     0.3594,          0,
                 1.0000,     0.3438,          0,
                 1.0000,     0.3281,          0,
                 1.0000,     0.3125,          0,
                 1.0000,     0.2969,          0,
                 1.0000,     0.2812,          0,
                 1.0000,     0.2656,          0,
                 1.0000,     0.2500,          0,
                 1.0000,     0.2344,          0,
                 1.0000,     0.2188,          0,
                 1.0000,     0.2031,          0,
                 1.0000,     0.1875,          0,
                 1.0000,     0.1719,          0,
                 1.0000,     0.1562,          0,
                 1.0000,     0.1406,          0,
                 1.0000,     0.1250,          0,
                 1.0000,     0.1094,          0,
                 1.0000,     0.0938,          0,
                 1.0000,     0.0781,          0,
                 1.0000,     0.0625,          0,
                 1.0000,     0.0469,          0,
                 1.0000,     0.0312,          0,
                 1.0000,     0.0156,          0,
                 1.0000,          0,          0,
                 0.9844,          0,          0,
                 0.9688,          0,          0,
                 0.9531,          0,          0,
                 0.9375,          0,          0,
                 0.9219,          0,          0,
                 0.9062,          0,          0,
                 0.8906,          0,          0,
                 0.8750,          0,          0,
                 0.8594,          0,          0,
                 0.8438,          0,          0,
                 0.8281,          0,          0,
                 0.8125,          0,          0,
                 0.7969,          0,          0,
                 0.7812,          0,          0,
                 0.7656,          0,          0,
                 0.7500,          0,          0,
                 0.7344,          0,          0,
                 0.7188,          0,          0,
                 0.7031,          0,          0,
                 0.6875,          0,          0,
                 0.6719,          0,          0,
                 0.6562,          0,          0,
                 0.6406,          0,          0,
                 0.6250,          0,          0,
                 0.6094,          0,          0,
                 0.5938,          0,          0,
                 0.5781,          0,          0,
                 0.5625,          0,          0,
                 0.5469,          0,          0,
                 0.5312,          0,          0,
                 0.5156,          0,          0,
                 0.5000,          0,          0 ]

        full_cmap     = (np.asarray(full_cmap) * 255).astype(np.uint8)
        full_cmap     = np.reshape(full_cmap, [-1, 3])
        step          = full_cmap.shape[0]//label_num
        idxs          = np.arange(0, full_cmap.shape[0], step)
        self.cmap     = full_cmap[idxs, :]
        self.inv_cmap = defaultdict()
        for i in range(label_num):
            r, g, b = self.cmap[i, :]
            inv_key = (((r<<8) + g)<<8) + b
            self.inv_cmap[inv_key] = i


class SaveTool(object):
    '''
    save image or visualize image
    '''
    def __init__(self, label_palette=None, range_palette=None, margin=3):
        self.margin = margin
        if label_palette is None:
            self.label_palette = np.reshape(ColorMap(label_num=256).cmap,[-1])
        else:
            self.label_palette  = label_palette
        if range_palette is None:
            self.range_palette = np.reshape(ColorMap('jet', label_num=256).cmap,[-1])
        else:
            self.range_palette  = range_palette

    def _colorize_mask(self, mask, mode='range'):
        new_mask = PilImage.fromarray(mask.astype(np.uint8)).convert('P')
        palette = self.range_palette if mode=='range' else self.label_palette
        new_mask.putpalette(palette)
        return new_mask

    def save_group_pilImage_RGB(self, images,
                                      palettes=None,
                                      texts=None,
                                      nr=1,
                                      nc=1,
                                      resize=None,
                                      autoScale=True,
                                      fontsize=18,
                                      save_path='dummy.png'):
        '''
        Args: images -- list of arrays in size [ht, wd] or [ht, wd , 3]
              palettes -- list of str indicates the palette to use, 'RGB' | 'Label' | 'Range'
              texts  -- list of str to be show on the sub-grid image
              nr/nc  -- int
              resize -- if not None, (ht, wd) to resize all given images.
              autoScale -- scale range image if 'true'
        '''
        if not isinstance(images, list):
            images = [images]

        if resize is not None:
            if isinstance(resize, list):
                resize = resize[0]
            if images[0].shape[0]>images[0].shape[1]:
                ht, wd = (images[0].shape[0]*resize)//images[0].shape[1], resize
            else:
                ht, wd = resize, (images[0].shape[1]*resize)//images[0].shape[0]
        else:
            ht, wd = images[0].shape[:2]

        pil_wd = nc * (wd + self.margin) - self.margin
        pil_ht = nr * (ht + self.margin) - self.margin
        save_img   = PilImage.new('RGB', size = (pil_wd, pil_ht), color=(255,255,255,0))
        if texts is not None:
            draw_img = PilImageDraw.Draw(save_img)

        # images
        for k, img in enumerate(images):
            if img is None:
                continue

            rk, ck   = k//nc, k%nc
            pwd, pht = ck*(wd+self.margin), rk*(ht+self.margin)

            # image
            if palettes[k].lower() == 'rgb':
                if resize is not None:
                    img = cv2.resize(img, (wd, ht))
                pil_img = PilImage.fromarray(img.astype(np.uint8))
            else:
                if resize is not None:
                    img = cv2.resize(img, (wd, ht), interpolation=cv2.INTER_NEAREST)
                if palettes[k].lower() == 'range':
                    if autoScale:
                        img = (img*255.)/(img.max() + 0.01)
                pil_img = self._colorize_mask(img.astype(np.uint8), palettes[k].lower())
            save_img.paste(pil_img, (pwd, pht))

            # text
            if texts is not None and texts[k] is not None:
                text = texts[k]
                if not isinstance(text, list):
                    text = [text]
                color = [(200, 200, 200), (180, 30, 150)]
                for tk in range(len(text)):
                    draw_img.text((pwd+10, pht+10+20*tk),
                            text[tk],
                            fill=color[tk%2],
                            font=ImageFont.truetype(FredokaOne, size=fontsize))

        save_img.save(save_path)

    def save_single_pilImage_gray(self, image,
                                        palette='range',
                                        resize=None,
                                        autoScale=True,
                                        save_path='./dummy.png'):
        '''
        @Param: image -- image in size [ht, wd]
                palette -- range | label, color map to use.
                resize -- int, size to save the iamge
                autoScale -- if palette=range, scale the value in image to 0~255.
        '''
        if resize is not None:
            if isinstance(resize, list):
                resize = resize[0]
            if image.shape[0]>image.shape[1]:
                ht, wd = (image.shape[0]*resize)//image.shape[1], resize
            else:
                ht, wd = resize, (image.shape[1]*resize)//image.shape[0]
            image = cv2.resize(image, (wd, ht), interpolation=cv2.INTER_NEAREST)
        else:
            ht, wd = image.shape[:2]

        save_img = PilImage.new('P', (wd, ht))
        if palette.lower() == 'range':
            if autoScale:
                image = (image*255.)/(image.max() + 0.01)
            save_img.putpalette(self.range_palette)
        else:
            save_img.putpalette(self.label_palette)
        save_img.paste(PilImage.fromarray(image.astype(np.uint8)), (0,0))
        save_img.save(save_path)

    def save_multiple_pilImage_gray(self, images, palette='range',save_path='./dummy_png'):
        '''
        @Func: this function is for saving a group of masks as single image file. To read it:
                ** I = smisc.imread(fname, mode='P')
                ** masks = np.reshape(I, [-1, ht, wd])

        @Param: images -- list of image in size [ht, wd] or array in size [N, ht, wd]
                palette -- range | label, color map to use.
        '''
        if isinstance(images, list):
            ht, wd = images[0].shape
            saveI = np.reshape(np.stack(images, axis=0), [-1, wd])
        else:
            ht, wd = images.shape[:2]
            saveI = np.reshape(images, [-1, wd])

        self.save_single_pilImage_gray(saveI,
                                       palette=palette,
                                       autoScale=False,
                                       save_path=save_path)


    def save_bgrI(self, bgrI, save_path):
        cv2.imwrite(save_path, bgrI)

    def convert_gray_to_rgb(self, image, mode='range'):
        palette = self.range_palette if mode=='range' else self.label_palette
        cmap = np.reshape(np.asarray(palette), (-1, 3))

        ht, wd = image.shape
        rgbI = np.zeros([ht, wd, 3], dtype=np.uint8)
        rgbI[:,:,0] = cmap[inI, 0]
        rgbI[:,:,1] = cmap[inI, 1]
        rgbI[:,:,2] = cmap[inI, 2]

        return rgbI


class OverlayDetectionBGR(object):
    """
    this calss visualize instance segmentation | object detection | semantic segmentation with
        cv2 to overlay masks | boxes | texts on the image.
    """
    def __init__(self, class_names=None):
        self.class_names  = class_names
        self.palette = np.asarray([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    def compute_colors_for_labels(self, classes, random=False):
        """
        Simple function that adds fixed colors depending on the class
        @Param: classes -- list of int
        """
        if not random:
            colors = np.asarray(classes)[:, None] * self.palette
            colors = (colors % 255).astype("uint8")
        else:
            colors = (np.random.rand(len(classes), 3) * 255).astype("uint8")
        return colors

    def overlay_boxes(self, bgrI, boxes, classes):
        '''
        @Param: bgrI -- [ht, wd, 3] in BGR mode
                boxes -- list ob bbox for [x0,y0,x1,y1]
                classes -- list of category id
        '''
        colors = self.compute_colors_for_labels(classes)
        for box, color in zip(boxes, colors):
            top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
            bgrI = cv2.rectangle(bgrI, top_left, bottom_right, tuple(color.tolist()), 1)

        if isinstance(bgrI, np.ndarray):
            return bgrI
        else:
            return bgrI.get()

    def overlay_mask(self, bgrI, masks, classes=None, show_mask=True,
                           random_color=False, img_ratio=0.6, mask_ratio=0.6):
        '''
        @Param: bgrI -- [ht, wd, 3] in BGR mode
                masks -- list of binary mask array in [ht, wd]
        '''
        if classes is None:
            random_color, classes = True, [0]*len(masks)
        colors = self.compute_colors_for_labels(classes, random=random_color)

        for mask, color in zip(masks, colors):
            contours, hierarchy = findContours(
                mask[:,:, None], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

            if not show_mask:
                bgrI = cv2.drawContours(bgrI, contours, -1, color, 3)
            else:
                tmp = bgrI[mask>0,:]*0.6 + np.asarray(color)*0.6
                bgrI[mask>0, :] = np.minimum(tmp, 255)
                bgrI = cv2.drawContours(bgrI, contours, -1, (255, 255, 255), 1)
        return bgrI

    def overlay_class_names_capbox(self, bgrI, classes, boxes, scores=None):
        '''
        @Param: bgrI -- [ht, wd, 3] in BGR mode
                boxes -- list ob bbox for [x0,y0,x1,y1]
                classes -- list of category id
                scores -- if not None, the confidence score for each det
        '''
        overlay_score = True
        if scores is None:
            overlay_score = False
            scores = [1.0]*len(classes)

        class_names = [self.class_names[k] for k in classes]
        colors = self.compute_colors_for_labels(classes)
        for box, score, label, color in zip(boxes, scores, class_names, colors):
            x, y = box[:2]
            text_str = "{}: {:.2f}".format(label, score) if overlay_score else "{}".format(label)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.4
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            bgrI = cv2.rectangle(bgrI,
                                (x, y),
                                (x + text_w, y - text_h - 4),
                                tuple(color.tolist()), -1)

            text_color = [255, 255, 255]
            cv2.putText(bgrI, text_str, (x, y),
                        font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return bgrI

    def overlay_detection_onBGR(self, bgrI, boxes=None, masks=None, classes=None, scores=None):
        '''
        @Param: bgrI -- [ht, wd, ch] in bgr mode
                boxes -- list of detected boxes, in [x0,y0,x1,y1]
                masks -- list of binary object mask with size [ht, wd]
                classes -- list of int, detected object classes, id corresponding to self.class_names
                scores  - list of float, confidence score for each detection
        '''
        if boxes is not None and classes is not None:
            bgrI = self.overlay_boxes(bgrI, boxes, classes)
        if masks is not None:
            bgrI = self.overlay_mask(bgrI, masks)
        if boxes is not None and classes is not None:
            bgrI = self.overlay_class_names_capbox(bgrI, classes, boxes, scores=scores)
        return bgrI


if __name__=='__main__':
    save_tool = SaveTool()
    ta = np.zeros([32,32])
    ta[:16, :16] = 1
    ta[16:, :16] = 2
    ta[:16, 16:] = 3
    ta[16:, 16:] = 4
    images = [ta, ta]
    palettes = ['Range', 'Label']
    save_tool.save_group_pilImage_RGB(images, palettes, nr=1, nc=2, save_path='dummy.png')

