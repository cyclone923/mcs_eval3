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

class ColorMap:
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
        save_img   = PilImage.new('RGB', (nc * (wd + self.margin) - self.margin,
                                          nr * (ht + self.margin) - self.margin))
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

    def convert_gray_to_rgb(self, image, mode='range'):
        palette = self.range_palette if mode=='range' else self.label_palette
        cmap = np.reshape(np.asarray(palette), (-1, 3))

        ht, wd = image.shape
        rgbI = np.zeros([ht, wd, 3], dtype=np.uint8)
        rgbI[:,:,0] = cmap[inI, 0]
        rgbI[:,:,1] = cmap[inI, 1]
        rgbI[:,:,2] = cmap[inI, 2]

        return rgbI

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

