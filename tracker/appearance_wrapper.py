# from tracker.siameseAppearance import DEBUG
from tracker import appearence
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .model import SmallSiamese
from PIL import Image
import PIL
from torchvision import datasets, transforms
from rich.console import Console
import numpy as np
from matplotlib import pyplot as plt
from .utils import draw_bounding_boxes
DEBUG = True

console = Console()
transform = transforms.Compose([
#             transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
sig = nn.Sigmoid()
def process_inputs(inputs):
    inputs = transform(inputs)             
    inputs = inputs.unsqueeze(0).cuda()

def evaluate_appearance(img_a, img_b, transform):
    model = SmallSiamese().cuda()
    state_dict = torch.load('/home/gulsh/mcs_opics/tracker/model_iter-3999.pth')
    model.load_state_dict(state_dict)    
    model.eval()
    sig = nn.Sigmoid()
    torch.cuda.set_device(0)
    with torch.no_grad():
        img1 = PIL.Image.fromarray(np.uint8(img_a)[:,:,::-1])
        img2 = PIL.Image.fromarray(np.uint8(img_b)[:,:,::-1])
        img1 = transform(img1).unsqueeze(0).cuda()
        img2 = transform(img2).unsqueeze(0).cuda()
        # print(img1.size(), img2.size())
        pred = sig(model(img1, img2))
    return pred[0].cpu().numpy()        

def object_appearance_match(image, scene_name,frame_num,objects_info, device, level):
    
    transform = transforms.Compose([
           transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    for obj_key in [k for k in objects_info.keys() if objects_info[k]['visible']]:
        top_x, top_y, bottom_x, bottom_y = objects_info[obj_key]['bounding_box']
        obj_current_image = image.crop((top_y, top_x, bottom_y, bottom_x))
        image_area = np.prod(obj_current_image.size)
        #wait till object is in clear view
        if 'base_image' not in objects_info[obj_key].keys() or \
            (len(objects_info[obj_key]['position_history'])<3 and image_area>=objects_info[obj_key]['base_image']['image_area']):
            objects_info[obj_key]['base_image'] = dict()
            objects_info[obj_key]['base_image']['image_area'] = np.prod(obj_current_image.size)
            objects_info[obj_key]['base_image']['prev_img'] = [obj_current_image,]
            # obj_current_image.save('/home/gulsh/mcs_opics/my1' + str(obj_key)+'.png')
        else:
            objects_info[obj_key]['base_image']['image_area'] = np.prod(obj_current_image.size)
            objects_info[obj_key]['base_image']['prev_img'].append(obj_current_image)
        
        objFrameNum = len(objects_info[obj_key]['base_image']['prev_img'])
        

        # if we have enough object images?
        if len(objects_info[obj_key]['base_image']['prev_img'])>5:
            previousObjs = objects_info[obj_key]['base_image']['prev_img'][:6]
            current_obj = objects_info[obj_key]['base_image']['prev_img'][-1]
            # objects_info[obj_key]['base_image']['prev_img'][-2].save('/home/gulsh/mcs_opics/my-2' + str(obj_key)+'.png')
            meanMatch = []
            # doing frame by frame object match between cuurent object [-1] and all objects seen before that
            
            for i in range(len(previousObjs)-1):
                matchScore  = evaluate_appearance(current_obj,previousObjs[i], transform)
                console.print(f"[green]Match score: {matchScore[0]} for Object key: {obj_key} between {objFrameNum} and {objFrameNum-5+i} frame[/green]")
                meanMatch.append(matchScore[0])
                if 'appearance' not in objects_info[obj_key].keys():
                    # objFrameNum = len(objects_info[obj_key]['base_image']['prev_img'])
                    objects_info[obj_key]['appearance'] = {}
                    objects_info[obj_key]['appearance']['matchScore'] = []
                    objects_info[obj_key]['appearance']['mismatch_count'] = 0
                    objects_info[obj_key]['appearance']['match'] = True
                    objects_info[obj_key]['appearance']['matchScore'].append([objFrameNum, objFrameNum-5+i,matchScore[0]])
                else:
                    objects_info[obj_key]['appearance']['matchScore'].append([objFrameNum, objFrameNum-5+i,matchScore[0]])
                
                if DEBUG:
                    fig = plt.figure(figsize=(8,8))
                    fig.add_subplot(1,2,1)
                    # img1 = PIL.Image.fromarray(np.uint8(current_obj)[:,:,::-1])
                    # img1 = transform(img1).unsqueeze(0).reshape(3,32,32)
                    img1 = PIL.Image.fromarray(np.array(current_obj))
                    img1 = img1.resize(size=(48,48))
                    
                    # img1 = img1.permute(1,2,0)
                    # img1 = img1.reshape(32,32,3)
                    # img1 = np.swapaxes(img1, 0,2)
                    plt.imshow(img1)
                    plt.text(48,58,str(matchScore), fontsize='large', color='black')
                    fig.add_subplot(1,2,2)
                    # obj2 = PIL.Image.fromarray(np.uint8(objects_info[obj_key]['base_image']['prev_img'][i])[:,:,::-1])
                    # obj2 = transform(obj2).unsqueeze(0).reshape(3,32,32)
                    img2 = PIL.Image.fromarray(np.array(objects_info[obj_key]['base_image']['prev_img'][i]))
                    img2 = img2.resize(size=(48,48))
                    # obj2 = obj2.permute(1,2,0)
                    # obj2 = obj2.reshape(32,32,3)
                    # obj2 = np.swapaxes(obj2,0,2)
                    plt.imshow(img2)
                    
                    plt.savefig(f'/home/gulsh/mcs_opics/{scene_name}/{frame_num}-{objFrameNum}-{i+1}.png')


                # What would be a good threshold ?
                if matchScore<0.5:
                    console.print('[red]Mismatch detected[/red]')
                    objects_info[obj_key]['appearance']['mismatch_count'] +=1

                else:
                    objects_info[obj_key]['appearance']['match'] = False

            console.print(f"[yellow]Mean match rate: [/yellow] for {obj_key} is: {np.mean(meanMatch)}")
    return objects_info
        # current_obj = objects_info[obj_key]['base_image']['prev_img'][-1]
        # if DEBUG:
        #     for i in range(len(objects_info[obj_key]['base_image']['prev_img'])-1):
        #         if i<5:
        #             fig = plt.figure(figsize=(4,2))
        #             fig.add_subplot(1,2,1)
        #             plt.imshow(current_obj)
        #             fig.add_subplot(1,2,2)
        #             plt.imshow(objects_info[obj_key]['base_image']['prev_img'][i])
        #             plt.text(objFrameNum+i,matchScore,str(matchScore))
        #             plt.savefig(f'/home/gulsh/mcs_opics/{scene_name}/{frame_num}/_{objFrameNum}-{i}.png')


        # img = draw_bounding_boxes(image, track_info['objects'])

        # import matplotlib.pyplot as plt
        # import matplotlib.gridspec as gridspec
        # if 'appearance' in objects_info[obj_key].keys():
        #     scores = [i for i in objects_info[obj_key]['appearance']['matchScore'][i][2]]
        #     print(scores)

        #     fig = plt.figure(figsize=(10, 8))
        #     outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        #     for i in range(3):
        #         inner = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[i],
        #                                                 wspace=0.1, hspace=0.1)
        #         row     = 0
        #         col     = 0
        #         maxCol  = 2

        #         for score in scores:
        #             ax = plt.Subplot(fig, inner[row,col])
        #             t= ax.text(0.5,0.5, 'score=%d\n' % (score))
        #             ax.set_xticks([])
        #             ax.set_yticks([])
        #             t.set_ha('right')
        #             fig.add_subplot(ax)
        #             col += 1
        #             if col == maxCol:
        #                 col = 0
        #                 row += 1
        #     plt.show()
    
