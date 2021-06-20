import os
from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image as PilImage
# dt = np.load('/home/gulsh/mcs_opics/masks-occluder/eval4dataset-3/shp_sqc_0052_02_A1_debug.p', allow_pickle=True)
# print(len(dt), dt.keys(), len(dt['images']))
# # img = Image.fromarray(dt['images'][32], 'RGB')
# # img.save('/home/gulsh/mcs_opics/tracker/training_data/my.png')
# # img.show()
# img = dt['images'][8].reshape(50,50,3)
# print(dt['textures'])
# plt.imshow(img)
# plt.savefig('/home/gulsh/mcs_opics/masks-occluder/myy.png')
# plt.show()
from save_tool import SaveTool
imgSaver = SaveTool()

def check_data(data):
    
    print(len(data[1]))
    
    for i in range(len(data['images'][:2])):
        rgb_img = data['images'][i].reshape(50,50,3)
    # print(data['shapes'][1500], data['textures'][1500])
        plt.imshow(rgb_img)
        plt.savefig(f'/home/gulsh/mcs_opics/masks-occluder/evalTest-1/rgb_{i}.png')
        
    for i in range(len(data['depth'][:2])):
        depth = data['depth'][i]
        imgSaver.save_single_pilImage_gray(depth, 'range',
             save_path=os.path.join('/home/gulsh/mcs_opics/masks-occluder/evalTest-1', 'depth_'+f'{i}'+'.png'))


# check_data(pickle.load(open('/home/gulsh/mcs_opics/masks-occluder/training_dataset_rgbd.p', 'rb')))

# def convert_data(data):
#     np.savez('/home/gulsh/mcs_opics/masks-occluder/rgbd.npz',data, allow_pickle =  True)
#     print('DONE!')
# convert_data(pickle.load(open('/home/gulsh/mcs_opics/masks-occluder/training_dataset_rgbd.p', 'rb')))

data = np.load('/home/gulsh/mcs_opics/masks-occluder/rgbd.npy', allow_pickle=True)
check_data(data)




# def see_data(path):
    
#     datas = pickle.load(open(path, 'rb'))
#     # print(dataPaths[:3])
#     print(datas[:5])
#     training_data = {'images': [], 'shapes': [], 'materials': [], 'textures': [], 'depth':[]}
#     texturecnt, depthMismatchCnt = 0,0
#     for data in tqdm(sorted(datas)):
#         # dt = np.load(path+'/'+data, allow_pickle=True)
#         if len(data['textures'][0]) == 0:
#             texturecnt+=1
#             continue
#         elif len(data['images'])!=len(data['depth']):
#             depthMismatchCnt+=1
#             continue
#         else:
#             #print(len(dt), dt['textures'])
#             for i in range(len(data['images'])):
#                 # training_data['images'].append(dt['images'][i])
#                 # training_data['shapes'].append(dt['shapes'][i])
#                 # training_data['materials'].append(dt['materials'][i])
#                 # training_data['depth'].append(dt['depth'][i])
#                 # #print(dt['textures'][i],len(dt['textures'][i]))
#                 # if len(data['textures'][i])>1:
#                 #     color = "+"
#                 #     training_data['textures'].append(color.join(dt['textures'][i]))
#                 # else:
#                 #     training_data['textures'].append(dt['textures'][i])
            
#     # print(len(training_data['images']), len(training_data['depth']), texturecnt, depthMismatchCnt)
#     # img = training_data['images'][-10].reshape(50,50,3)
#     # plt.imshow(img)
#     # plt.savefig('/home/gulsh/mcs_opics/masks-occluder/myy.png')
#     # plt.show()
#     # pickle.dump(training_data, open(os.path.join('/home/gulsh/mcs_opics/masks-occluder', 'training_dataset_rgbd.p'), 'wb'))
#     # for i in range(len(training_data['depth'][:250])):
#     #     rgb_img = training_data['images'][i].reshape(50,50,3)
#     # # print(data['shapes'][1500], data['textures'][1500])
#     #     plt.imshow(rgb_img)
#     #     plt.savefig(f'/home/gulsh/mcs_opics/masks-occluder/evalTest/rgb_{i}.png')

#     #     depth = training_data['depth'][i]
#     #     imgSaver.save_single_pilImage_gray(depth, 'range',
#     #          save_path=os.path.join('/home/gulsh/mcs_opics/masks-occluder/evalTest', 'depth_'+f'{i}'+'.png'))


# # see_data('/home/gulsh/mcs_opics/masks-occluder/training_dataset_rgbd.p')