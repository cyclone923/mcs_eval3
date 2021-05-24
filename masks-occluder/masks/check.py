import os
from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
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

def see_data(path):
    dataPaths = os.listdir(path)
    # print(dataPaths[:3])
    training_data = {'images': [], 'shapes': [], 'materials': [], 'textures': []}
    cnt = 0
    for data in tqdm(sorted(dataPaths)):
        dt = np.load(path+'/'+data, allow_pickle=True)
        if len(dt['textures'][0]) == 0:
            cnt+=1
            continue
        else:
            #print(len(dt), dt['textures'])
            for i in range(len(dt['images'])):
                training_data['images'].append(dt['images'][i])
                training_data['shapes'].append(dt['shapes'][i])
                training_data['materials'].append(dt['materials'][i])
                #print(dt['textures'][i],len(dt['textures'][i]))
                if len(dt['textures'][i])>1:
                    color = "+"
                    training_data['textures'].append(color.join(dt['textures'][i]))
                else:
                    training_data['textures'].append(dt['textures'][i])
            
    print(len(training_data['images']))
    # img = training_data['images'][-10].reshape(50,50,3)
    # plt.imshow(img)
    # plt.savefig('/home/gulsh/mcs_opics/masks-occluder/myy.png')
    # plt.show()
    pickle.dump(training_data, open(os.path.join('/home/gulsh/mcs_opics/masks-occluder', 'training_dataset.p'), 'wb'))

# see_data('/home/gulsh/mcs_opics/masks-occluder/eval4TrainingDataset-Merged')
def check_data(data):
    #print(len(data),len(data['images']))
    img = data['images'][1500].reshape(50,50,3)
    print(data['shapes'][1500], data['textures'][1500])
    # plt.imshow(img)
    # plt.savefig('/home/gulsh/mcs_opics/masks-occluder/myy.png')
    # plt.show()
    color = np.array(data['textures'], dtype=str)
    print()
    print(len(color))


check_data(pickle.load(open('/home/gulsh/mcs_opics/masks-occluder/training_dataset.p', 'rb')))
