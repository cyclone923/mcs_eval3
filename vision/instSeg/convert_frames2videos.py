import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x:x)

    for i in range(len(files)):
        filename=join(pathIn,files[i])
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    pathIn  = '../mcsvideo3-inter-depth/results/valid/1117_mcsvideo3'
    info_file = '../dataset/mcsvideo3/interaction_scenes/val.txt'
    # pathIn  = '../mcsvideo_voe/results/valid/MC-res50-513'
    # info_file = '../dataset/mcsvideo/VOE_scenes/val.txt'
    fps = 12.0

    with open(info_file) as f:
        fdir_list = [x.strip() for x in f.readlines()]

    for video in fdir_list:
        src_path = join(pathIn, video)
        print(src_path)
        if os.path.exists(src_path):
            dst_path = src_path+'_video.avi'
            videos = [video for f in os.listdir(src_path) if isfile(join(src_path, f))]
            convert_frames_to_video(src_path, dst_path, fps)

if __name__=="__main__":
    main()
