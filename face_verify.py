import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import pdb
from data.data_pipe import get_val_pair
import numpy as np
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-f", "--file", help="filename",required=True)
    parser.add_argument("-csv", "--csv", help="csv filename",required=True)
    args = parser.parse_args()

    conf = get_config(training=False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, inference=True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        #learner.load_state(conf, 'cpu_final.pth', True, True)
        learner.load_state(conf, '2019-03-19-20-18_accuracy:0.9307142857142857_step:300234_None.pth', False, True)
    else:
        #learner.load_state(conf, 'final.pth', True, True)
        learner.load_state(conf, '2019-03-19-20-18_accuracy:0.9307142857142857_step:300234_None.pth', False, True)
    learner.model.eval()
    print('learner loaded')
   
    '''vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, vgg2_fp, vgg2_fp_issame, nrof_folds=10, tta=True)
    print('vgg2_fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    exit(0)'''
    df = pd.read_csv(args.csv)
    imlst = df.groupby('classnm')['imgfile'].apply(lambda x: x.tolist()).to_dict()
    if args.update:
        targets, ftoid, idinfo = prepare_facebank(conf, imlst, learner.model, mtcnn, tta = args.tta,save = True)
        print('facebank updated')
    else:
        targets, ftoid, idinfo = load_facebank(conf)
        print('facebank loaded')
    faces = []
    predfns = []
    with open(args.file) as f:
        imgfiles = list(map(str.strip,f.readlines()))
        for imgfn in imgfiles:
            try:
                face = Image.open(imgfn)
            except:
                print('cannot open query image file {}'.format(imgfn))
                continue
            try:
                face = mtcnn.align(face)
            except:
                print('mtcnn failed for {}'.format(imgfn))
                face = face.resize((112,112), Image.ANTIALIAS)
            #data = np.array((cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2GRAY),)*3).T
            #face = Image.fromarray(data)
            data = np.array(face)
            face = Image.fromarray(data[:,:,::-1])
            faces.append(face)
    results, score ,d = learner.infer(conf, faces, targets, args.tta)
    print (score)
    for idx,imgfn in enumerate(imgfiles):
        i = results[idx]
        print ("For {} found face  {}".format(imgfn,"Unknown" if i == -1 else idinfo[i][1]))
        print (d[idx], d[idx][i])
        print (score[idx])
        print (idinfo[i])
    '''
    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    if args.save:
        video_writer = cv2.VideoWriter(conf.data_path/'recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:            
            try:
#                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice    
                results, score = learner.infer(conf, faces, targets, args.tta)
                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except:
                print('detect error')    
                
            cv2.imshow('face Capture', frame)

        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
    if args.save:
        video_writer.release()
'''
