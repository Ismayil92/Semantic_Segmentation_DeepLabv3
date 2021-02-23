#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import os
import sys
import six
import rospy
import math
import torch 
import logging
import model as network
from torchvision import transforms
from PIL import Image as pilImage
from utils.dataloader_utils import decode_seg_map_sequence,encode_segmap
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image


#Global Variables
bridge = CvBridge()
rgbImg = np.zeros([480,640],dtype = np.uint8)

depthImgFloat = np.zeros([480,640], dtype = np.float32)
depthImgUInt = np.zeros([480,640], dtype= np.uint8)

maskedDepthImg = np.zeros([480,640], dtype = np.uint8)
im_height, im_width = rgbImg.shape[:2]
fx = 0
fy = 0
cx = im_width/2
cy = im_height/2

def imgCallback(data):
    global rgbImg       
    try:
        rgbImg = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
def depthFrameCallback(depthData):
    global depthImgFloat
    global depthImgUInt 
    try: 
         depthImgFloat = bridge.imgmsg_to_cv2(depthData,"32FC1")
         cv_image_array = np.array(depthImgFloat, dtype = np.float32)
         #normalization
         depthImgUInt = cv.normalize(cv_image_array,cv_image_array,0,1,cv.NORM_MINMAX)    
    except CvBridgeError as e:
         print(e)

if __name__ == '__main__':
    count = 0
    rospy.init_node('DL_segmentation', anonymous=True)
    rate = rospy.Rate(10) # 10hz     
    # Subscribe ros image from the robot
    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, imgCallback) 
    depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, depthFrameCallback, queue_size=10)
    # Load the model file of the network
    abs_drc = os.path.abspath(__file__)
    path_to_scripts_ind = abs_drc.find('scripts')
    path_to_project = abs_drc[0:path_to_scripts_ind]
    PATH_TO_CKPT = os.path.join(path_to_project, 'Data/CP_epoch54.pth')   
    NUM_CLASSES = 3
    logging.info("Segmentation started: \n"  
                "red: knob \n "
                "blue: door_handle")
    # Load the model
    model = network.deeplabv3plus_resnet101(num_classes=NUM_CLASSES, output_stride=16)
    network.convert_to_separable_conv(model.classifier)
    logging.info("Loading model {}".format("deeplabv3plus_resnet101"))
    # Set a device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(PATH_TO_CKPT,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    logging.info("Model loaded!")
    model.eval()
    # Processing input images for dl network
    preprocess = transforms.Compose([
    transforms.Resize((513,513),interpolation=2),
    transforms.ToTensor(),   
    ])   
    
    while not rospy.is_shutdown(): 
        img = pilImage.fromarray(rgbImg)     
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda', dtype=torch.float32)
            model.to('cuda')
        with torch.no_grad():
            output = model(input_batch)
            probs = output.data.cpu().numpy()
            output_predictions = np.argmax(probs,axis=1)
            mask = decode_seg_map_sequence(output_predictions,'pascal')
            mask = mask.squeeze(0)
            mask = np.transpose(np.asarray(mask),(1,2,0))
            mask = cv.resize(mask,(640,480))         
            mask = (255*mask).astype(np.uint8)
            mask = cv.cvtColor(mask,cv.COLOR_RGB2GRAY)
            ret1, mask = cv.threshold(mask,10,255,cv.THRESH_BINARY)
            # Mask depth image     
            maskedDepthImg = cv.bitwise_and(depthImgUInt,depthImgUInt,mask=mask)
            # Now depth pixels should be converted to pointcloud  
            
            
            
            # Visualization
            #cv.namedWindow('rgb')
            cv.imshow('rgb', rgbImg)
            #cv.namedWindow('depth_map')
            cv.imshow('depth_map',depthImgUInt)
            #cv.namedWindow('prediction')
            cv.imshow('prediction', mask)
            #cv.namedWindow('masked depth image')
            cv.imshow('masked depth image', maskedDepthImg)
            cv.waitKey(3)         

        # Deleting dynamic memories
        rate.sleep()
    cv.release()
    cv.destroyAllWindows()
