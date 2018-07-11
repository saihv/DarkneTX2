from __future__ import division
from glob import glob
import os
import sys
import xml.etree.ElementTree as ET
import shutil
import math
import random
import cv2

rootFolder = './test2/'

imageFolder = rootFolder + 'images/'
xmlFolder = rootFolder + 'result/xml/USL-TAMU/'

width = 640
height = 360

class BoundingBox(object):
    pass

def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
  # Failed to find "index" occurrence of item.  
    return -1

def GetInt(name, root, index=0):
    return int(float(GetItem(name, root, index)))

if __name__ == '__main__':
    images_list = glob(os.path.join(imageFolder, '*.jpg'))

    totalIoU = 0

    for imagefile in sorted(images_list):
        print(imagefile)
        base = os.path.basename(imagefile)
        image = cv2.imread(imagefile)
        print(xmlFolder + os.path.splitext(base)[0] + '.xml')
        xmldataOP = ET.parse(xmlFolder + os.path.splitext(base)[0] + '.xml').getroot()
        xmldataGT = ET.parse(imageFolder + os.path.splitext(base)[0] + '.xml').getroot()
        
        boxOP = BoundingBox()
        boxOP.xmin = GetInt('xmin', xmldataOP, 0)
        boxOP.ymin = GetInt('ymin', xmldataOP, 0)
        boxOP.xmax = GetInt('xmax', xmldataOP, 0)
        boxOP.ymax = GetInt('ymax', xmldataOP, 0)

        boxGT = BoundingBox()
        boxGT.xmin = GetInt('xmin', xmldataGT, 0)
        boxGT.ymin = GetInt('ymin', xmldataGT, 0)
        boxGT.xmax = GetInt('xmax', xmldataGT, 0)
        boxGT.ymax = GetInt('ymax', xmldataGT, 0)

        cv2.rectangle(image, (boxOP.xmin, boxOP.ymin), (boxOP.xmax, boxOP.ymax), (0, 0, 255), 2)
        cv2.rectangle(image, (boxGT.xmin, boxGT.ymin), (boxGT.xmax, boxGT.ymax), (0, 255, 0), 2)

        xA = max(boxGT.xmin, boxOP.xmin)
        yA = max(boxGT.ymin, boxOP.ymin)
        xB = min(boxGT.xmax, boxOP.xmax)
        yB = min(boxGT.ymax, boxOP.ymax)
        
        intersection = (xB - xA + 1) * (yB - yA + 1)
        
        boxGTArea = (boxGT.xmax - boxGT.xmin + 1) * (boxGT.ymax - boxGT.ymin + 1)
        boxOPArea = (boxOP.xmax - boxOP.xmin + 1) * (boxOP.ymax - boxOP.ymin + 1)
        
        iou = intersection / float(boxGTArea + boxOPArea - intersection)

        if iou < 0:
            iou = 0.00

        totalIoU = totalIoU + iou
        
        cv2.putText(image, 'IoU: ' + str(iou) , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Bounding boxes", image)
        cv2.waitKey(0)

    print ("Average IoU on test dataset: {}".format(totalIoU/len(images_list)))

