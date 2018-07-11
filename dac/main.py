## This program is for DAC HDC contest ######
## 2017/11/22
## xxu8@nd.edu
## University of Notre Dame

import procfunc
import math
import numpy as np
import time
#### !!!! you can import any package needed for your program ######

if __name__ == "__main__":
    teamName = 'USL-TAMU'
    DAC = './test2'
    [imgDir, resultDir, timeDir, xmlDir, myXmlDir, allTimeFile] = procfunc.setupDir(DAC, teamName)

    [allImageName, imageNum] = procfunc.getImageNames(imgDir)
    ### process all the images in batch
    batchNumDiskToDram = 128 ## the # of images read from disk to DRAM in one time
    batchNumDramToGPU  = 16 ## the # of images read from DRAM to GPU in one time for batch processing on the GPU
    imageReadTime = math.ceil(float(imageNum)/float(batchNumDiskToDram))
    imageProcTimeEachRead = math.ceil(float(batchNumDiskToDram)/float(batchNumDramToGPU))
    resultRectangle = np.zeros((imageNum, 4)) ## store all the results about tracking accuracy
    resultRunTime = 0


if __name__=="__main__":
    for i in range(int(imageReadTime)):
        ImageDramBatch = procfunc.readImagesBatch(imgDir,allImageName, imageNum, i, batchNumDiskToDram)
        for j in range(int(imageProcTimeEachRead)):
            start = j*batchNumDramToGPU
            end = start + batchNumDramToGPU
            if end > len(ImageDramBatch):
                end = len(ImageDramBatch)
                if end < start:
                    break
            inputImageData = ImageDramBatch[start:end, :]
            time_start=time.time()
            resultRectangle[i * batchNumDiskToDram + start:i * batchNumDiskToDram + end, :] = procfunc.detectionAndTracking(inputImageData, end-start)
            time_end = time.time()
            resultRunTime = resultRunTime + time_end-time_start
        
    procfunc.storeResultsToXML(resultRectangle, allImageName, myXmlDir)
    procfunc.write(imageNum,resultRunTime,teamName, allTimeFile)
