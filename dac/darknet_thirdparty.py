from ctypes import *
import math
import random
import cv2
import time

class network():
    def sample(probs):
        s = sum(probs)
        probs = [a/s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs)-1

    def c_array(ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    class BOX(Structure):
        _fields_ = [("x", c_float),
                    ("y", c_float),
                    ("w", c_float),
                    ("h", c_float)]

    class IMAGE(Structure):
        _fields_ = [("w", c_int),
                    ("h", c_int),
                    ("c", c_int),
                    ("data", POINTER(c_float))]

    class MATRIX(Structure):
        _fields_ = [("rows", c_int),
                    ("cols", c_int),
                    ("vals", POINTER(POINTER(c_float)))]
    
    class METADATA(Structure):
        _fields_ = [("classes", c_int),
                    ("names", POINTER(c_char_p))]

    #lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    make_boxes = lib.make_boxes
    make_boxes.argtypes = [c_void_p]
    make_boxes.restype = POINTER(BOX)

    make_boxes_array = lib.make_boxes_array
    make_boxes_array.argtypes = [c_void_p, c_int]
    make_boxes_array.restype = POINTER(POINTER(BOX))

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    free_ptrs_batch = lib.free_ptrs_batch
    free_ptrs_batch.argtypes = [POINTER(c_void_p), c_int, c_int]

    num_boxes = lib.num_boxes
    num_boxes.argtypes = [c_void_p]
    num_boxes.restype = c_int

    make_probs = lib.make_probs
    make_probs.argtypes = [c_void_p]
    make_probs.restype = POINTER(POINTER(c_float))

    make_probs_array = lib.make_probs_array
    make_probs_array.argtypes = [c_void_p, c_int]
    make_probs_array.restype = POINTER(POINTER(POINTER(c_float)))

    make_index_array = lib.make_index_array
    make_index_array.argtypes = [c_int]
    make_index_array.restype = POINTER(c_int)

    make_class_array = lib.make_class_array
    make_class_array.argtypes = [c_int]
    make_class_array.restype = POINTER(c_int)

    detect = lib.network_predict
    detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    free_matrix = lib.free_matrix
    free_matrix.argtypes = [MATRIX]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    ndarray_image = lib.ndarray_to_image
    ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
    ndarray_image.restype = IMAGE

    ndarray_matrix = lib.ndarray_to_matrix
    ndarray_matrix.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
    ndarray_matrix.restype = MATRIX

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    network_detect = lib.network_detect
    network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

    network_detect_matrix = lib.network_detect_batch
    network_detect_matrix.argtypes = [c_void_p, MATRIX, c_float, c_float, c_float, POINTER(POINTER(BOX)), POINTER(POINTER(POINTER(c_float))), POINTER(c_int), POINTER(c_int)]

    def classify(net, meta, im):
        out = predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, net, meta, image, thresh=.001, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        boxes = self.make_boxes(net)
        probs = self.make_probs(net)
        num =   self.num_boxes(net)
        self.network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if probs[j][i] > 0:
                    res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res

    def detect_np(self, net, meta, np_img, thresh=.1, hier_thresh=.5, nms=.45):
        im = self.nparray_to_image(np_img)
        boxes = self.make_boxes(net)
        probs = self.make_probs(net)
        num = self.num_boxes(net)
        self.network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if probs[j][i] > 0:
                    res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res
        
    def detect_np_batch(self, net, meta, np_array, batchNum, thresh=.2, hier_thresh=.5, nms=.8):
        mat = self.nparray_to_matrix(np_array)
        boxes = self.make_boxes_array(net, batchNum)
        probs = self.make_probs_array(net, batchNum)
        num = self.num_boxes(net)
        idx = self.make_index_array(batchNum)
        cls = self.make_class_array(batchNum)
        self.network_detect_matrix(net, mat, thresh, hier_thresh, nms, boxes, probs, idx, cls)
        resBatch = []
        for k in range(batchNum):
            res = ((meta.names[cls[k]], probs[k][idx[k]][cls[k]], (boxes[k][idx[k]].x, boxes[k][idx[k]].y, boxes[k][idx[k]].w, boxes[k][idx[k]].h)))
            if res:
                resBatch.append(res)
            else:
                resBatch.append(('NoName', 0.0, (0,0,0,0)))

        self.free_matrix(mat)
        #self.free_ptrs(cast(idx, POINTER(c_void_p)), batchNum)
        #self.free_ptrs(cast(cls, POINTER(c_void_p)), batchNum)
        self.free_ptrs_batch(cast(probs, POINTER(c_void_p)), batchNum, num)
        self.free_ptrs(cast(boxes, POINTER(c_void_p)), batchNum)
        return resBatch

    def nparray_to_image(self, img):
        data = img.ctypes.data_as(POINTER(c_ubyte))
        image = self.ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

        return image

    def nparray_to_matrix(self, array):
        data = array.ctypes.data_as(POINTER(c_ubyte))
        resmatrix = self.ndarray_matrix(data, array.ctypes.shape, array.ctypes.strides)
        data = []
        return resmatrix


if __name__ == "__main__":
    detector = network()
    net = detector.load_net("tiny-yolo.cfg", "tiny-yolo_76000.weights", 0)
    meta = detector.load_meta("obj.data")
    img = cv2.imread('test/images/0002.jpg')
    t_start = time.time()
    r = detector.detect_np(net, meta, img)
    print(time.time() - t_start)
    print(r)

