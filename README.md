![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

#Modifications in this fork:#

1. Batch processing exposed for external interfaces such as Python or C++. [`network_detect_batch()`](https://github.com/saihv/DarkneTX2/blob/master/src/network.c#L559) has been rewritten in a better way. (src/network.c)
2. Small speedups in image resizing and letterboxing operations thorugh the use of pointer arithmetic while traversing the image pixels. (src/image.c)

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
