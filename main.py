import time
import cv2
import os
import numpy as np
import argparse


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-k", "--kernel", type=int, default=41,
	help="size of gaussian blur kernel")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# construct the kernel for the Gaussian blur and initialize whether
# or not we are in "privacy mode"
K = (args["kernel"], args["kernel"])
privacy = True

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


# input_source = "/path/to/video/fileName"
input_source = 0
# output_fileName = "/path/to/video/output.avi"
output_fileName = "output.avi"

cap = cv2.VideoCapture(input_source)
success, frame = cap.read()
vid_writer = cv2.VideoWriter(output_fileName,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))


# background = cv2.imread("download.jpg")
# background =cv2.resize(background, (600, 450))


frame_number = 0

while(cap.isOpened()):
    frame_number += 1    
    t = time.time()
    count = 0
    #Skip frames
    while count< 5:
        count +=1
        success, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    
    if success:
        # construct a blob from the input image and then perform a
        # forward pass of the Mask R-CNN, giving us (1) the bounding
        # box coordinates of the objects in the image along with (2)
        # the pixel-wise segmentation for each specific object
        frame = image_resize(frame, width= 300)
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob)
        (boxes, masks) = net.forward(["detection_out_final",
            "detection_masks"])
        
        print("Time to process : ",time.time() -t)

        # sort the indexes of the bounding boxes in by their corresponding
        # prediction probability (in descending order)
        idxs = np.argsort(boxes[0, 0, :, 2])[::-1]

        # initialize the mask, ROI, and coordinates of the person for the
        # current frame
        mask = None
        roi = None
        coords = None

        # loop over the indexes
        for i in idxs:
            # extract the class ID of the detection along with the
            # confidence (i.e., probability) associated with the
            # prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            # if the detection is not the 'person' class, ignore it
            if LABELS[classID] != "person":
                continue

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image and then compute the width and the
                # height of the bounding box
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                coords = (startX, startY, endX, endY)
                boxW = endX - startX
                boxH = endY - startY

                # extract the pixel-wise segmentation for the object,
                # resize the mask such that it's the same dimensions of
                # the bounding box, and then finally threshold to create
                # a *binary* mask
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                    interpolation=cv2.INTER_LINEAR)
                mask = (mask > args["threshold"])
                

                # extract the ROI and break from the loop (since we make
                # the assumption there is only *one* person in the frame
                # who is also the person with the highest prediction
                # confidence)
                # roi = frame[startY:endY, startX:endX][mask]
                roi = frame[startY:endY, startX:endX]
                break

        # initialize our output frame
        # output = background.copy()
        output = frame.copy()
        alpha = mask.astype(float)
        alpha = cv2.GaussianBlur(alpha, (7,7), 0)
        # cv2.imshow("Video Call", a)

        # if the mask is not None *and* we are in privacy mode, then we
        # know we can apply the mask and ROI to the output image
        if mask is not None and privacy:
            # blur the output frame
            output = cv2.GaussianBlur(output, K, 0)

            # add the ROI to the output frame for only the masked region
            (startX, startY, endX, endY) = coords
           
            # output[startY:endY, startX:endX][mask] = roi
        output[startY:endY, startX:endX] = cv2.multiply(1.0 - alpha,output[startY:endY, startX:endX], dtype = cv2.CV_64F) + cv2.multiply(alpha, roi, dtype = cv2.CV_64F)
        # show the output frame
        cv2.imshow("Video Call", output)

        vid_writer.write(output)

        key = cv2.waitKey(1) & 0xFF

        # if the `p` key was pressed, toggle privacy mode
        if key == ord("p"):
            privacy = not privacy

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break


vid_writer.release()    
cap.release()
cv2.destroyAllWindows()