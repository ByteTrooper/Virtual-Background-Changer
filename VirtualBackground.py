import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
# importing images for using images folder
import os

# using cv2 videocapture it lets us turn on webcam and we store that in capture
capture = cv2.VideoCapture(0)

# assigning image frame size
capture.set(3, 640)
capture.set(4, 488)

# This discriminates the foreground from background by observing the continously changing thing
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

# we will import os first and then take the all images from the folder
listImg = os.listdir("images")
print(listImg)

# Storing all the images in the imageList
imgList = []
for imgPath in listImg:
    # Reading all images for background in the for loop
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)
print(len(imgList))

indexImg = 0

# in resize function we are resizing the background size according to frame size
def resize(dst, img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    # inter_AREA is one of the interpolation method to downscale the images
    resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    return resized

while True:
    success, img = capture.read()
    bg = resize(imgList[indexImg], img)
    imgOut = segmentor.removeBG(img, bg, threshold=0.8)
    # to see the original video and changed one side by side
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    # to see the fps value of video
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))
    print(indexImg)

    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)

    # we use key to change background images
    # if we press "a" it will go to before image
    if key == ord('a'):
       if  indexImg>=0:
            indexImg -= 1
    # if we press "d" it will go to next image
    elif key == ord('d'):
        if indexImg < len(imgList) - 1:
            indexImg += 1
    elif key == ord('q'):
        break