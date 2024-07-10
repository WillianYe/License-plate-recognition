import cv2
import numpy as np


def preprocess(gray):
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    # the color space corresponding to blue and green
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])
    # lower_green = np.array([35, 43, 46])
    # upper_green = np.array([77, 255, 255])
    # Convert the BGR image to the HSV color space
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # mask_plate = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_green)
    # Find the corresponding color based on the threshold
    mask_plate = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    mask = cv2.cvtColor(mask_plate, cv2.COLOR_BGR2GRAY)
    # Morphological open operation
    Matrix = np.ones((20, 20), np.uint8)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, Matrix)
    mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, Matrix)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # Different template expansion operation for different size license plate
    if mask.shape[0] >= 2500:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
    elif mask.shape[0] > 1500 and mask.shape[0] < 2500:
        kernel = np.ones((12, 12), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
    else:
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
    return mask


def findPlateNumberRegion(img, a):
    region = []
    # find the contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        # calculate the area
        area = cv2.contourArea(cnt)
        # filter the small area
        if area < 8000:
            continue
        # find the min area
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # calculate the ratio of height and width
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        if ratio > 5 or ratio < 1.05:
            continue
        region.append(box)
    return region


def detect(img, name):
    # prepossess
    dilation = preprocess(img)
    # find the plate region
    region = findPlateNumberRegion(dilation, img)
    '''
    # draw the contours
    for box in region:
        plate=cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 2)
        # image with contours
        cv2.imwrite('locate_result/contours.png',plate)
    '''
    # cut the image
    for i in range(len(region)):
        box = region[i]
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.ma.argsort(ys)
        xs_sorted_index = np.ma.argsort(xs)
        x1 = box[xs_sorted_index[1], 0]
        x2 = box[xs_sorted_index[2], 0]
        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]
        ROI_plate = img[y1:y2, x1:x2]
        # ROI_plate = cv2.resize(ROI_nplate, (94,24))
        cv2.imwrite('locate_result/plate' + name + '_' + str(i) + '.png', ROI_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


names = ['1-1','1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2','3-3']
# names = ['2-1']
for name in names:
    path = 'images/' + name + '.jpg'
    img = cv2.imread(path)
    detect(img, name)
