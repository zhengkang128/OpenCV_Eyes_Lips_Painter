import faceBlendCommon as fbc
import cv2
import time
import numpy as np
import dlib
import os

def createLipsMasks(img, points, openKernelSize = 5, blurKernelSize = 31):
    btm_lips_points = np.concatenate((points[48:55], points[60:65]))
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [btm_lips_points], (255,255, 255))  

    top_lips_points = np.concatenate((points[54:61], np.flip(points[64:68],axis=0)))
    cv2.fillPoly(mask, [top_lips_points], (255,255, 255)) 


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (openKernelSize,openKernelSize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    mask = cv2.GaussianBlur(mask,(blurKernelSize,blurKernelSize),cv2.BORDER_DEFAULT)
    inverseMask = cv2.bitwise_not(mask)
    
    mask = mask.astype(float)/255
    inverseMask = inverseMask.astype(float)/255
    
    return mask.copy(), inverseMask.copy()

def applyLipstick(img, points, input_color = np.array([255,0,0]), brightness = 1.5):  
    mask, inverseMask = createLipsMasks(imDlib, points)
    transformed_color = getColoredTransformed(img, input_color, brightness)

    justLips = cv2.multiply(mask, transformed_color/255)
    justFace = cv2.multiply(inverseMask, imDlib/255)
    out = justLips + justFace
    final_output = (out*255).astype(np.uint8)
    return final_output

def createEyeMasks(img, eyePts, threshold_val = 75,  dilateKernelSize = 4, blurKernelSize = 101):
    mask = np.zeros_like(img) 
    cv2.fillPoly(mask, [eyePts], (255,255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inverse_mask = 255 - mask
    # add inverse mask so background is white. Since non pupil are mostly white pixels (high intensity)
    eye = cv2.bitwise_and(img, img, mask = mask) + cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR) 
    
    r = (eye[:,:,0])
    g = eye[:,:,1]
    b = eye[:,:,2]
    eye_channels = [r,g,b]


    #Iterate through all three channels to perform threshold and determine the 'best' channel
    for i, channel in enumerate(eye_channels):
        _, binaryIm = cv2.threshold(channel, threshold_val, 255, cv2.THRESH_BINARY_INV) #inverse thresholding
        curr_pixel_count = (len(binaryIm[binaryIm>threshold_val])) #count pixels

        ##Record the first instance of pixel count 
        if i == 0: 
            max_pixel_count = curr_pixel_count
            iris_mask = binaryIm.copy()

        #record most number of pixel count and save the thresholded image as a binary image
        if curr_pixel_count > max_pixel_count: 
            max_pixel_count = curr_pixel_count
            iris_mask = binaryIm.copy()
            
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateKernelSize,dilateKernelSize))
    iris_mask = cv2.dilate(iris_mask, kernel,1)
    ###Calculate moments and computing centroid
    M = cv2.moments(iris_mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = np.array([cX,cY])
    #########

    #Find largest contours point
    cnts, _ = cv2.findContours(np.uint8(iris_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for i,cnt in enumerate(cnts):
        if i ==0:
            largest_cnt = cnt
            largest_cnt_pts = cnt.shape[0]

        cnt_pts = cnt.shape[0]

        if cnt_pts>largest_cnt_pts:
            largest_cnt = cnt
            largest_cnt_pts = cnt.shape[0]
    largest_cnt = (np.reshape(largest_cnt,(-1,2)))
    
    ###Iterate through points of contours and obtain the average radius
    radius_avg = 0
    for i,pt in enumerate(largest_cnt):
        dist = (((centroid[0] - pt[0])**2) + ((centroid[1] - pt[1])**2))**0.5
        radius_avg+=dist
    radius_avg=int(1*((radius_avg//(i+1))+1))

    ### Create circular mask with obtained radius_avg  and centroid
    circular_mask = np.zeros_like(img)
    cv2.circle(circular_mask, tuple(centroid), radius_avg, (255,255,255) , -1)

    #mask iris_mask with circular mask to remove outlier points
    iris_mask = cv2.bitwise_and(circular_mask, circular_mask, mask = iris_mask)
    iris_mask=cv2.GaussianBlur(iris_mask, (blurKernelSize,blurKernelSize), cv2.BORDER_DEFAULT)/255
    iris_mask_inv = (1 - iris_mask)
    
    return iris_mask, iris_mask_inv

def getColoredTransformed(img, input_color = np.array([255,0,0]), brightness = 2):

    original_gray = (cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)/(255)) 
    maxGray = (np.amax(original_gray))
    new_mapped_gray = (original_gray.copy() + (1-maxGray)) * brightness

    transformed_color = (new_mapped_gray* input_color)
    transformed_color[:,:,0][transformed_color[:,:,0]>255] = 255
    transformed_color[:,:,1][transformed_color[:,:,1]>255] = 255
    transformed_color[:,:,2][transformed_color[:,:,2]>255] = 255
    transformed_color = transformed_color.astype(np.uint8)
    return transformed_color

def applyEyeColor(img, eyePts, threshold=75, input_color = np.array([125,125,255]), brightness = 1.5):  
    iris_mask, iris_mask_inv = createEyeMasks(img, eyePts)
    transformed_color = getColoredTransformed(img, input_color, brightness)
    justIris = iris_mask* (transformed_color/255)
    justFace = iris_mask_inv * (img/255)
    out = justFace + justIris
    final_output = (out*255).astype(np.uint8)
    return final_output

def colorBothEyes(img, points, threshold = 75, input_color = np.array([125,125,255]), brightness = 1.5):  
    lefteyePts = np.array(points[36:42])
    lefteyePts = np.reshape(lefteyePts, (-1,1,2))
    righteyePts = np.array(points[42:48])
    righteyePts = np.reshape(righteyePts, (-1,1,2))
    color_left_eye = applyEyeColor(img, lefteyePts, threshold, input_color=input_color, brightness=brightness)
    color_right_eye = applyEyeColor(color_left_eye, righteyePts, threshold, input_color=input_color, brightness=brightness)
    return color_right_eye



def empty(a):
    pass

def input_image():
    while True:
        img_path = input("Please enter image filename. (Enter 0 if you want to take a photo from webcam): ")
        if img_path == '0':

            cap = cv2.VideoCapture(0)
            while True:
                success, img = cap.read()
                cv2.imshow("Press s to take a picture", img)
                if cv2.waitKey(1) == ord('s'):
                    cap.release()
                    break

            cv2.destroyAllWindows()
            break
        else:
            list_inputs = os.listdir("input")
            if img_path not in list_inputs:
                print("File not found. Please check if image is in input folder.")
            else:
                img = cv2.imread("input/" + img_path)
                break
    return img


count = 0
if __name__ == '__main__':

    PREDICTOR_PATH =  "./shape_predictor_68_face_landmarks.dat"
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)



    ### Image Input ###
    img = input_image()
    ###
    points = fbc.getLandmarks(faceDetector, landmarkDetector, img)

    if len(points)<=0:
        print("No face detected. Please select another image.")
        img = input_image()
        points = fbc.getLandmarks(faceDetector, landmarkDetector, img)

    while True:
        try:
            maskLips, inverseMaskLips = createLipsMasks(img, points, openKernelSize = 5, blurKernelSize = 31)
            lefteyePts = np.array(points[36:42])
            lefteyePts = np.reshape(lefteyePts, (-1,1,2))
            righteyePts = np.array(points[42:48])
            righteyePts = np.reshape(righteyePts, (-1,1,2))
            leftEyeMask, inverseLeftEyeMask = createEyeMasks(img, lefteyePts, threshold_val = 75,  dilateKernelSize = 4, blurKernelSize = 101)
            rightEyeMask, inverseRightEyeMask = createEyeMasks(img, righteyePts, threshold_val = 75,  dilateKernelSize = 4, blurKernelSize = 101)
            break
        except:
            print("Mask could not be created, please try another image.")
            img = input_image()

    height,width,_ = img.shape
    cv2.namedWindow("Lipstick TrackBars")
    cv2.createTrackbar("Lipstick Red", "Lipstick TrackBars", 0, 255, empty) 
    cv2.createTrackbar("Lipstick Green", "Lipstick TrackBars", 0, 255, empty) 
    cv2.createTrackbar("Lipstick Blue", "Lipstick TrackBars", 0, 255, empty) 
    cv2.createTrackbar("Lipstick Brightness", "Lipstick TrackBars", 100, 500, empty) 
    cv2.createTrackbar("Lipstick Transparency", "Lipstick TrackBars", 100, 100, empty) 

    cv2.namedWindow("Eye TrackBars")
    cv2.createTrackbar("Threshold", "Eye TrackBars", 75, 255, empty) 
    cv2.createTrackbar("Eye Red", "Eye TrackBars", 0, 255, empty) 
    cv2.createTrackbar("Eye Green", "Eye TrackBars", 0, 255, empty) 
    cv2.createTrackbar("Eye Blue", "Eye TrackBars", 0, 255, empty) 
    cv2.createTrackbar("Eye Brightness", "Eye TrackBars", 100, 500, empty) 
    cv2.createTrackbar("Eye Transparency", "Eye TrackBars", 100, 100, empty) 

    
           # winx, winy, winw, winh = cv2.getWindowImageRect('Virtual Painter')
            
    
    os.system('clear')

    
    

    original_img = img.copy()
    prev_threshold = 75
    print("================================")
    print("Welcome to Virtual Makeup Painter")
    print("Press 's' to save img.")
    print("Press 'q' to exit.")
    print("================================")
    while True:

        lips_red = cv2.getTrackbarPos("Lipstick Red", "Lipstick TrackBars")
        lips_green = cv2.getTrackbarPos("Lipstick Green", "Lipstick TrackBars")
        lips_blue = cv2.getTrackbarPos("Lipstick Blue", "Lipstick TrackBars")
        lips_brightness = cv2.getTrackbarPos("Lipstick Brightness", "Lipstick TrackBars")
        lips_transparency = cv2.getTrackbarPos("Lipstick Transparency", "Lipstick TrackBars")

        input_color = np.array([lips_blue, lips_green, lips_red]).astype(np.uint8)
        lips_brightness = lips_brightness/100
        lips_transparency = lips_transparency/100

        lips_transformed_color = getColoredTransformed(original_img, input_color, lips_brightness)

        justLips = cv2.multiply(maskLips, lips_transformed_color/255)
        justFace = cv2.multiply(inverseMaskLips, original_img/255)
        out = justLips + justFace
        lipsImg = (out*255).astype(np.uint8)

        final_lips_img = ((1-lips_transparency) * lipsImg) + ((lips_transparency)*original_img)
        final_lips_img = final_lips_img.astype(np.uint8)


        eye_red = cv2.getTrackbarPos("Eye Red", "Eye TrackBars")
        eye_green = cv2.getTrackbarPos("Eye Green", "Eye TrackBars")
        eye_blue = cv2.getTrackbarPos("Eye Blue", "Eye TrackBars")
        eye_brightness = cv2.getTrackbarPos("Eye Brightness", "Eye TrackBars")
        eye_transparency = cv2.getTrackbarPos("Eye Transparency", "Eye TrackBars")
        eye_threshold = cv2.getTrackbarPos("Threshold", "Eye TrackBars")


        eye_input_color = np.array([eye_blue, eye_green, eye_red]).astype(np.uint8)
        eye_brightness = eye_brightness/100

        if eye_threshold!=prev_threshold:

            leftEyeMask, inverseLeftEyeMask = createEyeMasks(original_img.copy(), lefteyePts, threshold_val = eye_threshold,  dilateKernelSize = 4, blurKernelSize = 101)
            rightEyeMask, inverseRightEyeMask = createEyeMasks(original_img.copy(), righteyePts, threshold_val = eye_threshold,  dilateKernelSize = 4, blurKernelSize = 101)


        eye_transformed_color = getColoredTransformed(img, eye_input_color, eye_brightness)
        eye_transparency = eye_transparency/100

        justLeftIris = leftEyeMask* (eye_transformed_color/255)
        justLeftFace = inverseLeftEyeMask * (final_lips_img/255)
        out = justLeftIris + justLeftFace
        final_left_eye = (out*255).astype(np.uint8)

        justRightIris = rightEyeMask* (eye_transformed_color/255)
        justRightFace = inverseRightEyeMask * (final_left_eye/255)
        out = justRightIris + justRightFace
        final_img = (out*255).astype(np.uint8)

        final_img = ((1-eye_transparency) * final_img) + ((eye_transparency)*final_lips_img)
        final_img = final_img.astype(np.uint8)

        prev_threshold = eye_threshold

        count+=1


        cv2.imshow("Virtual Painter", final_img)
        if count == 1:
            cv2.resizeWindow("Virtual Painter", width, height)
            cv2.moveWindow("Lipstick TrackBars", width + 160,0)
            cv2.moveWindow("Eye TrackBars", width + 160, 300)
            cv2.resizeWindow("Lipstick TrackBars", 200, 400)
            cv2.resizeWindow("Eye TrackBars", 200, 400)
            cv2.moveWindow("Virtual Painter", 130,30)
            
        #cv2.moveWindow("Lipstick TrackBars", winx+winw,30)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key==ord('s'):
        	filename = str(int(time.time()))
        	print("Saving img")
        	cv2.imwrite("output/" + filename + ".jpg", final_img)
    cv2.destroyAllWindows()
