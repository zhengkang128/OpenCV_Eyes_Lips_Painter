# Virtual Lips and Eyes Painter

![Alt Text](https://raw.githubusercontent.com/zhengkang128/OpenCV_Eyes_Lips_Painter/main/etc/1.gif)


## Description
A simple GUI for eyes and lips painting using Python, OpenCV and Dlib. This is a project assignment given by the OpenCV course, Computer Vision 2: Applications. (https://opencv.org/course-computer-vision-two/)

Please continue reading to understand how to use this software.

For detailed explanation of the code, please refer to my notebook ``` Project1-Virtual-Makeup.ipynb ```

## Installation
1. Install libraries:
``` pip3 install opencv-python numpy dlib ```
2. Download shape_predictor_68_face_landmarks from (https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and put it in the same directory of this repository.


## Instructions
1. Put input images in the input folder.
2. Run virtual_painter.py script ``` python3 virtual_painter.py ```
3. You will be prompted to give an input image. Provide the filename of the image you want to use as input. (Or enter 0 to take a photo. An OpenCV webcam window will pop-up. Press 's' to take a photo to use as input.)
4. Two windows of trackbars will appear for input parameters for lips and eyes respectively. You may control the trackbars until you obtain your desired output

![Alt Text](https://raw.githubusercontent.com/zhengkang128/OpenCV_Eyes_Lips_Painter/main/etc/6.png)

## Input Parameters
1. Red, Green, Blue - To select desired color for eyes/lips (0-255)
2. Brightness - Controls the brightness of the color (Dim Colors: brightness < 100, Brighten Colors: brightness>100)
3. Transparency - Control the transparency of the eye/lips filter. (0 - Completely opaque, 100 - Completely transparent)
4. Threshold (Eyes Only)- Variable to segment eye lens from eyes. Adjust this to obtain a good segmentation of the eye lens.
