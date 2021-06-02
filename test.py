# import cv2
# # from nanocamera.NanoCam import Camera
# import nanocamera as nano

# camera = nano.Camera(camera_type=1, device_id=0, width=640, height=480, fps=30)
# print('CSI Camera ready? - ', camera.isReady())
# while camera.isReady():
#     try:
#         # read the camera image
#         frame = camera.read()
#         # display the frame
#         cv2.imshow("Video Frame", frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     except KeyboardInterrupt:
#         break

# # close the camera instance
# camera.release()

# # remove camera object
# del camera


import os
import cv2

capWebcam = cv2.VideoCapture(0)  # declare a VideoCapture object and associate to webcam, 0 => use 1st webcam

print("default resolution = " + str(capWebcam.get(cv2.CAP_PROP_FRAME_WIDTH)) + "x" + str(
    capWebcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

capWebcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)  # change resolution to 320x240 for faster processing
capWebcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

print("updated resolution = " + str(capWebcam.get(cv2.CAP_PROP_FRAME_WIDTH)) + "x" + str(
    capWebcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if capWebcam.isOpened() == False:
    print("error: capWebcam not accessed successfully\n\n")
    os.system("pause")

while cv2.waitKey(1) != 27 and capWebcam.isOpened():
    blnFrameReadSuccessfully, imgOriginal = capWebcam.read()

    if not blnFrameReadSuccessfully or imgOriginal is None:
        print("error: frame not read from webcam\n")
        os.system("pause")
        break 

    cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)

    cv2.imshow("imgOriginal", imgOriginal)

cv2.destroyAllWindows()