#!/usr/bin/env python3

# python3 application.py
import os

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

from keras.models import load_model
import numpy as np
import copy
import schedule
import cv2
import time 
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[-1], True)

classes = ['NONE', 'alef', 'ayin', 'bet', 'dalet', 'gimel', 'hei', 'het',
       'kafyad', 'khaf', 'kof', 'lamed', 'mem', 'nun', 'peh', 'reish',
       'samech', 'shin', 'tav', 'tet', 'tzadi', 'vav', 'yod', 'zayin']
       
dataColor = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
takingData = 0
className = 'NONE'
count = 0
showMask = 0
current_pred = classes[0]
current_img = None
current_window = None
model = load_model('model_2021_06_07_8_30')

def initClass(name):
    global className, count
    className = name
    os.makedirs(f'data/{name}', exist_ok=True)
    count = len(os.listdir('data/%s' % name))


def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(
        img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new

def predictImage():
    print(time.time())
    print(current_img.shape)
    pred = classes[np.argmax(model.predict(current_img)[0])]
    print(pred)
    cv2.putText(current_window, 'Prediction: %s' %
                (pred), (fx, fy+2*fh), font, 1.0, (245, 210, 65), 2, 1)

def main():
    global font, size, fx, fy, fh
    global takingData, dataColor
    global className, count
    global showMask
    global current_pred
    global current_window
    global current_img
    global model

    x0, y0, width = 120, 120, 300

    cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cam.set(cv2.CAP_PROP_FPS, 5)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    prev = time.time()
    images = []
    
    # schedule.every(2).seconds.do(predictImage)

    while True:
        schedule.run_pending()
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # mirror
        current_window = copy.deepcopy(frame)
        cv2.rectangle(current_window, (x0, y0), (x0+width -
                      1, y0+width-1), dataColor, 12)

        # draw text
        if takingData:
            dataColor = (0, 250, 0)
            cv2.putText(current_window, 'Data Taking: ON', (fx, fy),
                        font, 1.2, dataColor, 2, 1)
        else:
            dataColor = (0, 0, 250)
            cv2.putText(current_window, 'Data Taking: OFF',
                        (fx, fy), font, 1.2, dataColor, 2, 1)
        cv2.putText(current_window, 'Class Name: %s (%d)' % (className, count),
                    (fx, fy+fh), font, 1.0, (245, 210, 65), 2, 1)

        # get region of interest
        roi = frame[y0:y0+width, x0:x0+width]
        roi = binaryMask(roi)
        # cv2.imshow('Original2', roi)

        # apply processed roi in frame
        if showMask:
            current_window[y0:y0+width, x0:x0 +
                   width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # take data or apply predictions on ROI
        if takingData:
            cv2.imwrite('data/{0}/{0}_{1}.png'.format(className, count), roi)
            count += 1

        else:
            # img = np.float32(roi)/255
            current_img = roi
            current_img = np.expand_dims(current_img, axis=0)
            current_img = np.expand_dims(current_img, axis=-1)

            # TODO: make this shit faster
            if True:
                time_elapsed = time.time() - prev
                images.append(current_img)

                if time_elapsed > 1:
                    prev = time.time()
                    predictions = np.array([prediction.argmax() for prediction in model.predict(np.array(images))])
                    print([classes[pred] for pred in predictions])
                    pred = np.argmax(np.bincount(predictions))
                    print(classes[pred])
                    cv2.putText(current_window, 'Prediction: %s' %
                                (classes[pred]), (fx, fy+2*fh), font, 1.0, (245, 210, 65), 2, 1)
                    images = []

            # use below for demoing purposes
            cv2.putText(current_window, 'Prediction: %s' % (current_pred), (x0,y0-25), font, 1.0, (255,0,0), 2, 1)

        # show the window
        cv2.imshow('Original', current_window)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        # use q key to close the program
        if key == ord('1'):
            break
        # Toggle data taking
        elif key == ord('2'):
            takingData = not takingData

        elif key == ord('3'):
            showMask = not showMask

        # Toggle class
        elif key == ord('t'):
            initClass('alef')
        elif key == ord('c'):
            initClass('bet')
        elif key == ord('d'):
            initClass('gimel')
        elif key == ord('s'):
            initClass('dalet')
        elif key == ord('v'):
            initClass('hei')
        elif key == ord('u'):
            initClass('vav')
        elif key == ord('z'):
            initClass('zayin')
        elif key == ord('j'):
            initClass('het')
        elif key == ord('y'):
            initClass('tet')
        elif key == ord('h'):
            initClass('yod')
        elif key == ord('f'):
            initClass('khaf')
        elif key == ord('k'):
            initClass('lamed')
        elif key == ord('n'):
            initClass('mem')
        elif key == ord('b'):
            initClass('nun')
        elif key == ord('x'):
            initClass('samech')
        elif key == ord('g'):
            initClass('ayin')
        elif key == ord('p'):
            initClass('peh')
        elif key == ord('m'):
            initClass('tzadi')
        elif key == ord('e'):
            initClass('kof')
        elif key == ord('r'):
            initClass('reish')
        elif key == ord('a'):
            initClass('shin')
        elif key == ord(','):
            initClass('tav')
        elif key == ord('q'):
            initClass('NONE')
        elif key == ord('5'):
            initClass('kafyad')

    cam.release()


if __name__ == '__main__':
    initClass('NONE')
    main()
