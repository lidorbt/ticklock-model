#!/usr/bin/env python3

# python3 application.py
import os
from keras.models import load_model
import numpy as np
import copy
import cv2
import time
import Jetson.GPIO as GPIO
import tensorflow as tf
import requests
import threading

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[-1], True)

classes = ['NONE', 'alef', 'ayin', 'bet', 'dalet', 'gimel', 'hei', 'het',
       'kafyad', 'khaf', 'kof', 'lamed', 'mem', 'nun', 'peh', 'reish',
       'samech', 'shin', 'tav', 'tet', 'tzadi', 'vav', 'yod', 'zayin']
       
letters = {
    'alef': ['א'],
    'ayin': ['ע'],
    'bet': ['ב'],
    'dalet': ['ד'],
    'gimel': ['ג'],
    'hei': ['ה'],
    'het': ['ח'],
    'khaf': ['כ', 'ך'],
    'kof': ['ק'],
    'lamed': ['ל'],
    'mem': ['מ', 'ם'],
    'nun': ['נ', 'ן'],
    'peh': ['פ', 'ף'],
    'reish': ['ר'],
    'samech': ['ס'],
    'shin': ['ש'],
    'tav': ['ת'],
    'tet': ['ט'],
    'tzadi': ['צ', 'ץ'],
    'vav': ['ו'],
    'yod': ['י'],
    'zayin': ['ז']
}       

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
model = load_model('models/model_30')
chars_amount = 2
pred_interval = 5

server_api = 'https://us-central1-tiklock-36d87.cloudfunctions.net/api/'
lock_id = '5xDVhyuHpxM36WlNkxaa'

password = requests.get(server_api + lock_id).text

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

def openLock():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    GPIO.output(18, GPIO.LOW)
    time.sleep(5)
    GPIO.cleanup()

def try_unlock(unlock_letters):
    heb_unlock_letters = [letters[x][0] for x in unlock_letters]
    unlock_word = ''.join(heb_unlock_letters)
    return unlock_word == password

print('The password is: ', password)

def main():
    global font, size, fx, fy, fh
    global takingData
    global showMask
    global current_pred
    global current_window
    global current_img
    global model

    unlock_thread = threading.Thread(target=openLock)

    x0, y0, width = 120, 120, 300

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    prev = time.time()
    images = []

    unlock_letters = []
    is_unlocking = False

    while True:
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # mirror
        current_window = copy.deepcopy(frame)
        cv2.rectangle(current_window, (x0, y0), (x0+width - 1, y0 + width-1), dataColor, 12)

        # get region of interest
        roi = frame[y0: y0 + width, x0: x0 + width]
        roi = binaryMask(roi)

        # cv2.imshow('ROI', roi)

        # apply processed roi in frame
        if showMask:
            current_window[y0: y0 + width, x0: x0 +
                   width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        current_img = roi
        current_img = np.expand_dims(current_img, axis=-1)

        time_elapsed = time.time() - prev
        images.append(current_img)

        cv2.putText(current_window, 'Next capture in: %ss' % (str(pred_interval - time_elapsed)[:5]),
            (fx, fy+fh), font, 1.0, (245, 210, 65), 2, 1)


        if time_elapsed > pred_interval:
            prev = time.time()

            # current_pred = classes[np.argmax(model.predict(current_img)[0])]
            predictions = np.array([prediction.argmax() for prediction in model.predict(np.array(images[-chars_amount:]))])
            predicted_letters = [classes[pred] for pred in predictions]
            print(predicted_letters)
            current_pred = classes[np.argmax(np.bincount(predictions))]
            print(current_pred)

            images = []

            if current_pred == 'kafyad':
                if not is_unlocking:
                    print('Trying to unlock...')
                    is_unlocking = True

                else:
                    password_matched = try_unlock(unlock_letters)

                    if password_matched:
                        print('Unlocked successfuly!')
                        if not unlock_thread.is_alive():
                            unlock_thread.start()

                    else:
                        print('Failed to unlock!')

                    unlock_letters = []
                    is_unlocking = False
            else:
                if is_unlocking and current_pred not in ['NONE', 'kafyad']:
                    unlock_letters.append(current_pred)

        cv2.putText(current_window, 'Prediction: %s' % (current_pred), (fx, fy), font, 1.2, dataColor, 2, 1)

        # show the window
        cv2.imshow('Original', current_window)

        # Keyboard inputs
        key = cv2.waitKey(1) & 0xff

        # use q key to close the program
        if key == ord('1'):
            break
        elif key == ord('3'):
            showMask = not showMask

    cam.release()

if __name__ == '__main__':
    main()
