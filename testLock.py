import time
import Jetson.GPIO as GPIO

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(18, GPIO.OUT)
# GPIO.output(18, GPIO.LOW)

# GPIO.cleanup()

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(18, GPIO.OUT)
# GPIO.output(18, GPIO.LOW)

# time.sleep(10)p

# GPIO.cleanup()
 
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.LOW)
time.sleep(1)
GPIO.cleanup()
time.sleep(5)