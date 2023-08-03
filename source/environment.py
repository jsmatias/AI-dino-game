""" Environment 
"""

import time
from mss import mss
from PIL import Image, ImageOps
import numpy as np
import keyboard
import cv2

class Environment:
    """ Class
    """
    def __init__(self):
        ########
        # these are some various screenshot parameters that I found worked well for different resolutions
        # Use it as a starting point but use the open cv code snippets below to tweak your screenshot window
        # Do note that the lower the resolution you use, the faster the code runs
        # I saw a 200% screenshot framerate increase from dropping my screen resolution from 4k to 720p

        self.mon = {'top': 300, 'left': 10, 'width': 680, 'height': 125} #1080p resolution
        # self.mon = {'top': 185, 'left': 190, 'width': 115, 'height': 30} #720p resolution
        # self.mon = {'top': 1000, 'left': 0, 'width': 3840, 'height': 760} #4k resolution
        ########
        self.matrixShape = (76, 384)
        # self.matrixShape = (45, 230)
        self.tensorShape = (*self.matrixShape, 4)

        self.sct = mss()
        self.counter = 0
        self.startTime = -1
        self.imageBank = []
        self.imageBankLength = 4  # number of frames for the conv net
        self.actionMemory = 2  # init as 2 to show no action taken
        # image processing
        self.ones = np.ones(self.tensorShape)
        self.zeros = np.zeros(self.tensorShape)
        self.zeros1 = np.zeros(self.tensorShape)
        self.zeros2 = np.zeros(self.tensorShape)
        self.zeros3 = np.zeros(self.tensorShape)
        self.zeros4 = np.zeros(self.tensorShape)
        self.zeros1[:, :, 0] = 1
        self.zeros2[:, :, 1] = 1
        self.zeros3[:, :, 2] = 1
        self.zeros4[:, :, 3] = 1

    def startGame(self):
        # start the game, giving the user a few seconds to click on the chrome tab after starting the code
        for i in reversed(range(3)):
            print("game starting in ", i)
            time.sleep(1)

    def step(self, action):
        actions = {0: "space", 1: "down"}
        if action != self.actionMemory:
            if self.actionMemory != 2:
                keyboard.release(actions.get(self.actionMemory))
                # print(f"past action: {actions.get(self.actionMemory)}")
            if action != 2:
                keyboard.press(actions.get(action))
                # print(f"new action: {actions.get(action)}")
        self.actionMemory = action

        # This is where the screenshot happens
        screenshot = self.sct.grab(self.mon)
        img = np.array(screenshot)[:, :, 0]
        processedImg = self._processImg(img)
        state = self._imageBankHandler(processedImg)
        done = self._done(processedImg)
        reward = self._getReward(done)
        return state, reward, done

    def reset(self):
        self.startTime = time.time()
        keyboard.press("space")
        time.sleep(0.5)
        keyboard.release("space")
        return self.step(0)

    def _processImg(self, img):
        img = Image.fromarray(img)
        img = img.resize(self.matrixShape[::-1], Image.LANCZOS)
        if np.sum(img) > 2000000:
            img = ImageOps.invert(img)
        img = self._contrast(img)

        # You can use the following open CV code segment to test your in game screenshots
        # cv2.imshow("image",img)
        # cv2.waitKey(0)
        # # # if cv2.waitKey(25) & 0xFF == ord('q'):
        # cv2.destroyAllWindows()

        img = np.reshape(img, self.matrixShape)
        return img

    def _contrast(self, pixvals):
        minval = 32  # np.percentile(pixvals, 2)
        maxval = 171  # np.percentile(pixvals, 98)
        pixvals = np.clip(pixvals, minval, maxval)
        pixvals = (pixvals - minval) / (maxval - minval)
        pixvals[pixvals < 0.5] = 0
        pixvals[pixvals >= 0.5] = 1
        return pixvals

    def _imageBankHandler(self, img):
        img = np.array(img)
        while len(self.imageBank) < (self.imageBankLength):
            self.imageBank.append(np.reshape(img, (*self.matrixShape, 1)) * self.ones)

        bank = np.array(self.imageBank)
        toReturn = self.zeros
        img1 = (np.reshape(img, (*self.matrixShape, 1)) * self.ones) * self.zeros1
        img2 = bank[0] * self.zeros2
        img3 = bank[1] * self.zeros3
        img4 = bank[2] * self.zeros4

        toReturn = np.array(img1 + img2 + img3 + img4)

        self.imageBank.pop(0)
        self.imageBank.append(np.reshape(img, (*self.matrixShape, 1)) * self.ones)

        return toReturn

    def _getReward(self, done):
        if done:
            return -15
        else:
            return 1
            # return time.time() - self.startTime

    def _done(self, img):
        img = np.array(img)
        img = img[7:22, 128:267] # get the "game over" part of the image 

        val = np.sum(img)
        # Sum of the reset pixels when the game ends in the night mode
        expectedVal = 301.0
        # Sum of the reset pixels when the game ends in the day mode
        expectedVal2 = 301.0

        # This method checks if the game is done by reading the pixel values
        # of the area of the screen at the reset button. Then it compares it to
        # a pre determined sum. You might need to fine tune these values since each
        # person's viewport will be different. use the following print statements to
        # help you find the appropriate values for your use case

        # print("val: ", val)
        # print("Difference1: ", np.absolute(val-expectedVal2))
        # print("Difference2: ", np.absolute(val-expectedVal))
        if (
            np.absolute(val - expectedVal) > 15 and np.absolute(val - expectedVal2) > 15
        ):  # seems to work well
            return False
        return True
