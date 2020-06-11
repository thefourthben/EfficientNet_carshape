import os
import shutil
import random

path = "../Desktop/car_images"
count = 6
for root, dirs, files in os.walk(path):
    if root not in ["../Desktop/car_images/val", "../Desktop/car_images/test", "../Desktop/car_images/train"]:
        length = len(files)
        trainnum = random.sample(range(length), int(length*0.9))
        for i in range(length):
            oldpath = str(os.path.join(root, files[i]))
            newpath = oldpath[:-5] + str(count) + ".jpg"
            os.rename(oldpath, newpath)
            if i in trainnum:
                shutil.copy(newpath, "../Desktop/car_images/train")
            else:
                if random.random() < 0.5:
                    shutil.copy(newpath, "../Desktop/car_images/test")
                else:
                    shutil.copy(newpath, "../Desktop/car_images/val")
        count -= 1