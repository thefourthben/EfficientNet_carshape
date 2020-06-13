import os
import shutil
import random
from torchvision import transforms
from PIL import Image

def aug(p):
    x = Image.open(p)
    angle = random.choice(angles)
    tran1 = transforms.functional.rotate(x, angle)
    tran1.save(str(p[:-4]) + "1.jpg")
    tran2 = transforms.functional.rotate(x, angle*-1)
    tran2.save(str(p[:-4]) + "2.jpg")
    tran3 = transforms.functional.adjust_brightness(x, 0.5)
    tran3.save(str(p[:-4]) + "3.jpg")
    tran4 = transforms.functional.adjust_brightness(x, 1.5)
    tran4.save(str(p[:-4]) + "4.jpg")

path = "../Desktop/car_images"
count = 6
angles = [8, 15]
for root, dirs, files in os.walk(path):
    if root not in ["../Desktop/car_images/val", "../Desktop/car_images/test", "../Desktop/car_images/train"]:
        length = len(files)
        trainnum = random.sample(range(length), int(length*0.9))
        for i in range(length):
            oldpath = str(os.path.join(root, files[i]))
            newpath = oldpath[:-5] + str(count) + ".jpg"
            os.rename(oldpath, newpath)
            if i in trainnum:
                a = shutil.copy(newpath, "../Desktop/car_images/train", follow_symlinks=True)
                if random.random() < 0.5:
                    aug(a)
                os.rename(a, a[:-4] + "_.jpg")
            else:
                if random.random() < 0.5:
                    a = shutil.copy(newpath, "../Desktop/car_images/test", follow_symlinks=True)
                    if random.random() < 0.5:
                        aug(a)
                    os.rename(a, a[:-4] + "_.jpg")
                else:
                    a = shutil.copy(newpath, "../Desktop/car_images/val", follow_symlinks=True)
                    if random.random() < 0.5:
                        aug(a)
                    os.rename(a, a[:-4] + "_.jpg")
        count -= 1

