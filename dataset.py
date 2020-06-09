import torch as t
from t.utils.data import Dataset
import cv2
import random
from torchvision import transforms

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
class imageproc(Dataset):
    # add img aug 
    def __init__(self, img_size = 224, batch_size = 32, aug = True, image_weights = False):
        self.label = []
        self.img_files = []
        self.labels = ["bike", "motocycles", "car", "van", "truck", "trailer"]
        try:
            paths = [i for i in str(Path("./Desktop/car_images"))]
            for img in paths:
                if not os.path.isdir(img): 
                    raise Exception('%s does not exist' % path)
                f = glob.iglob(path + os.sep + '*.*')
                for x in f:
                    if os.path.splitext(x)[-1].lower() in img_formats:
                        self.img_files.append(x)
                        if img == "0bk":
                            self.label.append((img, 0))
                        elif img == "1m":
                            self.label.append((img, 1))
                        elif img == "2c":
                            self.label.append((img, 2))
                        elif img == "4bb":
                            self.label.append((img, 3))
                        elif img == "5ts":
                            self.label.append((img, 4))
                        elif img == "7tr":
                            self.label.append((img, 5))
        except:
            raise Exception('Error loading data from %s. See %s' % (path, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.image_weights = image_weights
        self.mosaic = self.augment  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        # if self.image_weights:
        #     index = self.indices[index]
        # Augment imagespace
        r = random.randint(1, 10)
        #rotations
        img = transforms.RandomRotation(35)
        # Augment colorspace
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        # Apply cutouts
        # if random.random() < 0.9:
        #     labels = cutout(img, labels)
        # random left-right flip


        # random up-down flip
    
        res=t.load(self.label[index][0])
        return res, self.label[index][1]

    def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
