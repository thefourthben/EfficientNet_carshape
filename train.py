import torch as t
from torch import nn
from torch.autograd import Variable
from torchnet import meter
import torchvision.datasets as tv
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from model.eff import EfficientNet as eff
from model import utils
import torch.onnx
import onnx
import onnxruntime
from PIL import Image
from time import process_time
from datasets import testdataset, traindataset, valdataset
from ptflops import get_model_complexity_info
from efficientnet_pytorch import EfficientNet



class training():
    def __init__(self, model):
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.lr = 0.01
        self.trainset = traindataset.traindataset(self.transform_train)
        self.trainloader = None
        self.valset = valdataset.valdataset(self.transform_train)
        self.valloader = None
        self.testset = testdataset.testdataset(self.transform_train)
        self.testloader = None
        self.whichmodel = model
        self.n_epoch = 0
        self.loss = t.nn.CrossEntropyLoss()
        self.factor = 0.5
        self.classes = ("bike", "motocycles", "car", "van", "truck", "trailer")
        self.block = [ 
            utils.BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, stride=[1, 1], se_ratio=0.25),
            utils.BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, stride=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, stride=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, stride=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, stride=[1, 1], se_ratio=0.25),
            utils.BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1, 1], se_ratio=0.25)
            ]
        self.globalparm = utils.GlobalParams(
            width_coefficient=1.0,
            depth_coefficient=1.1,
            image_size=224,
            dropout_rate=0.2,
            num_classes=6,
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            drop_connect_rate=0.2,
            depth_divisor=8,
            min_depth=None,
            use_se=False,
            local_pooling=True
            )

    def preproc(self):
        # Configuring the weights
        class_count = self.trainset.getnumclasses()
        num_samples = sum(class_count) * 0.9
        class_weights = [num_samples/class_count[i] for i in range(len(class_count))]
        weights = [class_weights[self.trainset.class_labels[i]] for i in range(int(num_samples))]
        sampler = t.utils.data.sampler.WeightedRandomSampler(weights, int(num_samples), replacement=False)

        # Loading and getting the trainset
        self.trainloader = t.utils.data.DataLoader(self.trainset,batch_size=16,
        num_workers=2, sampler = sampler)
        self.valloader = t.utils.data.DataLoader(self.valset, batch_size=16,
        num_workers=2, shuffle=True)

        # Creating and loading the test set
        self.testloader = t.utils.data.DataLoader(self.testset, batch_size=16, 
        num_workers=2, shuffle=False)
        print(len(self.testset))

    def train(self):
        self.preproc()
        # Getting the Model
        model = eff(self.block, self.globalparm)

        # Getting the complexity and structure of the model
        # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
        # Getting pretrained models and weights
        # utils.load_pretrained_weights(model, "efficientnet-b0", load_fc=False)
        # utils.load_pretrained_weights(model, "efficientnet-b3", load_fc=False)
        
        # Unfreezing the model parameters
        for param in model.parameters():
            param.requires_grad = True

        # Setting up the Loss function and optimizer
        optimizer = t.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.00001)
        # optimizer = t.optim.RMSprop(model.parameters(), lr = init_lr, momentum=0.9, weight_decay=1e-5)
        # optimizer = t.optim.Adam(model.parameters(), lr = self.lr, weight_decay=0.000125)
        # scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epoch)
        # scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.97)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = "max",factor=self.factor,
         patience=5, verbose=True)
        
        # setting the metric and confusion matrix
        lm = meter.AverageValueMeter()
        model.cuda()
        
        # Training 
        for num in range (self.n_epoch):
            model.train()
            lm.reset()
            total = 0
            correct = 0
            train_loss = 0
            confusion_matrix = torch.zeros(6, 6)
            for i, (data, label) in enumerate(self.trainloader):
                inputs = Variable(data).cuda()
                target = Variable(label).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                l = self.loss(outputs, target)
                train_loss = l
                l.backward()
                optimizer.step()
                lm.add(l.data.item())
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                # for tasd, p in zip(target.view(-1), predicted.view(-1)):
                #     confusion_matrix[tasd.long(), p.long()] += 1
            val_acc, val_loss, _ = self.val(model, self.valloader)
            print('Epoch: {epoch} | Train Loss: {loss} | Train Acc: {train_acc} | Val Loss: {val_loss} |Val Acc: {val_acc}'.format(
                epoch = num, 
                loss = lm.value()[0],
                train_acc = 100 * correct/total,
                val_acc = val_acc,
                val_loss = val_loss, 
            ) )
            # print(confusion_matrix)
            scheduler.step(val_acc)
            self.savemodel(model, num, val_acc, val_loss)
        final_acc, _, cm= self.val(model, self.testloader, True)
        print("Finished Training! The result is: {res} | Confusion matrix:\n {cm}".format(
            res = final_acc,
            cm = cm
        ))
        w = t.randn(1, 3, 224, 224).cuda()
        # model.set_swish(memory_efficient=False)
        t.onnx.export(model, w, "result.onnx", verbose= False)
        self.onnxconversion()


    def val(self, model, dataloader, test=False):
        model.cuda()
        model.eval()
        correct = 0
        total = 0
        l = 0
        confusion_matrix = torch.zeros(6, 6)
        with t.no_grad():
            for n, (inputs, label) in enumerate(dataloader):
                inputs, target = Variable(inputs).cuda(), Variable(label.long()).cuda()
                score = model(inputs)
                l = self.loss(score, target)
                _, predicted = t.max(score.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                if test:
                    for tasd, p in zip(target.view(-1), predicted.view(-1)):
                        confusion_matrix[tasd.long(), p.long()] += 1
        return (100 * correct/total), l, confusion_matrix
    
    def onnxconversion(self):
        onnx_model = onnx.load("result.onnx")
        onnx.checker.check_model(onnx_model)
        ort = onnxruntime.InferenceSession("result.onnx")
        correct = 0
        confusion_matrix = torch.zeros(6, 6)
        def to_numpy(imga):
            return imga.detach().cpu().numpy()
        res = []
        a = len(self.testset)
        time = 0
        for i in range(a):
            img = self.transform_train(Image.open(self.testset.imgs[i]))
            img = img.view(1, 3, 224, 224)
            # compute ONNX Runtime output prediction
            t0 = process_time()
            ort_inputs = {ort.get_inputs()[0].name: to_numpy(img)}
            ort_outs = ort.run(None, ort_inputs)
            t1 = process_time()
            correct += (np.argmax(ort_outs) == self.testset.class_labels[i]).sum().item()
            res.append(np.argmax(ort_outs))
            if len(res) % a == (a-1):
                for tasd, p in zip(t.Tensor(self.testset.class_labels).view(-1), t.Tensor(res).view(-1)):
                    confusion_matrix[tasd.long(), p.long()] += 1
                res = []
            time += t1 - t0
        print("Onnx Accuracy: {acc}| Time Spent: {ts} | Confusion Matrix: \n {cm}".format(
                acc=100*(correct / len(self.testset)),
                ts=time,
                cm=confusion_matrix
        ))
    
    
    def savemodel(self,model, e, a, l):
        t.save(model.state_dict(), "EfficientLite/E{e}_L{l}_A{a}.pth".format(
            l=l,
            a=a,
            e=e
        ))

if __name__ == "__main__":
    t.cuda.device(1)
    tr = training(eff)
    tr.train()    