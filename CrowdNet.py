#%%
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from random import random, randrange


class CrowdNetModel():

    def __init__(self):
        self.colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def randclr(self):
        return self.colours[randrange(0, 10)]

    def get_points(self,boxes):
        points = []
        for (x, y, w, h) in boxes:
            points.append([int(x + w/3), int(y + h/3)])
        return points

    def predict(self, frame, threshold=0.5):
        tensor = self.transform(frame).unsqueeze(0)
        with torch.no_grad():
            tensor = tensor.to(self.device)
        pred = self.model(tensor)[0]
        indexes = (pred['labels'] == 1) * (pred['scores'] > threshold)
        self.boxes = pred['boxes'].long()[indexes].data.cpu().numpy()
        self.points = self.get_points(self.boxes)
        self.masks = (pred['masks'].permute(1, 0, 2, 3)[0][indexes] > 0.5).data.cpu().numpy()
        self.size = len(self.boxes)
        self.frame = frame
        del tensor
        del pred
        torch.cuda.empty_cache()
        return self.size
    
    def segment(self, index, mask=False, box=False, point=False):
        frame = self.frame
        if mask:
            _mask = self.masks[index]
            r = np.zeros_like(_mask, dtype=np.uint8)
            g = np.zeros_like(_mask, dtype=np.uint8)
            b = np.zeros_like(_mask, dtype=np.uint8)
            r[_mask == 1], g[_mask == 1], b[_mask == 1] = self.randclr()
            colored = np.stack([r, g, b], axis=2)
            frame = cv2.addWeighted(frame, 1, colored, 0.5, 0)
        if box:
            x, y, w, h = self.boxes[index]
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.randclr())
        if point:
            x, y = self.points[index]
            cv2.circle(frame, (x, y), 10, self.randclr(), -1)
        return frame

    def magix(self, mask=False, box=False, point=False):
        tmp = self.frame
        for i in range(self.size):
            self.frame = self.segment(i, mask, box, point)
        frame = self.frame
        self.frame = tmp
        return frame
    

# %%


# %%
