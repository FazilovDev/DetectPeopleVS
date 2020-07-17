#%%
import torch
import CrowdNet
from PIL import Image, ImageFile
from IPython.display import display
import numpy as np
from CrowdNet import CrowdNetModel
import cv2

# %%
im = cv2.imread('crowd1.jpg')  
#cv2.imshow('image',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# %%
model = CrowdNetModel()

# %%
count = model.predict(im)
print(count)
#model.cluster()
frame = model.magix(mask=True, box=True, point=True)
#print(model.density)
i = Image.fromarray(frame).convert('RGB')
#cv2.imshow('WebCam', frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
display(i)
cv2.imwrite('photo.jpg',frame)
# %%


# %%
