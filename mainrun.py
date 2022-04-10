import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import sys
import glob
import numpy as np
import torch
import subprocess
import pathlib
import time
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from os.path import splitext


model_path = 'models/RRDB_ESRGAN_x4.pth' # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0

for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base,'processing...')
    # read images
    img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    r,g,b,alpha = cv2.split(img)
    alpha = cv2.cvtColor(alpha,cv2.COLOR_RGB2BGR)# convert to RGB to work with ESRGAN
    alpha = alpha * 1.0 / 255
    alpha = torch.from_numpy(np.transpose(alpha[:, :, [2, 1, 0]], (2, 0, 1))).float()
    alpha_LR = alpha.unsqueeze(0)
    alpha_LR = alpha_LR.to(device)

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        color = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        alpha = model(alpha_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    color = np.transpose(color[[2, 1, 0], :, :], (1, 2, 0))
    alpha = np.transpose(alpha[[2, 1, 0], :, :], (1, 2, 0))
    color = (color * 255.0).round()
    alpha = (alpha * 255.0).round()
    r,g,b = cv2.split(color)
    a = cv2.split(alpha)[-1]
    output = cv2.merge([r,g,b,a])
    cv2.imwrite('results/{:s}.png'.format(base), output)
