import torch
import models
import os
import numpy as np
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io

device = torch.device("cuda:0")
resume = '/mnt/ckpt/only-final-lr-0.01-iter-50000.pth'
folder = 'results/val/'
all_folder = os.path.join(folder, 'all')
png_folder = os.path.join(folder, 'png')
mat_folder = os.path.join(folder, 'mat')
batch_size = 1
assert batch_size == 1

try:
    os.mkdir(all_folder)
    os.mkdir(png_folder)
    os.mkdir(mat_folder)
except Exception:
    print('dir already exist....')
    pass

model = models.resnet101(pretrained=False)
model = torch.nn.DataParallel(model, device_ids=(0,1,2,3))
model.to(device)
model.eval()

#resume..
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

test_dataset = BSDS_RCFLoader(split="test")
test_loader = DataLoader(
    test_dataset, batch_size=batch_size*4,
    num_workers=1, drop_last=True, shuffle=False)

with torch.no_grad():
    for i, (image, ori, img_files) in enumerate(test_loader):
        h, w = ori.size()[2:]
        image = image.to(device)

        outs = model(image, (h, w))


        fuse = outs[-1].squeeze().detach().cpu().numpy()
        outs.append(ori)
        
        idx = 0
        print('working on .. {}'.format(i))
        for j in range(outs[0].shape[0]):
            name = img_files[j][5:-4]
            for result in outs:
                idx += 1
                result = result.squeeze().detach().cpu().numpy()
                print(result[j].shape)
                if len(result[j].shape) == 3:
                    result[j] = result[j].transpose(1, 2, 0).astype(np.uint8)
                    result[j] = result[j][:, :, [2, 1, 0]]
                    Image.fromarray(result[j]).save(os.path.join(all_folder, '{}-img.jpg'.format(name)))
                else:
                    result[j] = (result[j] * 255).astype(np.uint8)
                    Image.fromarray(result[j]).save(os.path.join(all_folder, '{}-{}.png'.format(name, idx)))
            Image.fromarray((fuse[j] * 255).astype(np.uint8)).save(os.path.join(png_folder, '{}.png'.format(name)))
            io.savemat(os.path.join(mat_folder, '{}.mat'.format(name)), {'result': fuse[j]})
    print('finished.')

