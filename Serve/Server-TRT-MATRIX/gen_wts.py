import torch
import struct
from efficientnet_pytorch import EfficientNet


# model = EfficientNet.from_pretrained('efficientnet-b0','../500B_50stft_98.0.pth',8,1)
model = torch.load('../2000B_100stft_99.2.pth')
model.eval()

f = open('test_stft.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
f.close()
