import os, cv2
import pickle
import torch
import torchvision
from libs.Loader import Dataset
import numpy as np

from Networks.StyleNet import StyleAugmentation
from torchvision import transforms

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    print_idx = True
    batch_size = 8
    num_examples = 4
    alpha_list = np.arange(0.1,1,0.2)
    imsz = [256, 128, 96]
    for sz in imsz:
        for alpha in alpha_list:
            content_dataset = Dataset('Database/COCO/2017/train2017/',sz,sz,test=True)
            content_loader = torch.utils.data.DataLoader(dataset = content_dataset,
                                                            batch_size  = batch_size,
                                                            shuffle     = False,
                                                            num_workers = 1,
                                                            drop_last   = True)
            train__setting = torchvision.datasets.STL10(root='./Database/',
                                                            split='train',
                                                            transform=transforms.Compose([ transforms.Resize(sz, interpolation=2),
                                                                                         transforms.ToTensor(), 
                                                                                        ]), 
                                                            target_transform=None,
                                                            download=True)
            content_loader = torch.utils.data.DataLoader(train__setting, 
                                                            batch_size=batch_size, 
                                                            shuffle=False, 
                                                            num_workers=1)

            
            Stylenet = StyleAugmentation(layer="r41", alpha=[alpha], prob=1.0, pseudo1=True, Noise=False, std=1.,mean=0.).cuda()
            output_img = np.zeros([batch_size*sz,(num_examples+1)*sz,3], dtype = np.uint8)
            for it, (content,_) in enumerate(content_loader):
                Image = np.uint8(content.permute(0,2,3,1).cpu().detach().numpy()*255)
                for n in range(batch_size):
                    output_img[n*sz:(n+1)*sz, 0:sz] = Image[n]
                    if print_idx:
                        output_img[n*sz:(n+1)*sz, 0:sz] = cv2.putText(output_img[n*sz:(n+1)*sz, 0:sz],'Image',  (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                for j in range(num_examples):
                    styled = Stylenet(content.cuda()) # Manual index. Edit the StyleNet file
                    idx = Stylenet.idx[0]
                    styled = np.uint8(styled.permute(0,2,3,1).cpu().data.numpy()*255)
                    for n in range(styled.shape[0]):
                        output_img[n*sz:(n+1)*sz, (j+1)*sz:(j+2)*sz] = styled[n]
                        if print_idx:
                            output_img[n*sz:(n+1)*sz, (j+1)*sz:(j+2)*sz] = cv2.putText(output_img[n*sz:(n+1)*sz, (j+1)*sz:(j+2)*sz], str(idx),  (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                break;
            output_img = np.array(output_img)
            print(sz, alpha, output_img.shape)
            cv2.imwrite('image_test_{0}_{1:.1f}.png'.format(sz, alpha),cv2.cvtColor( output_img ,cv2.COLOR_BGR2RGB))