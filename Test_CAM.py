import os
from os import listdir
from os.path import isfile, join

import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms as trn

from PIL import Image

from torch.nn import functional as F

import matplotlib.pyplot as plt
import random

from scipy.misc import imresize as imresize
import cv2

from Networks import classification as nets
from Networks.StyleNet import StyleAugmentation
from Networks.libs.Loader import Dataset

import config_classification as conf

def verifyDir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

#model_path = "models/"
model_path = "StyleTransfer_Weights/"
output_path = "output/"

verifyDir(output_path)

Styles = [ "", "SA-", "Aug-", "Aug+SA-"]
Layers = ["conv2", "conv4", "conv4", "Mixed_7c", "branch3"]
Nets = ["WideResNet101.pth", "Xception96.pth", "Xception256.pth", "InceptionV3.pth", "InceptionV4.pth"]
Names = ["WideResNet101", "Xception96", "Xception256", "InceptionV3", "InceptionV4"]
sizes = [(96,8),(96,16),(256,16),(299,16),(299,16)]

models_weights = []

for model, dims, name, layer in zip(Nets, sizes, Names, Layers):
  for style in Styles:
    models_weights.append({"arch": style+model, "layer": layer, "style": style, "model": name, "dim": [dims[0], dims[0]]})

models_weights = [models_weights[:4][3]]

print(models_weights)

class Stylization(nn.Module):
  def __init__(self, layer='r41', alpha=[0.5], remove_stochastic=True, prob=1.0, pseudo1=True, Noise=False, std=1.,mean=0., idx=0):
    super(Stylization, self).__init__()
    self.net = StyleAugmentation(layer, alpha, remove_stochastic, prob, pseudo1, Noise, std, mean, idx).cuda() # CUDA is available
    for name, child in self.net.named_children():
        for param in child.parameters():
            param.requires_grad = False
  def forward(self, img, idx, alpha):
    img=F.interpolate(img, 256, mode='bilinear', align_corners=True)
    self.net.idx = idx
    self.net.alpha = alpha
    with torch.no_grad():
      img = self.net(img).detach()
    return img

def imageTransform(img_dim=[224, 224]):
  width, height = img_dim[0], img_dim[1]
  # load the image transformer
  tf = trn.Compose([trn.ToPILImage(), trn.Resize((width, height)), trn.ToTensor(), trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  return tf

def getWeightsFeatures(model, layer_conv, model_name):
  features_blobs = []
  
  def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))
  # hook the feature extractor
  features_names = [layer_conv] # this is the last conv layer of the resnet
  
  #i = 0
  #for k, v in model.state_dict().items():
  #  print("{} Layer {}".format(i, k))
  #  i = i+1
  
  #print(model)

  for name in features_names:
    if model_name == "Xception":
      model._modules[name].register_forward_hook(hook_feature)
    elif model_name == "InceptionV4":
      model.features[20]._modules[name][0].register_forward_hook(hook_feature)
    elif model_name == "InceptionV3":
      model._modules.get(name).register_forward_hook(hook_feature)
    elif model_name == "WideResNet101":
      model.layer3[3]._modules[name].register_forward_hook(hook_feature)
      
  # get the softmax weight
  params = list(model.parameters())
  weight_softmax = params[-2].data.numpy()
  weight_softmax[weight_softmax<0] = 0

  return features_blobs, weight_softmax

def generateImageMap(images, names_model, best_pred, img_name):
  final_frame = []
  height, width = 120, 380
  blank_column = 255.0*np.ones(shape=[420, width, 3])
  blank_row = 255.0*np.ones(shape=[height, 2560 + width, 3])
  names_style = ["N/A", "SA", "Trad", "Trad+SA"]
  for i in range(0, len(models_weights), 4):
    final_frame.append(cv2.hconcat((blank_column, images[i], images[i+1], images[i+2], images[i+3])))

  print(final_frame[0].shape)

  final_cam = cv2.vconcat((blank_row, final_frame[0], final_frame[1], final_frame[2], final_frame[3], final_frame[4]))
  
  textSize = 1.2
  thickness = 2
  
  # columna nombres de los modelos
  for i in range(0,5):
    y = 210 +420*(i) + height
    if names_model[i*4] == "InceptionV3":
      names_model[i*4] = "InceptionV3-299"
    if names_model[i*4] == "InceptionV4":
      names_model[i*4] = "InceptionV4-299"
    if names_model[i*4] == "Xception96":
      names_model[i*4] = "Xception-96"
    if names_model[i*4] == "Xception256":
      names_model[i*4] = "Xception-256"
    cv2.putText(final_cam, names_model[i*4], (30, y), cv2.FONT_HERSHEY_COMPLEX, textSize, (255, 0, 0), thickness, cv2.LINE_AA)
  
  # fila nombre de los estilos
  for i in range(0,4):
    x = 320+640*i+width
    cv2.putText(final_cam, names_style[i], (x, 70), cv2.FONT_HERSHEY_COMPLEX, textSize, (255, 0, 0), thickness, cv2.LINE_AA)
  
  # filas predicciones por cada modelo y cada estilo
  for i in range(0, len(models_weights), 4):
    print(best_pred[i], best_pred[i+1], best_pred[i+2], best_pred[i+3])
    y = 420*int(i/4) + height + 40
    for j in range(0,4):
      x = 640*j + width + 40
      print((i, x, y))
      cv2.putText(final_cam, str(best_pred[i+j]), (x, y), cv2.FONT_HERSHEY_COMPLEX, textSize, (222, 231, 223), thickness, cv2.LINE_AA)
  
  cv2.imwrite(output_path+img_name+"cam.jpg", final_cam)

  return

def load_model(obj, model_name):
  file_path = model_path+model_name+"/"+obj["arch"]
  #file_path = model_path+obj["arch"]
  checkpoint = torch.load(file_path)
  print("\nAccuracy: ", checkpoint["best_prec"])
  model = nets.ChooseNet(model_name, pretrained=conf)
  state_dict = { str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items() }
  model.load_state_dict(state_dict)
  model.eval()
  return model, checkpoint["best_prec"]

def returnCAM(feature_conv, weight_softmax, class_idx):
  # generate the class activation maps upsample to 256x256
  size_upsample = (256, 256)
  nc, h, w = feature_conv.shape
  print("Last conv shape: ", feature_conv.shape)
  output_cam = []
  for idx in class_idx:
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(imresize(cam_img, size_upsample))
  return output_cam

def testCAM(img, img_name):
  verifyDir(output_path)
  images_CAM = []
  names_CAM = []
  acc_CAM = []
  for obj in models_weights:
    architecture = obj["arch"]
    style = obj["style"]
    layer_conv = obj["layer"]
    model_name = obj["model"]
    if model_name == "Xception96" or model_name == "Xception256":
      model_name = "Xception"
    model, best_pred = load_model(obj, model_name)
    
    print("Model: ", style+model_name)
    
    tf = imageTransform(obj["dim"])
    input_img = Variable(tf(img).unsqueeze(0))

    features_blobs, weight_softmax = getWeightsFeatures(model, layer_conv, model_name)
    
    # forward pass
    
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    #h_x = F.cross_entropy(logit, 1).data.squeeze()
    
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    print('Class activation map is saved ... ')
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    img_cv2 = img
    height, width, _ = img_cv2.shape
    img_cv2 = cv2.resize(img_cv2, (256, 256))
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(256, 256)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img_cv2 * 0.5
    
    cam_result = cv2.resize(result, (640, 420))
    
    images_CAM.append(cam_result)
    names_CAM.append(style+obj["model"])
    acc_CAM.append(probs[0])
    cv2.imwrite(output_path+img_name+architecture+'_cam.jpg', result)
    
  print("prediction:", acc_CAM[0])
  #generateImageMap(images_CAM, names_CAM, acc_CAM, img_name)
  
  return acc_CAM[0], images_CAM[0]

if __name__ == '__main__':
  dir_path = str(random.randint(1, 10))
  dir_path = "2"
  mypath = "img/" + dir_path + "/"
  
  #styles_type = [78020, 8050, 1020, 6050, 100, 76300]
  styles_type = np.arange(10000)
  #alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  alphas = [0.5] #np.arange(0.4, 0.7, 0.1)
  
  #WASM_images = []
  
  content_dataset = Dataset(mypath, 256,256,test=True)
  content_loader = torch.utils.data.DataLoader(dataset=content_dataset, batch_size=16, shuffle=False, num_workers=16, drop_last=True)

  Stylenet = Stylization()
  for it, (img, name) in enumerate(content_loader):
    print("Image nro:", it, "img name:", name[0])
    if it == 1:
      break
    img_name = dir_path+"_"+str(name[0])

    print("No Styling ... ")
    img_nostyled = np.uint8(img[0].permute((1,2,0)).numpy()*255)[:,:,::-1]

    cv2.imwrite(output_path+img_name+".jpg", img_nostyled)
    acc_no, cam_no = testCAM(img_nostyled, img_name+"_")
    
    wasm_it = 0.0
    wasm_al = 0.0
    wasm_no = 0.0
    n = 0.
    
    print("Styling ... ")
    for style_type in styles_type:
      for alpha in alphas:

        print("Styling image: ", name[0]," style number:", style_type, "intensity (alpha):", alpha)
        styled = Stylenet(img.cuda(), style_type, alpha)
        img_styled = np.uint8(styled[0].permute((1,2,0)).cpu().numpy()*255)[:,:,::-1]
        
        cv2.imwrite(output_path+img_name+"_"+str(style_type)+"_"+str(alpha)+'_style.jpg', img_styled)
        acc, cam = testCAM(img_styled, img_name+"_"+str(style_type)+"_"+str(alpha)+"_")
        
        wasm_it += np.float32(np.multiply(acc, cam))
        wasm_al += np.float32(np.multiply(acc*alpha, cam))
        wasm_no += np.float32(cam)
    
      n+=1.
    
    wasm_it = (wasm_it/n - np.float32(np.multiply(acc_no, cam_no)))**2
    wasm_al = (wasm_al/n - np.float32(np.multiply(acc_no, cam_no)))**2
    wasm_no = (wasm_no/n - np.float32(cam_no))**2
    
    wasm_it = np.uint8((wasm_it - wasm_it.min())/(wasm_it.max() - wasm_it.min())*255.)
    wasm_al = np.uint8((wasm_al - wasm_al.min())/(wasm_al.max() - wasm_al.min())*255.)
    wasm_no = np.uint8((wasm_no - wasm_no.min())/(wasm_no.max() - wasm_no.min())*255.)
    #wasm_it = cv2.normalize(cam_no, wasm_it, 0, 255, cv2.NORM_MINMAX)
    #wasm_al = cv2.normalize(cam_no, wasm_al, 0, 255, cv2.NORM_MINMAX)
    #wasm_no = cv2.normalize(cam_no, wasm_no, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(output_path+"wasm_"+img_name+"_"+str(alpha)+'_style.jpg', wasm_it)
    cv2.imwrite(output_path+"wasm-al_"+img_name+"_"+str(alpha)+'_style.jpg', wasm_al)
    cv2.imwrite(output_path+"wasm-no_"+img_name+"_"+str(alpha)+'_style.jpg', wasm_no)
    #WASM_images.append(wasm_it)
    
  
