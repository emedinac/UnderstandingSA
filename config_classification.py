# External Hyperparameters
gpus='1' # numbers ('0,1')or '-'
workers = 0
weights = ''
pretrained = False
out_path = 'models/'
dataset = "STL10"

# Network Hyperparameters
architecture = 'WideResNet101'
name_arch = 'Aug+SA-WideResNet101'
n_channels = 3
n_classes = 10 # implicit.
epochs = 200
train_batch = 32
test_batch = 64

lr = 1e-1 # 1e-6 is the best
momentum = 0.9
nesterov=True
weight_decay = 5e-4
dropFC = 0.3 # None
init = 'xavier,gauss' # 'uniform,-0.1,0.1'   or   'he,uniform'
scheduler = '60,120,160,190-0.2' # schedule - lr decay
# scheduler = str(list(range(2,epochs,5)))[1:-1].replace(' ','')+'-0.93' # schedule - lr decay
# scheduler = 'min/0.06/2'
earlystop = False
# Training Structure
type_optimizer = 'SGD' # SGD adam
betas=(0.5, 0.999)
loss = 'CE' # available BCE, DICE,

style = ("r41", [0.4, 0.8], False, 0.6, False, True, 2., 0.) # None
resize = (96,96) # (64,64) # (256,256)

augmentation = True
rot = None # (-15,-10,-5,0,5,10,15)
trans = ([0.1,0.1], (0.8,1.2), 0) # translation, scale, shear
hflip = 0.5
vflip = None
cutout = 8
color = (0.2, 0.2, 0.2, 0.05)
erase = (0.5, 0.3, 0.2)
