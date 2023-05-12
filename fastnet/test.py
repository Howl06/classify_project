import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os

classes = ('Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
           'Common wheat', 'Fat Hen', 'Loose Silky-bent',
           'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.51819474, 0.5250407, 0.4945761], std=[0.24228974, 0.24347611, 0.2530049])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=torch.load('checkpoints/fasternet/best.pth')
model.eval()
model.to(DEVICE)

path = 'test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
