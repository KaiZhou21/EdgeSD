import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from model.vgg_models import Back_VGG
import glob

pth_dir = "pth/*.pth"


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


img_transform = transforms.Compose([
    transforms.Resize(416),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms.Normalize([0.517, 0.514, 0.492], [0.186, 0.173, 0.181]) # ISTD
])

sample_dir = "sample/*.jpg"
to_pil = transforms.ToPILImage()

if __name__ == '__main__':
    check_mkdir(os.path.join("results"))
    for i, pth_name in enumerate(sorted(glob.glob(pth_dir))):
        if torch.cuda.is_available():
            model = Back_VGG(channel=32, if_dropout=False).cuda()
            model.load_state_dict(torch.load(pth_name, map_location='cuda:0'))
        else:
            model = Back_VGG(channel=32, if_dropout=False)
            model.load_state_dict(torch.load(pth_name, map_location=torch.device('cpu')))
        with torch.no_grad():
            img_list = glob.glob(sample_dir)
            for idx, img_name in enumerate(img_list):
                print('predicting for %d / %d' % (idx + 1, len(img_list)))
                img = Image.open(img_name)
                w, h = img.size
                if torch.cuda.is_available():
                    img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                else:
                    img_var = Variable(img_transform(img).unsqueeze(0))
                coarse, edge, refine = model(img_var)
                refine_prob = torch.sigmoid(refine)
                prediction = np.array(transforms.Resize((h, w))(to_pil(refine_prob.data.squeeze(0).cpu())))
                save_dir = os.path.join("results", img_name.split("\\")[-1].split(".")[0] + "_" + str(i) + ".png")
                Image.fromarray(prediction).save(save_dir)
