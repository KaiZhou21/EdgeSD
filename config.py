import platform
import os
import glob

s = platform.system()
dataset_name = "Sci-SBU"
if s == "Windows":
    basic_dir = "D:/Dataset/" + dataset_name
    sp = "\\"
    pth_dir = "pretrain/vgg16.pth"
    batchsize = 5
    gpu_ids = "0"
else:
    basic_dir = "/root/wuwen/Dataset/" + dataset_name
    sp = "/"
    batchsize = 20
    pth_dir = "pretrain/vgg16.pth"
    gpu_ids = "3"

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

image_root = sorted(glob.glob(basic_dir + '/train/img/*.jpg') + glob.glob(basic_dir + '/test/img/*.jpg'))
gt_root = sorted(glob.glob(basic_dir + '/train/gt/*.png') + glob.glob(basic_dir + '/test/gt/*.png'))
mask_root = sorted(glob.glob(basic_dir + '/train/mask/*.png') + glob.glob(basic_dir + '/test/mask/*.png'))
edge_root = sorted(glob.glob(basic_dir + '/train/edge/*.png') + glob.glob(basic_dir + '/test/edge/*.png'))
gray_root = sorted(glob.glob(basic_dir + '/train/gray/*.jpg') + glob.glob(basic_dir + '/test/gray/*.jpg'))

test_img_path = sorted(glob.glob(basic_dir + '/test/img/*.jpg'))
test_label_path = sorted(glob.glob(basic_dir + '/test/Bmask/*.png'))
test_label_dir = basic_dir + '/test/Bmask'

print("platform:{} Batchsize:{} total_number_of_sample:{}".format(s, batchsize, len(image_root)))
