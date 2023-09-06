import torch
import time
from torch.utils import data
import numpy as np
import argparse
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from model.vgg_models import Back_VGG
from utils import clip_gradient, adjust_lr
from config import image_root, gt_root, mask_root, gray_root, edge_root, test_img_path, test_label_path, sp, batchsize, \
    test_label_dir,dataset_name
from utils import smoothness_loss, BER
from dataset import Dataset_train, Dataset_test
import os
import shutil

to_pil = transforms.ToPILImage()

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=4, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
opt = parser.parse_args()
save_path = 'pth/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def train(train_loader, test_loader, model, optimizer, epoch):
    m = "a"
    if epoch == 1:
        m = "w"
    model.train()
    for images, gts, masks, grays, edges in tqdm(train_loader):
        # ------------------------------------training stage-----------------------------------
        # 情况梯度
        optimizer.zero_grad()
        # 将图像加载到GPU
        images, gts, masks, grays, edges = images.cuda(), gts.cuda(), masks.cuda(), grays.cuda(), edges.cuda()
        # 一个batch中的像素数量
        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)
        # 接收模型的三个输出
        coarse, edge, refine = model(images)
        # sigmoid归一化
        coarse_prob = torch.sigmoid(coarse)
        coarse_prob = coarse_prob * masks
        # sigmoid归一化
        refine_prob = torch.sigmoid(refine)
        refine_prob = refine_prob * masks
        # 损失一
        smoothLoss_cur1 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(coarse), grays)
        sal_loss1 = ratio * CE(coarse_prob, gts * masks) + smoothLoss_cur1
        # 损失二
        smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(refine), grays)
        sal_loss2 = ratio * CE(refine_prob, gts * masks) + smoothLoss_cur2
        # 损失三
        edge_loss = opt.edge_loss_weight * CE(torch.sigmoid(edge), edges)
        bce = sal_loss1 + edge_loss + sal_loss2
        # 反向传播
        loss = bce
        loss.backward()
        # 防止梯度爆炸
        clip_gradient(optimizer, opt.clip)
        # 优化器计步器，更新学习率
        optimizer.step()

        # if i % 10 == 0 or i == len(train_loader):
        #     print(
        #         '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.
        #             format(datetime.now(), epoch, opt.epoch, i, len(train_loader), sal_loss1.data, edge_loss.data,
        #                    sal_loss2.data))
    # 测试环节
    with torch.no_grad():
        test_loss = 0
        model.eval()
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        for idx, (image, mask) in enumerate(test_loader):
            # 计算原图尺寸
            yuantu = Image.open(test_images[idx])
            w, h = yuantu.size
            # 计算预测结果
            image = image.cuda()
            mask = mask.cuda()
            coarse, edge, refine = model(image)
            refine_prob = torch.sigmoid(refine)
            loss = CE(refine_prob, mask)
            test_loss += loss.item()
            filename = os.path.splitext(test_images[idx])[0].split(sp)[-1]
            prediction = np.array(transforms.Resize((h, w))(to_pil(refine_prob.data.squeeze(0).cpu())))
            save_dir = os.path.join(temp_dir, '%s.png' % filename)
            Image.fromarray(prediction).save(save_dir)
    test_loss = test_loss / len(test_loader.dataset)
    # 计算测试环节的BER值
    ber, acc = BER(temp_dir, test_label_dir)
    log = 'Epoch:{:0>3} BER:{:.2f} ACC:{:.2f} TEST_LOSS:{:.4f}'.format(epoch, ber, acc, test_loss)
    print(log)

    tem = '{:0>3} {:.2f} {:.2f} {:.4f}'.format(epoch, ber, acc, test_loss)
    with open(log_file_name, m) as f:
        f.write(tem + "\n")
    pth_name = 'Epoch_{:0>3}_BER_{:.2f}_ACC_{:.2f}.pth'.format(epoch, ber, acc)
    torch.save(model.state_dict(), save_path + pth_name)


if __name__ == '__main__':
    # 定义模型
    model = Back_VGG(channel=32, if_dropout=False)
    # 加载模型至GPU
    model.cuda()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    temp_dir = "results"
    now = time.localtime()
    nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)  # 这一步就是对时间进行格式化
    log_file_name = dataset_name + '_' + nowt + ".txt"

    # 加载训练文件路径
    image_paths = image_root
    gt_paths = gt_root
    mask_paths = mask_root
    gray_paths = gray_root
    edge_paths = edge_root
    test_images = test_img_path
    test_masks = test_label_path

    train_ds = Dataset_train(image_paths, gt_paths, mask_paths, gray_paths, edge_paths)
    test_ds = Dataset_test(test_images, test_masks)

    train_loader = data.DataLoader(train_ds, batch_size=batchsize)
    test_loader = data.DataLoader(test_ds, batch_size=1)

    # 定义损失函数 1.交叉熵损失  2.**损失
    CE = torch.nn.BCELoss()
    smooth_loss = smoothness_loss()
    # 训练阶段
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, test_loader, model, optimizer, epoch)
    # 删除缓存文件夹
    shutil.rmtree(temp_dir)
