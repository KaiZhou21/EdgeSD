import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from config import image_root, gt_root, mask_root, gray_root, edge_root
import glob

transform_RGB = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像标准化
    # transforms.Normalize([0.517, 0.514, 0.492], [0.186, 0.173, 0.181]) # ISTD
])

transform_gray = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])


# ----------------Dataset类------------------------
class Dataset_train(data.Dataset):
    def __init__(self, image_path, gt_path, mask_path, gray_path, edge_path):
        self.image_path = image_path
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.gray_path = gray_path
        self.edge_path = edge_path

    def __getitem__(self, index):
        # 路径
        image = self.image_path[index]
        gt = self.gt_path[index]
        mask = self.mask_path[index]
        gray = self.gray_path[index]
        edge = self.edge_path[index]

        # 读取
        image = Image.open(image).convert('RGB')
        gt = Image.open(gt).convert('L')
        mask = Image.open(mask).convert('L')
        gray = Image.open(gray).convert('L')
        edge = Image.open(edge).convert('L')

        #
        if image.size == gt.size == mask.size == gray.size == edge.size:
            # 增强
            image = transform_RGB(image)
            gt = transform_gray(gt)
            mask = transform_gray(mask)
            gray = transform_gray(gray)
            edge = transform_gray(edge)
            return image, gt, mask, gray, edge
        else:
            print("-------训练数据内部出错--------")
            return 0

    def __len__(self):
        return len(self.image_path)


class Dataset_test(data.Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path

    def __getitem__(self, index):
        image = self.image_path[index]
        mask = self.mask_path[index]

        image = Image.open(image).convert('RGB')
        image = transform_RGB(image)

        mask = Image.open(mask).convert('L')
        mask = transform_gray(mask)


        return image, mask

    def __len__(self):
        return len(self.image_path)


if __name__ == '__main__':
    image_paths = sorted(glob.glob(image_root + "*.jpg"))
    gt_paths = sorted(glob.glob(gt_root + "*.png"))
    mask_paths = sorted(glob.glob(mask_root + "*.png"))
    gray_paths = sorted(glob.glob(gray_root + "*.jpg"))
    edge_paths = sorted(glob.glob(edge_root + "*.png"))

    train_ds = Dataset_train(image_paths, gt_paths, mask_paths, gray_paths, edge_paths)
    train_loader = data.DataLoader(train_ds, batch_size=8)
    for images, gts, masks, grays, edges in train_loader:
        print(images.shape)
        print(gts.shape)
        print(masks.shape)
        print(grays.shape)
        print(edges.shape)
