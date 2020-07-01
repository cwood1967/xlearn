from dataset import PombeDataset
import transforms

root = '/Users/cjw/Desktop/SeanMovie'
image_dir = 'Images'
mask_dir = 'Masks'

t = transforms.get_transforms()
p = PombeDataset(root, image_dir, mask_dir, t)
v = p.__getitem__(0)