import torchvision
import torch
from dataset import PombeDataset
import transforms


def collate(batch):
    return tuple(zip(*batch))

def test_forward():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    root = "/Users/cjw/Desktop/SeanMovie"
    image_dir = "Images"
    mask_dir = "Masks"
    pdataset = PombeDataset(root, image_dir, mask_dir, transforms.get_transforms())
    data_loader = torch.utils.data.DataLoader(
        pdataset, batch_size=3, shuffle=True, num_workers=2,
        collate_fn=collate
    )
    
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k :v for k, v in t.items()} for t in targets]
    #output = model(images, targets)
    model.eval()
    xi, xt = pdataset.__getitem__(1)
    print(xi.shape)
    predictions = model(3*[xi])
    return predictions    
