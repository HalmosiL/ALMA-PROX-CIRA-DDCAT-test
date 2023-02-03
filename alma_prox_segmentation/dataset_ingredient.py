from typing import Callable, Optional, Tuple

from sacred import Ingredient
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Cityscapes, VOCSegmentation
from torchvision.transforms import InterpolationMode

from utils import label_map_cityscapes
from dataset import SemDataSplit
from transforms import Compose, ToTensor, Normalize

dataset_ingredient = Ingredient('dataset')

@dataset_ingredient.named_config
def cityscapes():
    name = 'cityscapes'
    root = 'data/cityscapes'
    split = 'val'
    size = (1024, 2048)
    num_images = None


@dataset_ingredient.capture
def get_cityscapes(root: str, size: int, split: str,
                   num_images: Optional[int] = None, batch_size: int = 1) -> Tuple[DataLoader, Optional[Callable]]:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        label_map_cityscapes,
        transforms.PILToTensor(),
        transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    ])
    dataset = Cityscapes(root=root, split=split, target_type='semantic',
                         transform=transform, target_transform=target_transform)
    if num_images is not None:
        assert num_images <= len(dataset)
        dataset = Subset(dataset, indices=list(range(num_images)))
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    return loader, label_map_cityscapes


@dataset_ingredient.capture
def get_cityscapes_resized(root, size, split, batch_size=1):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    )

    image_list = root + split + ".txt"

    loader = torch.utils.data.DataLoader(   
        dataset=SemDataSplit(
            split=split,
            data_root=root,
            data_list=image_list,
            transform=val_transform
        ),
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    return loader, label_map_cityscapes

_loaders = {
    'cityscapes': get_cityscapes_resized
}


@dataset_ingredient.capture
def get_dataset(name: str) -> DataLoader:
    return _loaders[name]()
