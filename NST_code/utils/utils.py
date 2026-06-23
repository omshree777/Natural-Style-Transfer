from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(root))
        self.files=[p for p in self.files if p.endswith(('.jpg','.png','.jpeg'))]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        image_path=os.path.join(self.root,self.files[index])
        image=Image.open(image_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image
def get_transforms(size,crop,final_size):
    transformlist=[]
    if size < 0:
        transformlist.append(transforms.Resize(size))
    if crop:
        transformlist.append(transforms.RandomCrop(final_size))
    else:
        transformlist.append(transforms.Resize(final_size))
    transformlist.append(transforms.ToTensor())
    return transforms.Compose(transformlist)

def adaptive_instance_normalization(content_feat,style_feat):
    #[batch_size,channels,h,w]
    size=content_feat.size()
    style_mean,style_std=calc_mean_std(style_feat)
    content_mean,content_std=calc_mean_std(content_feat)
    normalized_content_feat= (content_feat - content_mean.expand(size))/ content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)
def calc_mean_std(feat,eps=1e-5):
    #[batch_size,channels,h,w]
    size=feat.size()
    assert(len(feat)==4)
    batch_size,channels=size[:2]
    feat_mean=feat.view(batch_size,channels,-1).mean(dim=2).view(batch_size,channels,1,1)
    feat_var=feat.view(batch_size,channels,-1).var(dim=2,unbiased=False) + eps
    feat_std=feat_var.sqrt().view(batch_size,channels,1,1)
    return feat_mean,feat_std