import torch
import torch.nn as nn
from torchvision.io import decode_image, write_jpeg
from torchvision.transforms import v2



class ImagePatcher(nn.Module):
    """
    forward pass: 
    image goes in -> patches come out 
    if image is shape [W,H,3], patch = 16 we want the resulting tensor to be 
    WH/P^2,       



    """
    def __init__(self,tgt_img_size: tuple[int,int], patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.transforms = v2.Compose([
            v2.Resize(tgt_img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # assumes only 3 channel 
        self.linear = nn.Linear((3*self.patch_size**2),self.embed_dim)


    def preprocess(self, img: torch.Tensor):
        # assumes img is [1,3,H,W]
        # ignore channels past rgb for now
        img = img[:,:3,:,:]

        resized_img = self.transforms(img)
        B,C,H,W = resized_img.shape
        P = self.patch_size 
        img_patches = resized_img.view(B,C, H//P, P, W//P, P)
        img_permuted = torch.permute(img_patches,(0,2,4,1,3,5))
        img_flattened = img_permuted.reshape((B,H*W//(P*P),C*P*P))

        return img_flattened # [W*H/P^2, C*P^2]

    def forward(self, input):
        return self.linear(input)
        


if __name__ == "__main__":

                # (C, H, W)
    cat = decode_image("data/cat.jpg")
    batch_cats = torch.concat((cat.unsqueeze(dim=0),cat.unsqueeze(dim=0)))
    transform = v2.Resize((224,224))
    
    print(cat.shape)
    cats_resized = transform(batch_cats)
    B,C,H,W = cats_resized.shape
    P = 16 
    cats_patched = cats_resized.view(B,C, H//P, P, W//P, P)    
    cats_shifted = torch.permute(cats_patched,(0,2,4,1,3,5))
    cats_flattened = cats_shifted.reshape((B,H*W//(P*P),C*P*P))


    patcher = ImagePatcher((224,224),16,512)

    input_cats = patcher.preprocess(batch_cats)
    ret = patcher(input_cats)
    print()
    #     # 3, 224 ,224 
    # print(transform(cat).shape)
    #     # 3, (224/16) , 16, (224/16), 16 
    
    # cat2 = transform(cat)
    # C,H,W = cat2.shape
    # P = 16

    # patched_cat = cat2.view(C, H//P, P, W//P, P)
    #                                         # h,w, c,P,P
    # shifted_cat = torch.permute(patched_cat,(1,3,0,2,4))

    #                             # N patches, and C*P*P feature dim
    # flat_cat = shifted_cat.reshape((H*W//(P*P),C*P*P))



    # for x in range((shifted_cat.shape[1])):
    #     for y in range((shifted_cat.shape[2])):
    #         gt_patch = cat2[:,P*x:P*(x+1),P*y:P*(y+1)]
    #         curr_patch = shifted_cat[:,x,y,:,:]
            
    #         if torch.sum(gt_patch - curr_patch) != 0:
    #             print(f"patch {x}, {y} is not the same!")

    #         write_jpeg(gt_patch,f"output/patch_gt{x}_{y}.jpg")
    #         write_jpeg(curr_patch,f"output/patch_pred{x}_{y}.jpg")

    # print(patched_cat.shape) 
     
    
