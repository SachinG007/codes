import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from randaugment import RandomAugment
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import os
import json
import open_clip
import utils
import torch
import torch.distributed as dist
import re

class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   

def create_dataset(transform_test):
    min_scale = 0.5
    config = {}
    config['image_size'] = 384
    config['image_root'] = "/project_data/datasets/mscoco2014/"
    config['ann_root'] = "/project_data/datasets/mscoco2014/"
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # transform_train = transforms.Compose([                        
    #         transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(),
    #         RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
    #                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
    #         transforms.ToTensor(),
    #         normalize,
    #     ])        
    # transform_test = transforms.Compose([
    #     transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     normalize,
    #     ])  

    test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
    return test_dataset



model, _, transform_test = open_clip.create_model_and_transforms(
        "coca_ViT-B-32",
        "",
        device="cuda",
    )
test_dataset = create_dataset(transform_test)
test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
model = model.eval()
import fsspec
of = fsspec.open("/home/sachingo/datanet_private/logs/finetuned_coco_nofilter_mediumscale/checkpoints/epoch_10.pt", "rb")
# of = fsspec.open("/home/sachingo/new_openclip/open_clip/logs/finetuned_coco_tmars_mediumscale/checkpoints/epoch_10.pt", "rb")
result_dir = "./coco_eval/nofilter_medium_finetuned/"
with of as f:
    checkpoint = torch.load(f, map_location="cpu")
sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()

result = []
print(f"Len Data Loader : {len(test_loader)}")
for i, (image, image_id) in enumerate(test_loader):
    print(f"******Current batch  : {i}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        # import pdb;pdb.set_trace()
        image = image.cuda()
        generated = model.generate(image)
        for k in range(generated.shape[0]):
            answer = open_clip.decode(generated[k]).split("<end_of_text>")[0].replace("<start_of_text>", "")
            print(answer)
            result.append({"image_id": image_id[k].item(), "caption": answer})

def save_result(result, result_dir, filename, remove_duplicate=''):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file


test_result_file = save_result(result, result_dir, 'test', remove_duplicate='image_id')  

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

coco_test = coco_caption_eval('annotation/coco_gt',test_result_file,'test')