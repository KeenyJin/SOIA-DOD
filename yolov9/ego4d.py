import sys, os, torch, json
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

sys.path.append('/home/jim0228')
import custom.utils
from custom.dataset import Ego4dSTAImage
from custom.sta_metrics import OverallMeanAveragePrecision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

model_path = "runs/train/exp"
print(model_path)
print('Checkpoint epoch:', torch.load(os.path.join(model_path, "weights/best.pt"), map_location='cpu')['epoch'])
model = DetectMultiBackend(os.path.join(model_path, "weights/best.pt"),
                           data='../custom/ego4d_yolo.yaml',
                           device=torch.device('cuda:0'), dnn=False, fp16=False)

bs = 1
max_det = 40
iou_thres = 0.7
conf_thres = 0.0001
imgsz = 1024
print('image size (long):', imgsz)

# model.eval()

names = model.names
model.warmup(imgsz=(1, 3, int(1080 * imgsz / 1440), int(1440 * imgsz / 1440)))

dataset_val = Ego4dSTAImage(
    img_folder='../sta/images/',
    vid_folder='../sta/lmdb/',
    anno_file='../ego4d_data/v2/annotations/fho_sta_val.json',
    num_frames=8,
    sampling_duration=30,
    transform=None,
    use_vid=False
)
val_id2uid = dataset_val.get_id2uid()
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, bs, drop_last=False)
data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=custom.utils.collate_fn, num_workers=6)

with torch.no_grad():
    res_lst = dict()
    for images, targets in tqdm(data_loader_val, total=len(data_loader_val)):
        imgs = []
        for img in images:
            w, h = img.size
            r = imgsz / max(w, h)
            imgs.append(transforms.Resize((int(h * r), int(w * r)))(transforms.ToTensor()(img)))
        imgs = torch.stack(imgs, dim=0).to(model.device)
        pred = model(imgs)
        pred = pred[0][1]
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(imgs.shape[2:], det[:, :4], targets[0]['orig_size']).round()

                # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     print(f"{n} {names[int(c)]}{'s' * (n > 1)}, ")

                # Write results
                image_id = val_id2uid[targets[i]['image_id'].item()]
                res = []
                for *xyxy, conf, cls in det:
                    res.append([conf.item(), torch.stack(xyxy, dim=0).tolist(), cls.int().item()])
                res_lst[image_id] = [
                    {"score": s, "noun_category_id": c, "box": b, "verb_category_id": 0, "time_to_contact": 0}
                    for s, b, c in res
                ]

save_file = os.path.join(model_path, 'sta_results.json')
with open(save_file, 'w') as f:
    json.dump({'results': res_lst}, f)

# with open(save_file, 'r') as f:
#     res_lst = json.load(f)["results"]
#
with open('../ego4d_data/v2/annotations/fho_sta_val.json', 'r') as f:
    annotations = json.load(f)
assert len(annotations["annotations"]) == len(res_lst)

top_k = [1, 5, 10, 15, 20, 25, 30]
aps = [OverallMeanAveragePrecision(top_k=k) for k in top_k]
print('top k:', *top_k)
for ann in tqdm(annotations['annotations']):
    uid = ann['uid']
    gt = {
        'boxes': np.vstack([x['box'] for x in ann['objects']]),
        'nouns': np.array([x['noun_category_id'] for x in ann['objects']]),
        'verbs': np.array([x['verb_category_id'] for x in ann['objects']]),
        'ttcs': np.array([x['time_to_contact'] for x in ann['objects']])
    }
    prediction = res_lst[uid]
    if len(prediction) > 0:
        pred = {
            'boxes': np.vstack([x['box'] for x in prediction]),
            'nouns': np.array([x['noun_category_id'] for x in prediction]),
            'verbs': np.array([x['verb_category_id'] for x in prediction]),
            'ttcs': np.array([x['time_to_contact'] for x in prediction]),  # time_to_contact, not ttc
            'scores': np.array([x['score'] for x in prediction])
        }
    else:
        pred = {}

    for ap in aps:
        ap.add(pred, gt)

for k, ap in enumerate(aps):
    scores = ap.evaluate()
    names = ap.get_names()
    names[-1] = "* " + names[-1]
    print(f'top k: {top_k[k]}=====================')
    for name, val in zip(names, scores):
        print(f"{name}: {val:0.2f}")
    print('* metric used to score submissions for the challenge\n')
