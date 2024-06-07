import os
import lmdb
import cv2
import io
import torch
import json
from typing import List
from PIL import Image
from pathlib import Path
from torchvision import transforms
import numpy as np


class Ego4DHLMDB():
    def __init__(self, path_to_root: Path, readonly=False, lock=False,
                 frame_template="{video_id:s}_{frame_number:010d}", map_size=1099511627776) -> None:
        self.environments = {}
        self.path_to_root = path_to_root
        if isinstance(self.path_to_root, str):
            self.path_to_root = Path(self.path_to_root)
        self.path_to_root.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.lock = lock
        self.map_size = map_size
        self.frame_template = frame_template

    def _get_parent(self, parent: str) -> lmdb.Environment:
        return lmdb.open(str(self.path_to_root / parent), map_size=self.map_size, readonly=self.readonly,
                         lock=self.lock)

    def put_batch(self, video_id: str, frames: List[int], data: List[np.ndarray]) -> None:
        with self._get_parent(video_id) as env:
            with env.begin(write=True) as txn:
                for frame, value in zip(frames, data):
                    if value is not None:
                        txn.put(self.frame_template.format(video_id=video_id, frame_number=frame).encode(),
                                cv2.imencode('.jpg', value)[1])

    def put(self, video_id: str, frame: int, data: np.ndarray) -> None:
        if data is not None:
            with self._get_parent(video_id) as env:
                with env.begin(write=True) as txn:
                    txn.put(self.frame_template.format(video_id=video_id, frame_number=frame).encode(),
                            cv2.imencode('.jpg', data)[1])

    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())

                file_bytes = np.asarray(
                    bytearray(io.BytesIO(data).read()), dtype=np.uint8
                )
                return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())
                    file_bytes = np.asarray(
                        bytearray(io.BytesIO(data).read()), dtype=np.uint8
                    )
                    out.append(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
            return out

    def get_existing_keys(self):
        keys = []
        for parent in self.path_to_root.iterdir():
            with self._get_parent(parent.name) as env:
                with env.begin() as txn:
                    keys += list(txn.cursor().iternext(values=False))
        return keys


class Ego4DHLMDB_STA_Still_Video(Ego4DHLMDB):
    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())

                file_bytes = np.asarray(
                    bytearray(io.BytesIO(data).read()), dtype=np.uint8
                )
                return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())
                    file_bytes = np.asarray(
                        bytearray(io.BytesIO(data).read()), dtype=np.uint8
                    )
                    out.append(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
            return out


def prepare(image, target):
    w, h = image.size

    image_id = target["image_id"]
    image_id = torch.tensor([image_id])

    if target["annotations"] is not None:
        anno = target["annotations"]

        boxes = [obj["box"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        nouns = [obj["noun_category_id"] for obj in anno]
        nouns = torch.tensor(nouns, dtype=torch.int64)

        verbs = [obj["verb_category_id"] for obj in anno]
        verbs = torch.tensor(verbs, dtype=torch.int64)

        ttcs = [obj["time_to_contact"] for obj in anno]
        ttcs = torch.tensor(ttcs, dtype=torch.float32)

        target = {"boxes": boxes, "labels": nouns, "image_id": image_id, "orig_size": torch.as_tensor([int(h), int(w)]),
                  "size": torch.as_tensor([int(h), int(w)]), "verbs": verbs, "ttcs": ttcs}
    else:
        target = {"image_id": image_id, "orig_size": torch.as_tensor([int(h), int(w)]),
                  "size": torch.as_tensor([int(h), int(w)])}

    return image, target


class Ego4DSTAImage(torch.utils.data.Dataset):
    def __init__(self, img_folder, anno_file, is_test=False):
        self._img_folder = img_folder
        with open(anno_file) as f:
            self._annotations = json.load(f)
        self.is_test = is_test
        if not self.is_test:
            self._cleanup()
        print("Ego4DSTAImage loaded")
        self.id2uid = []
        for annotation in self._annotations['annotations']:
            self.id2uid.append(f"{annotation['video_uid']}_{annotation['frame']:07d}")

    def _load_image(self, video_id, frame):
        """ Load images from lmdb. """
        return Image.open(os.path.join(self._img_folder, f"{video_id}_{frame:07d}.jpg"))

    def _cleanup(self):
        removed_boxes = 0
        removed_anns = 0
        anns = self._annotations['annotations']
        self._annotations['annotations'] = []
        for i in range(len(anns)):
            ann = anns[i]
            if 'objects' in ann:
                _obj = []
                for obj in ann['objects']:
                    box = obj['box']
                    if box[2] - box[0] > 0 and box[3] - box[1] > 0:
                        _obj.append(obj)
                    elif box[2] - box[0] < 0 and box[3] - box[1] < 0:
                        obj['box'] = [box[2], box[3], box[0], box[1]]
                        _obj.append(obj)
                    else:
                        removed_boxes += 1

                if len(_obj) > 0:
                    ann['objects'] = _obj
                    self._annotations['annotations'].append(ann)
                else:
                    removed_anns += 1

        print(f"Removed {removed_boxes} degenerate objects and {removed_anns} annotations with no objects")

    def __getitem__(self, idx):
        annotation = self._annotations['annotations'][idx]
        image = self._load_image(annotation['video_uid'], annotation['frame'])
        target = None
        if not self.is_test:
            target = annotation['objects']
        image_id = idx
        target = {'image_id': image_id, 'annotations': target}
        image, target = prepare(image, target)
        return image, target

    def __len__(self):
        return len(self._annotations['annotations'])

    def get_id2uid(self):
        return self.id2uid


def batch_images_transform(images, cfg, clip_size, clip_preprocess, device="cuda"):
    yolo_size = cfg['image_yolo_size']

    imgs_yolo = []
    for img in images:
        w, h = img.size
        r = yolo_size / max(w, h)
        imgs_yolo.append(transforms.Resize((int(h * r), int(w * r)))(transforms.ToTensor()(img)))

    if clip_size is None or clip_preprocess is None:
        return imgs_yolo, None

    _clip_preprocess = transforms.Compose([
        transforms.Resize((clip_size, clip_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Compose(clip_preprocess.transforms[2:]),
    ])
    imgs_clip = []
    for img in images:
        imgs_clip.append(_clip_preprocess(img))
    imgs_clip = torch.stack(imgs_clip, dim=0).to(device)

    return imgs_yolo, imgs_clip


def build_dataset(image_set, cfg):
    assert image_set in ['train', 'val', 'test_unannotated'], "image_set must be one of 'train', 'val', 'test_unannotated'"

    img_path = Path(cfg['img_path'])
    anno_file = Path(cfg['anno_path']) / f'fho_sta_{image_set}.json'

    dataset = Ego4DSTAImage(img_path, anno_file, is_test=image_set == 'test_unannotated')

    return dataset
