import clip
import torch
import torch.nn as nn
from torchvision.ops import box_iou
from copy import deepcopy

from util import MLP

import os, sys
yolov9_path = os.path.abspath('./yolov9')
if yolov9_path not in sys.path:
    sys.path.append(yolov9_path)
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import non_max_suppression, scale_boxes


class SOIA_DOD(nn.Module):
    def __init__(self, cfg, image_backbone, yolo, device=torch.device('cuda')):
        super(SOIA_DOD, self).__init__()
        self.cfg = cfg
        self.image_backbone = image_backbone
        self.image_feat_proj = nn.Linear(image_backbone.transformer.width, cfg['enc_embed_dim'])
        self.yolo = yolo
        self.yolo.eval()
        self.yolo.warmup(imgsz=(1, 3, int(1080 * cfg['image_yolo_size'] / 1440), int(1440 * cfg['image_yolo_size'] / 1440)))
        self.device = device

        self.obj_cls_encoder = nn.Embedding(cfg['noun_classes'], cfg['enc_embed_dim'])
        self.obj_bbox_encoder = nn.Sequential(
            nn.Linear(4, cfg['enc_embed_dim']),
            nn.ReLU(),
            nn.Linear(cfg['enc_embed_dim'], cfg['enc_embed_dim'])
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg['enc_embed_dim'], nhead=cfg['enc_num_heads'],
                                                   dim_feedforward=cfg['enc_ff_dim'], batch_first=True, norm_first=True)
        encoder = [deepcopy(encoder_layer) for _ in range(cfg['enc_num_layers'])]
        encoder.append(nn.LayerNorm(cfg['enc_embed_dim']))
        self.encoder = nn.ModuleList(encoder)

        self.obj_scorer = nn.Linear(cfg['enc_embed_dim'], 1)
        self.verb_predictor = nn.Linear(cfg['enc_embed_dim'], cfg['verb_classes'])
        self.ttc_predictor = nn.Sequential(
            MLP(cfg['enc_embed_dim'], cfg['enc_embed_dim'], 1, cfg['ttc_mlp_num_layers']),
            nn.Softplus()
        )

        self.obj_loss = nn.BCEWithLogitsLoss()
        verb_loss_weight = torch.ones(cfg['verb_classes']).float()
        self.verb_loss = nn.CrossEntropyLoss(weight=verb_loss_weight)
        self.ttc_loss = nn.SmoothL1Loss(beta=0.25)

        print(f'\nYOLO predicts {cfg["yolo_max_det"]} objects for each image')
        print(f'Top-{cfg["top_k_verb"]} verbs are predicted for each detection\n')

    def forward(self, images_yolo, images_clip, targets):
        # YOLOv9
        box, cls, yolo_score, box_orig = [], [], [], []
        with torch.no_grad():
            for i, image in enumerate(images_yolo):
                image = image.unsqueeze(0).to(self.device)
                obj_candidates = self.yolo(image)
                obj_candidates = obj_candidates[0][1]
                obj_candidates = non_max_suppression(obj_candidates, self.cfg['yolo_conf_thres'], self.cfg['yolo_iou_thres'],
                                                     None, False, max_det=self.cfg['yolo_max_det'])
                det = obj_candidates[0]
                det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], targets[i]['orig_size']).round()
                boxes = det[:, :4]
                box_orig.append(boxes)

                h, w = targets[i]['orig_size']
                box.append(torch.stack([boxes[:, 0] / w, boxes[:, 1] / h, boxes[:, 2] / w, boxes[:, 3] / h], dim=-1))

                yolo_score.append(det[:, 4])
                cls.append(det[:, 5].int())

                if det.size(0) < self.cfg['yolo_max_det']:
                    box[-1] = torch.cat((box[-1], torch.stack([torch.tensor([0.0, 0.0, 0.0, 0.0]).to(box[0])] * (self.cfg['yolo_max_det'] - det.size(0)))))
                    box_orig[-1] = torch.cat((box_orig[-1], torch.stack([torch.tensor([0.0, 0.0, 0.0, 0.0]).to(box_orig[0])] * (self.cfg['yolo_max_det'] - det.size(0)))))
                    cls[-1] = torch.cat((cls[-1], torch.tensor([82 for _ in range(self.cfg['yolo_max_det'] - det.size(0))]).to(cls[0])))
                    yolo_score[-1] = torch.cat((yolo_score[-1], torch.tensor([0.0 for _ in range(self.cfg['yolo_max_det'] - det.size(0))]).to(yolo_score[0])))
        box = torch.stack(box, dim=0).detach()
        cls = torch.stack(cls, dim=0).detach()
        yolo_score = torch.stack(yolo_score, dim=0).detach()
        box_orig = torch.stack(box_orig, dim=0).detach()

        obj_feat = self.obj_bbox_encoder(box) + self.obj_cls_encoder(cls)

        # CLIP
        x = self.image_backbone.conv1(images_clip.type(self.image_backbone.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.image_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.image_backbone.positional_embedding.to(x.dtype)
        x = self.image_backbone.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.image_backbone.transformer.resblocks[:-1](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        img_feat = self.image_feat_proj(x[:, 1:, :].float())
        num_grids = img_feat.size(1)

        x = torch.cat([img_feat, obj_feat], 1)

        # Encoder
        if self.training:
            obj_auxs, verb_auxs, ttc_auxs = [], [], []
            for idx, layer in enumerate(self.encoder):
                x = layer(x)
                if idx == self.cfg['enc_num_layers'] - 3 or idx == self.cfg['enc_num_layers'] - 2:
                    y = x[:, num_grids: num_grids + self.cfg['yolo_max_det'], :]
                    obj_auxs.append(self.obj_scorer(y).squeeze(-1))
                    verb_auxs.append(self.verb_predictor(y))
                    ttc_auxs.append(self.ttc_predictor(y).squeeze(-1))
        else:
            for idx, layer in enumerate(self.encoder):
                x = layer(x)
        x = x[:, num_grids: num_grids + self.cfg['yolo_max_det'], :]

        obj_score = self.obj_scorer(x).squeeze(-1)
        verb_preds = self.verb_predictor(x)
        ttc_preds = self.ttc_predictor(x).squeeze(-1)

        losses, detections = dict(), []

        # Training
        if self.training:
            target_boxes = [target['boxes'] for target in targets]
            target_objs = [target['labels'] for target in targets]
            target_verbs = [target['verbs'] for target in targets]
            target_ttcs = [target['ttcs'] for target in targets]

            obj_answer = torch.zeros_like(obj_score)
            verb_answer = torch.zeros_like(obj_score)
            ttc_answer = torch.zeros_like(obj_score)
            for i in range(len(targets)):
                for k, target_obj in enumerate(target_objs[i]):
                    for j in range(self.cfg['yolo_max_det']):
                        if target_obj == cls[i][j] and box_iou(target_boxes[i][k].unsqueeze(0), box_orig[i][j].unsqueeze(0)) > self.cfg['pos_sample_iou']:
                            obj_answer[i][j] = 1.0
                            verb_answer[i][j] = target_verbs[i][k]
                            ttc_answer[i][j] = target_ttcs[i][k]

            losses['object'] = self.obj_loss(obj_score, obj_answer).mean() * self.cfg['obj_loss_coef']

            positive_obj = obj_answer == 1.0

            if torch.where(positive_obj, 1, 0).sum().item() > 0:
                losses['verb'] = self.verb_loss(verb_preds[positive_obj], verb_answer[positive_obj].long()) * self.cfg['verb_loss_coef']
                losses['ttc'] = self.ttc_loss(ttc_preds[positive_obj], ttc_answer[positive_obj]) * self.cfg['ttc_loss_coef']

                losses['object_aux'], losses['verb_aux'], losses['ttc_aux'] = 0, 0, 0
                for obj_aux in obj_auxs:
                    losses['object_aux'] += (self.obj_loss(obj_aux, obj_answer).mean()
                                             * self.cfg['obj_loss_coef'] * self.cfg['aux_loss_coef'])
                for verb_aux in verb_auxs:
                    losses['verb_aux'] += (self.verb_loss(verb_aux[positive_obj], verb_answer[positive_obj].long())
                                           * self.cfg['verb_loss_coef'] * self.cfg['aux_loss_coef'])
                for idx, ttc_aux in enumerate(ttc_auxs):
                    losses['ttc_aux'] += (self.ttc_loss(ttc_aux[positive_obj], ttc_answer[positive_obj])
                                          * self.cfg['ttc_loss_coef'] * self.cfg['aux_loss_coef'])

        # Validation, Test
        else:
            batch_size, num_det, verb_classes = verb_preds.shape
            verb_preds = verb_preds.reshape(-1, verb_classes).softmax(dim=1)

            if self.cfg['top_k_verb'] == 1:
                verb_scores = torch.stack([torch.max(pred) for pred in verb_preds], dim=0).reshape(batch_size, num_det)
                verb_preds = torch.stack([torch.argmax(pred) for pred in verb_preds], dim=0).reshape(batch_size, num_det)
            else:
                verb_scores = torch.stack([torch.topk(pred, self.cfg['top_k_verb'])[0] for pred in verb_preds], dim=0)
                verb_preds = torch.stack([torch.topk(pred, self.cfg['top_k_verb'])[1] for pred in verb_preds], dim=0)
                verb_scores = verb_scores.reshape(batch_size, num_det, self.cfg['top_k_verb']).reshape(batch_size, -1)
                verb_preds = verb_preds.reshape(batch_size, num_det, self.cfg['top_k_verb']).reshape(batch_size, -1)
                yolo_score = yolo_score.repeat_interleave(self.cfg['top_k_verb'], dim=1)
                box_orig = box_orig.repeat_interleave(self.cfg['top_k_verb'], dim=1)
                cls = cls.repeat_interleave(self.cfg['top_k_verb'], dim=1)
                ttc_preds = ttc_preds.repeat_interleave(self.cfg['top_k_verb'], dim=1)

            detection_score = yolo_score * verb_scores

            for s, c, b, v, t in zip(detection_score, cls, box_orig, verb_preds, ttc_preds):
                result = dict()
                sorted_by_score = list(sorted(zip(s, c, b, v, t), key=lambda y: -y[0].item()))
                if self.cfg['test'] and len(sorted_by_score) > self.cfg['test_max_pred']:
                    sorted_by_score = sorted_by_score[:self.cfg['test_max_pred']]
                result["scores"] = [y[0] for y in sorted_by_score]
                result["nouns"] = [y[1] for y in sorted_by_score]
                result["boxes"] = [y[2] for y in sorted_by_score]
                result["verbs"] = [y[3] for y in sorted_by_score]
                result["ttcs"] = [y[4] for y in sorted_by_score]
                detections.append(result)

        return losses, detections


def build_model(cfg, device):
    print("Start building model")

    image_backbone, clip_preprocess = clip.load(cfg['backbone'], device=device)
    image_backbone = image_backbone.visual.float()
    print("CLIP loaded")

    yolo = DetectMultiBackend(cfg['yolo_checkpoint'], device=device, data='configs/ego4d_yolo.yaml',
                              dnn=False, fp16=False)
    for p in yolo.parameters():
        p.requires_grad = False
    print("YOLO loaded")

    model = SOIA_DOD(cfg, image_backbone, yolo, device=device)

    return model, clip_preprocess
