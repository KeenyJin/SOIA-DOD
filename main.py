import os
import argparse
import random
import json
import time
import math
import sys
import datetime
import yaml
import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm

import util as utils
from model import build_model
from dataset import build_dataset, batch_images_transform
from sta_metrics import OverallMeanAveragePrecision

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('Set SOIA-DOD', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, default='configs/config.yaml')

    # training parameters
    parser.add_argument('--output_dir', type=str, default='./logs', help='path where to save, empty for no saving')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, help='checkpoint file to resume training')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    # distributed training parameters
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', type=int, default=0, help='number of distributed processes')
    parser.add_argument('--amp', action='store_true', help="Train with mixed precision")

    return parser


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed} at rank {utils.get_rank()}")


@torch.no_grad()
def visualize(model, dataset, device, output_dir, cfg, clip_size, clip_preprocess):
    model.eval()
    id2uid = dataset.get_id2uid()

    visualize_path = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualize_path, exist_ok=True)

    det_num = 5  # The number of detections you are going to visualize
    eval_idxs = list(range(0, len(dataset), 2500))  # Indices of the data you are going to visualize

    gts, preds = dict(), dict()
    for idx in tqdm(eval_idxs):
        image, _target = dataset[idx]
        image_yolo, image_clip = batch_images_transform([image], cfg, clip_size, clip_preprocess, device=device)
        target = [{k: utils.to_device(v, device) for k, v in _target.items()}]
        uid = id2uid[_target['image_id'].item()]

        with torch.cuda.amp.autocast(enabled=cfg['amp']):
            _, detections = model(images_yolo=image_yolo, images_clip=image_clip, targets=target)

        gt_boxes = _target["boxes"].tolist()
        gt_nouns = _target["labels"].tolist()
        gt_verbs = _target["verbs"].tolist()
        gt_ttcs = _target["ttcs"].tolist()

        gt = []
        for box, noun, verb, ttc in zip(gt_boxes, gt_nouns, gt_verbs, gt_ttcs):
            gt.append({
                "box": box, "noun_category_id": noun, "verb_category_id": verb, "time_to_contact": ttc
            })
        gts[uid] = gt

        fig, ax = plt.subplots()
        ax.imshow(image)
        for gt_box in gt_boxes:
            rect = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        plt.axis('off')
        plt.savefig(os.path.join(visualize_path, f'{uid}-gt.png'),
                    bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()

        boxes = [box.tolist() for box in detections[0]['boxes'][:det_num]]
        nouns = [noun.item() for noun in detections[0]['nouns'][:det_num]]
        verbs = [verb.item() for verb in detections[0]['verbs'][:det_num]]
        ttcs = [ttc.item() for ttc in detections[0]['ttcs'][:det_num]]

        pred = []
        for box, noun, verb, ttc in zip(boxes, nouns, verbs, ttcs):
            pred.append({
                "box": box, "noun_category_id": noun, "verb_category_id": verb, "time_to_contact": ttc
            })
        preds[uid] = pred

        fig, ax = plt.subplots()
        ax.imshow(image)
        colors = ('r', 'b', 'y', 'c', 'm', 'k', 'g')
        for i, box in enumerate(boxes):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none')
            ax.add_patch(rect)
        plt.axis('off')
        plt.savefig(os.path.join(visualize_path, f'{uid}-pred.png'),
                    bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()

    with open(os.path.join(visualize_path, "results.json"), 'w') as f:
        json.dump({'ground_truth': gts, 'predictions': preds}, f, indent=2)


@torch.no_grad()
def test(model, device, output_dir, cfg, clip_size, clip_preprocess, epoch):
    dataset_test = build_dataset(image_set='test_unannotated', cfg=cfg)
    id2uid = dataset_test.get_id2uid()
    if cfg['distributed']:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, cfg['batch_size_per_gpu_val'], drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                                 collate_fn=utils.collate_fn, num_workers=cfg['num_workers'])

    model.eval()

    res_dict = dict()
    for idx, batch in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        images, targets = batch
        images_yolo, images_clip = batch_images_transform(images, cfg, clip_size, clip_preprocess, device=device)
        targets = [{k: utils.to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=cfg['amp']):
            _, detections = model(images_yolo=images_yolo, images_clip=images_clip, targets=targets)

        res = {target['image_id'].item(): output for target, output in zip(targets, detections)}
        for id, result in res.items():
            r_lst = []
            for score, noun, box, verb, ttc in zip(result["scores"], result["nouns"], result["boxes"], result["verbs"],
                                                   result["ttcs"]):
                r_lst.append({"score": score.item(), "noun_category_id": noun.item(), "box": box.tolist(),
                              "verb_category_id": verb.item(), "time_to_contact": ttc.item()})
            res_dict[id2uid[id]] = r_lst

    if not cfg['distributed']:
        with open(os.path.join(output_dir, "results", f"test_epoch{epoch}.json"), 'w') as f:
            json.dump({'version': 2.0, 'challenge': 'ego4d_short_term_object_interaction_anticipation',
                       'results': res_dict}, f)
        return

    with open(os.path.join(output_dir, "results", f"test_epoch{epoch}_rank{utils.get_rank()}.json"), 'w') as f:
        json.dump({'results': res_dict}, f)

    if utils.is_main_process():
        while True:
            time.sleep(1)
            dir_lst = os.listdir(os.path.join(output_dir, "results"))
            file_not_ready = False
            results = []
            for result in dir_lst:
                if f"test_epoch{epoch}" in result:
                    results.append(result)
            if len(results) == cfg['world_size']:
                all_results = {}
                for file in results:
                    try:
                        with open(os.path.join(output_dir, "results", file), 'r') as f:
                            all_results.update(json.load(f)["results"])
                            print(file, 'loaded')
                    except:
                        file_not_ready = True
                        break
                if file_not_ready:
                    continue
                save_file = os.path.join(output_dir, "results", f"test_epoch{epoch}.json")
                with open(save_file, 'w') as f:
                    json.dump({'version': 2.0, 'challenge': 'ego4d_short_term_object_interaction_anticipation',
                               'results': all_results}, f)
                    print(save_file, 'saved')
                    return


def mAP(all_results, cfg, epoch, output_dir):
    ap = OverallMeanAveragePrecision(top_k=5)
    with open(os.path.join(cfg['anno_path'], 'fho_sta_val.json'), 'r') as f:
        annotations = json.load(f)
    assert len(annotations["annotations"]) == len(all_results)
    for ann in tqdm(annotations['annotations']):
        uid = ann['uid']
        gt = {
            'boxes': np.vstack([x['box'] for x in ann['objects']]),
            'nouns': np.array([x['noun_category_id'] for x in ann['objects']]),
            'verbs': np.array([x['verb_category_id'] for x in ann['objects']]),
            'ttcs': np.array([x['time_to_contact'] for x in ann['objects']])
        }
        prediction = all_results[uid]
        if len(prediction) > 0:
            pred = {
                'boxes': np.vstack([x['box'] for x in prediction]),
                'nouns': np.array([x['noun_category_id'] for x in prediction]),
                'verbs': np.array([x['verb_category_id'] for x in prediction]),
                'ttcs': np.array([x['time_to_contact'] for x in prediction]),
                'scores': np.array([x['score'] for x in prediction])
            }
        else:
            pred = {}
        ap.add(pred, gt)
    scores = ap.evaluate()
    names = ap.get_names()
    score = {epoch: dict()}
    for name, val in zip(names, scores):
        print(f"{name}: {val}")
        score[epoch][name] = val

    map_file = os.path.join(output_dir, "map.json")
    if os.path.isfile(map_file):
        with open(map_file, 'r') as f:
            score.update(json.load(f))
    with open(map_file, 'w') as f:
        json.dump(score, f, indent=4)
        print(map_file, 'saved')


@torch.no_grad()
def evaluate(model, dataloader, device, output_dir, cfg, id2uid, clip_size, clip_preprocess, epoch):
    model.eval()

    res_dict = dict()
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, targets = batch
        images_yolo, images_clip = batch_images_transform(images, cfg, clip_size, clip_preprocess, device=device)
        targets = [{k: utils.to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=cfg['amp']):
            _, detections = model(images_yolo=images_yolo, images_clip=images_clip, targets=targets)

        res = {target['image_id'].item(): output for target, output in zip(targets, detections)}
        for id, result in res.items():
            r_lst = []
            for score, noun, box, verb, ttc in zip(result["scores"], result["nouns"], result["boxes"], result["verbs"],
                                                   result["ttcs"]):
                r_lst.append({"score": score.item(), "noun_category_id": noun.item(), "box": box.tolist(),
                              "verb_category_id": verb.item(), "time_to_contact": ttc.item()})
            res_dict[id2uid[id]] = r_lst

    if not cfg['distributed']:
        mAP(res_dict, cfg, epoch, output_dir)
        return

    save_file = os.path.join(output_dir, "results", f"val_epoch{epoch}_rank{utils.get_rank()}.json")
    with open(save_file, 'w') as f:
        json.dump(res_dict, f)

    if utils.is_main_process():
        while True:
            time.sleep(1)
            dir_lst = os.listdir(os.path.join(output_dir, "results"))
            file_not_ready = False
            results = []
            for result in dir_lst:
                if f"val_epoch{epoch}" in result:
                    results.append(result)
            if len(results) == cfg['world_size']:
                all_results = {}
                for file in results:
                    try:
                        with open(os.path.join(output_dir, "results", file), 'r') as f:
                            all_results.update(json.load(f))
                            print(file, 'loaded')
                    except:
                        file_not_ready = True
                        break
                if file_not_ready:
                    continue
                mAP(all_results, cfg, epoch, output_dir)
                return


def main(args):
    utils.init_distributed_mode(args)
    print("Loading config file from {}".format(args.config_file))
    with open(args.config_file, 'r') as f:
        cfg = yaml.full_load(f)
    cfg.update(vars(args))
    if args.rank == 0:
        if args.test:
            save_cfg_path = os.path.join(args.output_dir, "config_test.yaml")
        elif args.eval:
            save_cfg_path = os.path.join(args.output_dir, "config_val.yaml")
        else:
            save_cfg_path = os.path.join(args.output_dir, "config_train.yaml")
        print("Saving config file to {}".format(save_cfg_path))
        with open(save_cfg_path, 'w') as f:
            yaml.dump(cfg, f)
    print(cfg)

    device = torch.device(args.device)
    set_seed(args.seed + utils.get_rank())

    model, clip_preprocess = build_model(cfg, device)
    clip_size = model.image_backbone.input_resolution

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    param_dicts = utils.get_param_dict(cfg, model_without_ddp)
    if cfg['optimizer'] not in ("Adam", "AdamW", "RAdam"):
        raise NotImplementedError
    print('Optimizer:', cfg['optimizer'], 'lr:', cfg['lr'], 'lr_backbone:', cfg['lr_backbone'], 'weight_decay:',
          cfg['weight_decay'])
    optimizer = getattr(torch.optim, cfg['optimizer'])(param_dicts, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    if args.distributed:
        if not (args.eval or args.test):
            dataset_train = build_dataset(image_set='train', cfg=cfg)
            sampler_train = DistributedSampler(dataset_train)
            batchsampler_train = torch.utils.data.BatchSampler(sampler_train, cfg['batch_size_per_gpu_train'],
                                                                drop_last=True)
            dataloader_train = DataLoader(dataset_train, batch_sampler=batchsampler_train,
                                          collate_fn=utils.collate_fn, num_workers=args.num_workers)
        if not args.test:
            dataset_val = build_dataset(image_set='val', cfg=cfg)
            val_id2uid = dataset_val.get_id2uid()
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            batchsampler_val = torch.utils.data.BatchSampler(sampler_val, cfg['batch_size_per_gpu_val'],
                                                             drop_last=False)
            dataloader_val = DataLoader(dataset_val, batch_sampler=batchsampler_val,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers)
    else:
        if not (args.eval or args.test):
            dataset_train = build_dataset(image_set='train', cfg=cfg)
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            batchsampler_train = torch.utils.data.BatchSampler(sampler_train, cfg['batch_size_per_gpu_train'],
                                                               drop_last=True)
            dataloader_train = DataLoader(dataset_train, batch_sampler=batchsampler_train,
                                          collate_fn=utils.collate_fn, num_workers=args.num_workers)
        if not args.test:
            dataset_val = build_dataset(image_set='val', cfg=cfg)
            val_id2uid = dataset_val.get_id2uid()
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            batchsampler_val = torch.utils.data.BatchSampler(sampler_val, cfg['batch_size_per_gpu_val'],
                                                             drop_last=False)
            dataloader_val = DataLoader(dataset_val, batch_sampler=batchsampler_val,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if not (args.eval or args.test):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['lr_drop'])

    start_epoch = 0
    if (not args.resume) and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        print("=> Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if not (args.eval or args.test) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: ' + str(n_parameters))
    print("params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    if args.visualize and utils.is_main_process():
        visualize(model, dataset_val, device, args.output_dir, cfg, clip_size, clip_preprocess)
        return

    if args.test:
        print("Start test")
        os.environ['EVAL_FLAG'] = 'TRUE'
        print(f'Epoch: {start_epoch - 1}')
        test(model, device, args.output_dir, cfg=cfg, clip_size=clip_size, clip_preprocess=clip_preprocess,
             epoch=start_epoch - 1)
        return

    if args.eval:
        print("Start evaluation")
        os.environ['EVAL_FLAG'] = 'TRUE'
        print(f'Epoch: {start_epoch - 1}')
        evaluate(model, dataloader_val, device, args.output_dir, id2uid=val_id2uid, epoch=start_epoch - 1,
                 cfg=cfg, clip_size=clip_size, clip_preprocess=clip_preprocess)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        losses_log = 0
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        model.train()
        if args.distributed:
            model.module.yolo.eval()
        else:
            model.yolo.eval()
        for idx, batch in enumerate(dataloader_train):
            images, targets = batch
            images_yolo, images_clip = batch_images_transform(images, cfg, clip_size, clip_preprocess, device=device)
            targets = [{k: utils.to_device(v, device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=cfg['amp']):
                losses, _ = model(images_yolo=images_yolo, images_clip=images_clip, targets=targets)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(losses)
            losses_reduced = sum(loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            losses = sum(losses.values())

            if args.amp:  # amp backward function
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # original backward function
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            losses_log += losses
            if idx % cfg['log_interval'] == 0:
                if idx != 0:
                    print(f'\nEpoch {epoch} [{idx:5d}/{len(dataloader_train)}] Avg loss:',
                          losses_log / cfg['log_interval'], f'\n[Step {idx:5d}] loss:', loss_dict_reduced)
                else:
                    print(f'\nEpoch {epoch} [{idx:5d}/{len(dataloader_train)}] loss:',
                          losses_log, f'\n[Step {idx:5d}] loss:', loss_dict_reduced)
                losses_log = 0

        lr_scheduler.step()

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print(f'Epoch {epoch} training time: {epoch_time_str}')

        if args.output_dir:
            checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth'),
                                os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth')]
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }
                utils.save_on_master(weights, checkpoint_path)

        evaluate(model, dataloader_val, device, args.output_dir, id2uid=val_id2uid, epoch=epoch, cfg=cfg,
                 clip_size=clip_size, clip_preprocess=clip_preprocess)

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print(f'Epoch {epoch} total time: {epoch_time_str}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time: {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script for SOID-DOD', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    main(args)
