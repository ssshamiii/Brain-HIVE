import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision.models import inception_v3, Inception_V3_Weights
import clip
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import scipy as sp
import os
from PIL import Image
import datetime
import json
import glob


@torch.no_grad()
def two_way_identification(
    all_brain_recons,
    all_images,
    model,
    preprocess,
    feature_layer=None,
    return_avg=True,
    device: torch.device = torch.device("cpu"),
):
    preds = model(
        torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device)
    )
    reals = model(
        torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device)
    )
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[: len(all_images), len(all_images) :]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    else:
        return success_cnt, len(all_images) - 1


def pixcorr(all_images, all_brain_recons):
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = (
        preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()
    )

    corrsum = 0
    n = min(len(all_images_flattened), len(all_brain_recons_flattened))
    for i in tqdm(range(n)):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[
            0
        ][1]
    corrmean = corrsum / n

    pixcorr = corrmean
    return pixcorr


def ssim(all_images, all_brain_recons):
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu())

    ssim_score = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images)):
        ssim_score.append(
            structural_similarity(
                rec,
                im,
                multichannel=True,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )

    ssim = np.mean(ssim_score)
    return ssim


def alexnet(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    from torchvision.models import alexnet, AlexNet_Weights

    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(
        alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]
    ).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_per_correct = two_way_identification(
        all_brain_recons.float(),
        all_images,
        alex_model,
        preprocess,
        "features.4",
        device=device,
    )
    alexnet2 = np.mean(all_per_correct)

    all_per_correct = two_way_identification(
        all_brain_recons.float(),
        all_images,
        alex_model,
        preprocess,
        "features.11",
        device=device,
    )
    alexnet5 = np.mean(all_per_correct)
    return alexnet2, alexnet5


def inception(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(
        inception_v3(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_per_correct = two_way_identification(
        all_brain_recons,
        all_images,
        inception_model,
        preprocess,
        "avgpool",
        device=device,
    )

    inception = np.mean(all_per_correct)
    return inception


def clip_(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    all_per_correct = two_way_identification(
        all_brain_recons,
        all_images,
        clip_model.encode_image,
        preprocess,
        None,
        device=device,
    )  # final layer
    clip_ = np.mean(all_per_correct)
    return clip_


def effnet(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(
        efficientnet_b1(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = eff_model(preprocess(all_images))["avgpool"]
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = eff_model(preprocess(all_brain_recons))["avgpool"]
    fake = fake.reshape(len(fake), -1).cpu().numpy()

    effnet = np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()
    return effnet


def swav(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(
        device
    )
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = swav_model(preprocess(all_images))["avgpool"]
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons))["avgpool"]
    fake = fake.reshape(len(fake), -1).cpu().numpy()

    swav = np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()
    return swav


def eval_images(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    real_images = real_images.to(device).float()
    fake_images = fake_images.to(device).float()

    pixcorrs = pixcorr(real_images, fake_images)
    ssims = ssim(real_images, fake_images)
    alex2, alex5 = alexnet(real_images, fake_images, device=device)
    inceptions = inception(real_images, fake_images, device=device)
    clips = clip_(real_images, fake_images, device=device)
    effnets = effnet(real_images, fake_images, device=device)
    swavs = swav(real_images, fake_images, device=device)

    return {
        "eval_pixcorr": pixcorrs.item(),
        "eval_ssim": ssims.item(),
        "eval_alex2": alex2.item(),
        "eval_alex5": alex5.item(),
        "eval_inception": inceptions.item(),
        "eval_clip": clips.item(),
        "eval_effnet": effnets.item(),
        "eval_swav": swavs.item(),
    }
