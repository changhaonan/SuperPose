import h5py
import tqdm
import torch
import logging

from torch.utils.data import DataLoader

confs = {
    'superpoint': {
        'output': 'feats-spp',
        'model': {
            'name': 'spp_det',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_h': 512,
            'resize_w': 512
        },
        'conf': {
            'descriptor_dim': 256,
            'nms_radius': 3,
            'max_keypoints': 4096,
            'keypoints_threshold': 0.6
        }
    }
}


@torch.no_grad()
def keypoints_filter(keypoints, seg):
    """filter keypoints by segmentation"""
    keypoints_int = keypoints.long()
    width, height = seg.shape[1], seg.shape[0]
    valid_indicator = torch.ones((keypoints.shape[0]), dtype=torch.bool)
    valid_indicator[keypoints_int[:, 0] > width] = False
    valid_indicator[keypoints_int[:, 1] > height] = False
    valid_indicator[seg[keypoints_int[:, 1], keypoints_int[:, 0]] == 0] = False
    return valid_indicator


@torch.no_grad()
def spp(img_lists, seg_lists, feature_out, cfg):
    """extract keypoints info by superpoint and filter by segmentation"""
    from src.utils.model_io import load_network
    from src.models.extractors.SuperPoint.superpoint import SuperPoint as spp_det
    from src.datasets.normalized_dataset_ext import NormalizedDatasetExt
    
    conf = confs[cfg.network.detection]
    model = spp_det(conf['conf']).cuda()
    model.eval()
    load_network(model, cfg.network.detection_model_path, force=True)

    dataset = NormalizedDatasetExt(img_lists, seg_lists, conf['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)

    feature_file = h5py.File(feature_out, 'w')
    logging.info(f'Exporting features to {feature_out}')
    for data in tqdm.tqdm(loader):
        inp = data['image'].cuda()
        seg = data['seg'].cuda()
        pred = model(inp)

        # get indicator
        valid_indicator = keypoints_filter(pred['keypoints'][0], seg[0][0])

        pred = {
            'keypoints' : pred['keypoints'][0][valid_indicator].cpu().numpy(),
            'scores' : pred['scores'][0][valid_indicator].cpu().numpy(),
            'descriptors' : pred['descriptors'][0][:, valid_indicator].cpu().numpy()
        }
        pred['image_size'] = data['size'][0].numpy()

        grp = feature_file.create_group(data['path'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
        
        del pred
    
    feature_file.close()
    logging.info('Finishing exporting features.')


def main(img_lists, seg_lists, feature_out, cfg):
    if cfg.network.detection == 'superpoint':
        spp(img_lists, seg_lists, feature_out, cfg)
    else:
        raise NotImplementedError