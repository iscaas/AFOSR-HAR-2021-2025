_base_ = 'rtmdet_s_8xb32-300e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=4))

train_cfg = dict(max_epochs=50)
auto_scale_lr = dict(base_batch_size=128)

# Modify dataset related settings
data_root = 'data/coco/Aerial_Video_Object_Detection/'
'''
metainfo = {
    'classes': ('None', 'Bike/Bicycle', 'Car', 'Carrying_object', 'Person' ),
    'palette': [
        (0,0,0), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)
    ]
}
'''
metainfo = {
    'classes': ('Box', 'Car', 'Person', 'parking lot' ),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)
    ]
}

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/')))

test_dataloader = val_dataloader

# Modify metric related settings
# validation evaluator
val_evaluator = dict(ann_file=data_root + 'valid/annotations.json')

# test evaluator
test_evaluator = val_evaluator

# We can use the pre-trained Deformable Detr model to obtain higher performance
#load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'