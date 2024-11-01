_base_ = 'rtmdet_l_swin_b_4xb32-100e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=4))

train_cfg = dict(max_epochs=50)
auto_scale_lr = dict(base_batch_size=64)

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
#load_from = 'https://github.com/orange0-jp/orange-weights/releases/download/v0.1.0rtmdet-swin-convnext/rtmdet_l_swin_b_4xb32-100e_coco-0828ce5d.pth'