import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument('--root-dir', type=str, default='./',
                        help='Root dataset directory containing dataset_c1, dataset_c2, dataset_c3, dataset')
    parser.add_argument('--output-dir', type=str, default='./attention',
                        help='Root output directory for CAMs')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen-smooth', action='store_true',
                        help='Reduce noise by taking the first principal component')
    parser.add_argument('--method', type=str, default='gradcam++',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam',
                            'finercam'
                        ],
                        help='CAM method')
    return parser.parse_args()


def process_image(model, cam_algorithm, target_layers, image_path, device, aug_smooth, eigen_smooth):
    """Process a single image and return CAM, optionally GB and CAM+GB"""
    # Read original image
    orig_img = cv2.imread(image_path, 1)  # BGR
    if orig_img is None:
        raise ValueError(f"Cannot read image {image_path}")
    h, w = orig_img.shape[:2]

    # Convert to RGB float [0,1]
    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)

    targets = None
    with cam_algorithm(model=model, target_layers=target_layers) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=aug_smooth,
                            eigen_smooth=eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]

        # Resize CAM to original image size
        cam_resized = cv2.resize(grayscale_cam, (w, h))
        cam_image = show_cam_on_image(rgb_img, cam_resized, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # Optional: Guided Backpropagation & CAM+GB
    # gb_model = GuidedBackpropReLUModel(model=model, device=device)
    # gb = gb_model(input_tensor, target_category=None)
    # cam_mask = cv2.merge([cam_resized, cam_resized, cam_resized])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    return cam_image  # , gb, cam_gb


if __name__ == '__main__':
    args = get_args()

    # Define method mapping
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
    }
    cam_algorithm = methods[args.method]

    # Load pretrained model
    device = torch.device(args.device)
    model = models.resnet50(pretrained=True).to(device).eval()
    target_layers = [model.layer4]

    # Traverse datasets
    datasets = ["client_1", "client_2", "client_3", "client_4", "client_5", "client_6",
                "client_7", "client_8", "client_9", "client_10", "dataset/NEU_721"]
    splits = ["train", "val", "test"]
    for dataset in datasets:
        for split in splits:
            image_dir = os.path.join(args.root_dir, dataset, split, "images")
            if not os.path.exists(image_dir):
                print(f"⚠️ Missing folder {image_dir}, skipped")
                continue

            # Output folder mirrors the input structure
            out_dir = os.path.join(args.output_dir, dataset, split, "images")
            os.makedirs(out_dir, exist_ok=True)

            for img_name in os.listdir(image_dir):
                if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(image_dir, img_name)
                print(f"Processing {img_path}")

                cam_image = process_image(
                    model=model,
                    cam_algorithm=cam_algorithm,
                    target_layers=target_layers,
                    image_path=img_path,
                    device=device,
                    aug_smooth=args.aug_smooth,
                    eigen_smooth=args.eigen_smooth
                )

                # Save output with SAME filename
                cv2.imwrite(os.path.join(out_dir, img_name), cam_image)

    print("✅ All images processed and saved!")
