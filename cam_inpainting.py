import os
import cv2
import numpy as np

def extract_most_red(cam_image_path, original_image_path, mask_output_dir, remain_output_dir, top_percent=0.2):
    """
    Extract the most red area from a CAM image and save mask (white=most red) and remaining original image.

    Args:
        cam_image_path: Path to CAM image (from GradCAM, BGR)
        original_image_path: Path to original image
        mask_output_dir: Folder to save binary mask
        remain_output_dir: Folder to save remaining image
        top_percent: Top percentile to consider as "most red"
    """
    # Load CAM image
    cam_img = cv2.imread(cam_image_path)
    if cam_img is None:
        print(f"⚠️ Cannot read CAM image {cam_image_path}")
        return

    # Load original image
    orig_img = cv2.imread(original_image_path)
    if orig_img is None:
        print(f"⚠️ Cannot read original image {original_image_path}")
        return

    # Resize CAM to match original if needed
    if cam_img.shape != orig_img.shape:
        cam_img = cv2.resize(cam_img, (orig_img.shape[1], orig_img.shape[0]))

    # Extract red channel intensity
    red_channel = cam_img[:, :, 2].astype(np.float32) / 255.0

    # Threshold top X% of red values
    thresh_val = np.percentile(red_channel, 100 * (1 - top_percent))
    mask = (red_channel >= thresh_val).astype(np.uint8) * 255  # white = most red

    # Save mask
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_name = os.path.basename(cam_image_path)
    cv2.imwrite(os.path.join(mask_output_dir, mask_name), mask)

    # Remaining image = original image with masked region removed (set to black)
    remain_img = orig_img.copy()
    remain_img[mask == 255] = 0  # remove most red area

    os.makedirs(remain_output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(remain_output_dir, mask_name), remain_img)


if __name__ == "__main__":
    root_cam_dir = "./attention"        # CAM output folder
    root_orig_dir = ""               # Root dataset folder
    mask_root = "./attention_mask"
    remain_root = "./attention_remain"

    datasets = ["client_1", "client_2", "client_3", "client_4", "client_5", "client_6",
                "client_7", "client_8", "client_9", "client_10", "dataset/NEU_721"]
    splits = ["train", "val", "test"]

    for dataset in datasets:
        for split in splits:
            cam_dir = os.path.join(root_cam_dir, dataset, split, "images")
            orig_dir = os.path.join(root_orig_dir, dataset, split, "images")
            if not os.path.exists(cam_dir) or not os.path.exists(orig_dir):
                continue

            mask_dir = os.path.join(mask_root, dataset, split, "images")
            remain_dir = os.path.join(remain_root, dataset, split, "images")

            for img_name in os.listdir(cam_dir):
                if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                cam_path = os.path.join(cam_dir, img_name)
                orig_path = os.path.join(orig_dir, img_name)
                extract_most_red(cam_path, orig_path, mask_dir, remain_dir, top_percent=0.2)

    print("✅ Masks (white=most red) and remaining images saved!")
