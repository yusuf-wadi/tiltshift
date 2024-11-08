"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
from tqdm import tqdm
from sbsconvert import SBSConverter


first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)
    
def process_video(input_path, output_path, device, model, model_type, transform, net_w, net_h, optimize, side):
    video_names = glob.glob(os.path.join(input_path, "*"))
    print("Processing video:", video_names[0])
    print("Output path:", output_path)
    
    video_in = cv2.VideoCapture(video_names[0])
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define output video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    side_by_side_out = cv2.VideoWriter(output_path.replace("_depth", "_dsbs"), fourcc, fps, (width*2, height)) if side else None
    depth_only_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = video_in.read()
            if not ret:
                break
            
            original_image_rgb = np.flip(frame, 2)
            image = transform({"image": original_image_rgb/255})["image"]
            
            prediction = process(device, model, model_type, image, (net_w, net_h),
                                 original_image_rgb.shape[1::-1], optimize, True)
            
            # Create side-by-side view
            if side:
                original_image_bgr = np.flip(original_image_rgb, 2)
                side_by_side = create_side_by_side(original_image_bgr, prediction, grayscale=False)
                # Write frames to output videos
                side_by_side_out.write(side_by_side)
            
            # Normalize depth map for visualization
            depth_colored = cv2.applyColorMap(np.uint8((prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255), cv2.COLORMAP_INFERNO)
            
            
            depth_only_out.write(depth_colored)
    
    # Release resources
    video_in.release()
    if side: side_by_side_out.release()
    depth_only_out.release()
    
    print("Video processing complete. Output saved to:", output_path)
    
def _to_sbs(input_path, output_path, depth_path, shift_scale=0.1):
    
    converter = SBSConverter()
    converter.process_video(input_path, depth_path, output_path, shift_scale)


def run(input_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False, video=False, to_sbs=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    # get input
    if input_path is None:
        print("No input path specified")
        exit()

    print("Start processing")

    video_paths = glob.glob(os.path.join(input_path, "*"))
    print("Video paths: ", video_paths)
    video_paths = [path.replace("\\", "/") for path in video_paths]
    print("  video: ", video_paths)
    ext = video_paths[0].split('.')[-1]
    depth_path = f"output/{video_paths[0].split('/')[-1].split('.')[0]}_output+{model_type}" + "_depth." + ext
    process_video(input_path, depth_path, device, model, model_type, transform, net_w, net_h, optimize, side)
    if to_sbs:
        output_path = f"output/{video_paths[0].split('/')[-1].split('.')[0]}_output+{model_type}_sbs" + "." + ext
        _to_sbs(video_paths[0], output_path, depth_path)


    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )
    
    parser.add_argument('--video', dest='video', action='store_true', help='Use video as input')
    parser.add_argument('--to_sbs', dest='to_sbs', action='store_true', help='Convert video to side-by-side stereo video')

    args = parser.parse_args()


    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale, args.video, args.to_sbs)
