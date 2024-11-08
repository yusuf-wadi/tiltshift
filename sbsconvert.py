import cv2
import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip

class SBSConverter:
    def __init__(self, device='cuda'):
        self.device = device
        
    def generate_disparity_map(self, depth_frame):
        depth_tensor = torch.from_numpy(depth_frame).float().to(self.device)
        depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
        normalized_depth = (depth_tensor - depth_min) / (depth_max - depth_min)
        
        return normalized_depth
    
    def reduce_temporal_flicker(self, disparity, temporal_window=5):
        disparity = disparity.squeeze()
        height, width = disparity.shape
        
        for y in range(height):
            for x in range(width):
                disparity[y, x] = torch.median(disparity[y, x-temporal_window//2:x+temporal_window//2+1])
        
        return disparity
        
        
    def generate_right_view(self, left_frame, disparity, shift_scale=0.1):
        left_tensor = torch.from_numpy(left_frame).float().to(self.device)
        height, width = left_tensor.shape[0], left_tensor.shape[1]
        
        x = torch.linspace(-1, 1, width, device=self.device)
        y = torch.linspace(-1, 1, height, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        disparity = disparity.to(self.device)
        shift = disparity * shift_scale
        
        grid_x = grid_x - shift.squeeze()
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0)
        
        right_view = F.grid_sample(
            left_tensor.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border'
        )
        
        return right_view.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def process_video(self, input_path, depth_path, output_path, shift_scale=0.1):
        # Load original video with audio
        original_video = VideoFileClip(input_path)
        audio = original_video.audio
        
        cap = cv2.VideoCapture(input_path)
        depth_cap = cv2.VideoCapture(depth_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create temporary video without audio
        temp_output = 'temp_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width*2, height))

        with torch.cuda.device(self.device):
            while True:
                ret, frame = cap.read()
                depth_ret, depth_frame = depth_cap.read()
                
                if not ret or not depth_ret:
                    break
                    
                if len(depth_frame.shape) == 3:
                    depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                
                disparity = self.generate_disparity_map(depth_frame)
                disparity = self.reduce_temporal_flicker(disparity)
                right_view = self.generate_right_view(frame, disparity, shift_scale)
                sbs_frame = np.hstack((frame, right_view.astype(np.uint8)))
                out.write(sbs_frame)
        
        cap.release()
        depth_cap.release()
        out.release()

        # Combine video with original audio
        video_without_audio = VideoFileClip(temp_output)
        final_video = video_without_audio.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Cleanup
        original_video.close()
        video_without_audio.close()
        import os
        if os.path.exists(temp_output):
            os.remove(temp_output)


import argparse

if __name__ == '__main__':
    
    converter = SBSConverter()
    
    parser = argparse.ArgumentParser(description='Converts a video to a side-by-side stereo video')
    parser.add_argument('input', type=str, help='Path to input video')
    parser.add_argument('depth', type=str, help='Path to depth video')
    parser.add_argument('output', type=str, help='Path to output video')
    parser.add_argument('--shift-scale', type=float, default=0.1, help='Scale of disparity shift')
    args = parser.parse_args()
    
    converter.process_video(args.input, args.depth, args.output, args.shift_scale)
    
