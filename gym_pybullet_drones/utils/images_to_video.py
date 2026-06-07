#!/usr/bin/env python3
"""
将保存的 PNG 图像序列转换为视频文件

使用方法:
    python images_to_video.py <image_folder> [--fps 30] [--output output.mp4]

示例:
    python images_to_video.py results/recording_06.06.2026_12.34.56
    python images_to_video.py results/recording_06.06.2026_12.34.56 --fps 60 --output flight_video.mp4
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import re

def natural_sort_key(filename):
    """用于自然排序数字文件名"""
    numbers = re.compile(r'(\d+)')
    return [int(text) if text.isdigit() else text.lower() for text in numbers.split(filename)]

def images_to_video(image_folder, fps=24, output_path=None, resolution=None):
    """
    将图像序列转换为视频
    
    参数:
        image_folder: 包含图像的文件夹路径
        fps: 视频帧率（默认24）
        output_path: 输出视频文件路径
        resolution: 视频分辨率，例如 (640, 480)，None 则自动检测
    """
    image_folder = Path(image_folder)
    
    if not image_folder.exists():
        print(f"错误: 文件夹 {image_folder} 不存在")
        return False
    
    # 查找所有PNG文件
    image_files = sorted([f for f in image_folder.glob("frame_*.png")], 
                        key=lambda x: natural_sort_key(x.name))
    
    if not image_files:
        print(f"错误: 在 {image_folder} 中未找到 frame_*.png 文件")
        return False
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 读取第一张图像获取分辨率
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"错误: 无法读取图像 {image_files[0]}")
        return False
    
    if resolution is None:
        resolution = (first_image.shape[1], first_image.shape[0])  # (width, height)
    
    print(f"图像分辨率: {resolution[0]}x{resolution[1]}")
    print(f"视频帧率: {fps} fps")
    print(f"视频时长: {len(image_files)/fps:.2f} 秒")
    
    # 设置输出路径
    if output_path is None:
        output_path = image_folder / "output.mp4"
    else:
        output_path = Path(output_path)
    
    # 创建视频写入器
    # 使用 H.264 编码器 (mp4v) 以获得更好的兼容性
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    if not out.isOpened():
        print("错误: 无法创建视频写入器。请检查是否安装了 ffmpeg")
        return False
    
    print(f"生成视频: {output_path}")
    
    # 逐帧写入视频
    for i, image_file in enumerate(image_files):
        frame = cv2.imread(str(image_file))
        
        if frame is None:
            print(f"警告: 无法读取图像 {image_file}，跳过")
            continue
        
        # 如果需要调整分辨率
        if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
            frame = cv2.resize(frame, resolution)
        
        out.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(image_files)} 帧")
    
    out.release()
    print(f"✓ 视频生成完成: {output_path}")
    print(f"✓ 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 PNG 图像序列转换为 MP4 视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置 (24 fps)
  python images_to_video.py results/recording_06.06.2026_12.34.56
  
  # 指定帧率为 60 fps
  python images_to_video.py results/recording_06.06.2026_12.34.56 --fps 60
  
  # 指定帧率和输出文件名
  python images_to_video.py results/recording_06.06.2026_12.34.56 --fps 60 --output my_video.mp4
  
  # 自定义分辨率
  python images_to_video.py results/recording_06.06.2026_12.34.56 --fps 60 --resolution 1280 720
        """
    )
    
    parser.add_argument("image_folder", help="包含 frame_*.png 的文件夹路径")
    parser.add_argument("--fps", type=int, default=24, 
                       help="输出视频的帧率 (默认: 24)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出视频文件路径 (默认: image_folder/output.mp4)")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       default=None, help="视频分辨率 (默认: 自动检测)")
    
    args = parser.parse_args()
    
    resolution = tuple(args.resolution) if args.resolution else None
    success = images_to_video(args.image_folder, fps=args.fps, 
                             output_path=args.output, resolution=resolution)
    
    if not success:
        exit(1)
