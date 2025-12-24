#!/usr/bin/env python3
"""
内存优化测试脚本

用于测试超大rosbag数据转换的内存优化效果。

使用方法:
    python kuavo_data/test_memory_optimization.py --rosbag_path /path/to/large_rosbag.bag --chunk_size 1000
"""

import argparse
import psutil
import os
import time
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from kuavo_data.CvtRosbag2Lerobot import populate_dataset, create_empty_dataset, port_kuavo_rosbag
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import kuavo_data.common.kuavo_dataset as kuavo
from omegaconf import DictConfig, OmegaConf


def get_memory_usage():
    """获取当前进程的内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def monitor_memory(interval=1.0):
    """监控内存使用情况"""
    max_memory = 0
    start_memory = get_memory_usage()
    
    import threading
    import time
    
    def monitor_loop():
        nonlocal max_memory
        while True:
            current = get_memory_usage()
            max_memory = max(max_memory, current)
            time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    return lambda: (get_memory_usage(), max_memory, start_memory)


def test_chunk_size(rosbag_path: Path, chunk_size: int, output_dir: Path):
    """测试不同chunk_size的内存使用"""
    print(f"\n{'='*60}")
    print(f"Testing with chunk_size={chunk_size}")
    print(f"{'='*60}")
    
    # 初始化配置（简化版，实际应从配置文件加载）
    cfg = OmegaConf.create({
        'rosbag': {
            'chunk_size': chunk_size,
            'rosbag_dir': str(rosbag_path.parent),
            'lerobot_dir': str(output_dir),
            'num_used': None
        },
        'dataset': {
            'only_arm': True,
            'eef_type': 'rq2f85',
            'which_arm': 'both',
            'use_depth': False,
            'depth_range': [0, 1000],
            'dex_dof_needed': 1,
            'is_binary': False,
            'delta_action': False,
            'relative_start': False,
            'train_hz': 10,
            'main_timeline': 'head_cam_h',
            'main_timeline_fps': 30,
            'sample_drop': 10,
            'task_description': 'Test Task',
            'resize': {
                'width': 640,
                'height': 480
            }
        }
    })
    
    kuavo.init_parameters(cfg)
    
    # 开始监控
    get_stats = monitor_memory(interval=0.5)
    start_time = time.time()
    
    try:
        # 创建数据集
        repo_id = f'lerobot/test_chunk_{chunk_size}'
        dataset = create_empty_dataset(
            repo_id=repo_id,
            robot_type="kuavo4pro",
            mode="image",
            has_effort=False,
            has_velocity=False,
            root=str(output_dir),
        )
        
        # 处理rosbag
        bag_files = [rosbag_path]
        dataset = populate_dataset(
            dataset=dataset,
            bag_files=bag_files,
            task="Test Task",
            episodes=None,
            chunk_size=chunk_size,
        )
        
        end_time = time.time()
        current_memory, max_memory, start_memory = get_stats()
        
        print(f"\nResults:")
        print(f"  Processing time: {end_time - start_time:.2f} seconds")
        print(f"  Start memory: {start_memory:.2f} MB")
        print(f"  End memory: {current_memory:.2f} MB")
        print(f"  Peak memory: {max_memory:.2f} MB")
        print(f"  Memory increase: {max_memory - start_memory:.2f} MB")
        
        return {
            'chunk_size': chunk_size,
            'time': end_time - start_time,
            'start_memory': start_memory,
            'end_memory': current_memory,
            'peak_memory': max_memory,
            'memory_increase': max_memory - start_memory
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Test memory optimization for large rosbag conversion')
    parser.add_argument('--rosbag_path', type=str, required=True, help='Path to rosbag file')
    parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[0, 500, 1000, 2000],
                        help='Chunk sizes to test (0 means no chunking)')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='Output directory for test datasets')
    
    args = parser.parse_args()
    
    rosbag_path = Path(args.rosbag_path)
    if not rosbag_path.exists():
        print(f"Error: Rosbag file not found: {rosbag_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing memory optimization with rosbag: {rosbag_path}")
    print(f"Rosbag size: {rosbag_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Chunk sizes to test: {args.chunk_sizes}")
    
    results = []
    for chunk_size in args.chunk_sizes:
        result = test_chunk_size(rosbag_path, chunk_size, output_dir)
        if result:
            results.append(result)
        time.sleep(2)  # 等待内存释放
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Chunk Size':<15} {'Time (s)':<12} {'Peak Memory (MB)':<18} {'Memory Increase (MB)':<20}")
    print("-" * 60)
    for r in results:
        print(f"{r['chunk_size']:<15} {r['time']:<12.2f} {r['peak_memory']:<18.2f} {r['memory_increase']:<20.2f}")


if __name__ == "__main__":
    main()









