#!/usr/bin/env python3
"""
Rosbag转换性能测试脚本

测试维度：
1. 内存占用峰值
2. 转换时间
3. 各阶段耗时分析
4. 对比原始方法 vs 分块流式方法

使用方法：
    python test_rosbag_conversion.py /path/to/your/test.bag

环境要求：
    - Python 3.10+
    - 无需GPU，可在CPU设备上运行
    - 需要安装ROS相关包（rospy, rosbag等）
    
安装依赖：
    pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag roslz4
    pip install -r requirements_test.txt
    
详细安装说明请参考：kuavo_data/TEST_SETUP.md
"""

import os
import sys
import gc
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import traceback

# 添加项目根目录到Python路径（支持直接运行脚本，无需pip install）
# 这样即使没有运行 pip install -e . 也能正常工作
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 检查numpy
try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

# 内存监控（可选）
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory monitoring disabled.")
    print("Install with: pip install psutil")


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    method_name: str
    total_time_sec: float = 0.0
    scan_time_sec: float = 0.0
    process_time_sec: float = 0.0
    align_time_sec: float = 0.0
    write_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    initial_memory_mb: float = 0.0
    memory_increase_mb: float = 0.0
    num_frames: int = 0
    frames_per_sec: float = 0.0
    success: bool = False
    error_msg: str = ""
    memory_samples: List[float] = field(default_factory=list)


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self.samples = []
        self.peak = 0.0
        self.initial = self.get_current_memory()
    
    def get_current_memory(self) -> float:
        """获取当前内存使用(MB)"""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def sample(self):
        """采样当前内存"""
        mem = self.get_current_memory()
        self.samples.append(mem)
        if mem > self.peak:
            self.peak = mem
        return mem
    
    def get_peak(self) -> float:
        return self.peak
    
    def get_increase(self) -> float:
        return self.peak - self.initial


def test_original_method(bag_file: str, output_dir: str) -> PerformanceMetrics:
    """
    测试原始方法：一次性加载全部数据
    """
    metrics = PerformanceMetrics(method_name="Original (Full Load)")
    monitor = MemoryMonitor()
    metrics.initial_memory_mb = monitor.initial
    
    print("\n" + "="*60)
    print("Testing: Original Method (Full Load)")
    print("="*60)
    
    try:
        # 使用绝对导入（项目根目录已添加到sys.path）
        import kuavo_data.common.kuavo_dataset as kuavo
        
        gc.collect()
        monitor.sample()
        
        start_time = time.time()
        
        # 创建reader
        bag_reader = kuavo.KuavoRosbagReader()
        
        # 一次性加载和处理（原始方法）
        print("Loading and processing rosbag...")
        process_start = time.time()
        bag_data = bag_reader.process_rosbag(bag_file)
        process_end = time.time()
        
        metrics.process_time_sec = process_end - process_start
        monitor.sample()
        
        # 统计帧数
        if 'observation.state' in bag_data:
            metrics.num_frames = len(bag_data['observation.state'])
        
        end_time = time.time()
        metrics.total_time_sec = end_time - start_time
        metrics.peak_memory_mb = monitor.get_peak()
        metrics.memory_increase_mb = monitor.get_increase()
        metrics.memory_samples = monitor.samples
        
        if metrics.total_time_sec > 0:
            metrics.frames_per_sec = metrics.num_frames / metrics.total_time_sec
        
        metrics.success = True
        
        print(f"✓ Completed: {metrics.num_frames} frames in {metrics.total_time_sec:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.2f} MB")
        print(f"  Memory increase: {metrics.memory_increase_mb:.2f} MB")
        
    except Exception as e:
        metrics.success = False
        metrics.error_msg = str(e)
        print(f"✗ Failed: {e}")
        traceback.print_exc()
    
    # 清理
    gc.collect()
    
    return metrics


def test_chunked_method(bag_file: str, output_dir: str, chunk_size: int = 100) -> PerformanceMetrics:
    """
    测试分块流式方法
    """
    metrics = PerformanceMetrics(method_name=f"Chunked Streaming (chunk={chunk_size})")
    monitor = MemoryMonitor()
    metrics.initial_memory_mb = monitor.initial
    
    print("\n" + "="*60)
    print(f"Testing: Chunked Streaming Method (chunk_size={chunk_size})")
    print("="*60)
    
    try:
        # 使用绝对导入（项目根目录已添加到sys.path）
        import kuavo_data.common.kuavo_dataset as kuavo
        from kuavo_data.common.kuavo_dataset_chunked import ChunkedRosbagProcessor
        
        gc.collect()
        monitor.sample()
        
        start_time = time.time()
        
        # 创建reader和processor
        bag_reader = kuavo.KuavoRosbagReader()
        processor = ChunkedRosbagProcessor(
            msg_processer=bag_reader._msg_processer,
            topic_process_map=bag_reader._topic_process_map,
            camera_names=kuavo.DEFAULT_CAMERA_NAMES,
            train_hz=kuavo.TRAIN_HZ,
            main_timeline_fps=kuavo.MAIN_TIMELINE_FPS,
            sample_drop=kuavo.SAMPLE_DROP,
            only_half_up_body=kuavo.ONLY_HALF_UP_BODY
        )
        
        # 第一遍：扫描时间戳
        print("Phase 1: Scanning timestamps...")
        scan_start = time.time()
        main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)
        scan_end = time.time()
        metrics.scan_time_sec = scan_end - scan_start
        monitor.sample()
        print(f"  Found {len(main_timestamps)} frames, took {metrics.scan_time_sec:.2f}s")
        print(f"  Memory after scan: {monitor.get_current_memory():.2f} MB")
        
        # 第二遍：分块处理
        print("Phase 2: Chunked processing...")
        frame_count = [0]
        chunk_count = [0]
        
        def on_frame(aligned_frame: dict, frame_idx: int):
            frame_count[0] += 1
            if frame_count[0] % 100 == 0:
                monitor.sample()
        
        def on_chunk_done():
            chunk_count[0] += 1
            mem = monitor.sample()
            print(f"  Chunk {chunk_count[0]} done, memory: {mem:.2f} MB")
            gc.collect()
        
        process_start = time.time()
        processor.process_in_chunks(
            bag_file=bag_file,
            main_timestamps=main_timestamps,
            all_timestamps=all_timestamps,
            frame_callback=on_frame,
            chunk_size=chunk_size,
            save_callback=on_chunk_done
        )
        process_end = time.time()
        metrics.process_time_sec = process_end - process_start
        
        metrics.num_frames = frame_count[0]
        
        end_time = time.time()
        metrics.total_time_sec = end_time - start_time
        metrics.peak_memory_mb = monitor.get_peak()
        metrics.memory_increase_mb = monitor.get_increase()
        metrics.memory_samples = monitor.samples
        
        if metrics.total_time_sec > 0:
            metrics.frames_per_sec = metrics.num_frames / metrics.total_time_sec
        
        metrics.success = True
        
        print(f"✓ Completed: {metrics.num_frames} frames in {metrics.total_time_sec:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.2f} MB")
        print(f"  Memory increase: {metrics.memory_increase_mb:.2f} MB")
        
    except Exception as e:
        metrics.success = False
        metrics.error_msg = str(e)
        print(f"✗ Failed: {e}")
        traceback.print_exc()
    
    gc.collect()
    
    return metrics


def test_timestamp_scan_only(bag_file: str) -> PerformanceMetrics:
    """
    测试仅扫描时间戳（验证第一遍扫描的效率）
    """
    metrics = PerformanceMetrics(method_name="Timestamp Scan Only")
    monitor = MemoryMonitor()
    metrics.initial_memory_mb = monitor.initial
    
    print("\n" + "="*60)
    print("Testing: Timestamp Scan Only")
    print("="*60)
    
    try:
        # 使用绝对导入（项目根目录已添加到sys.path）
        import kuavo_data.common.kuavo_dataset as kuavo
        from kuavo_data.common.kuavo_dataset_chunked import ChunkedRosbagProcessor
        
        gc.collect()
        monitor.sample()
        
        start_time = time.time()
        
        bag_reader = kuavo.KuavoRosbagReader()
        processor = ChunkedRosbagProcessor(
            msg_processer=bag_reader._msg_processer,
            topic_process_map=bag_reader._topic_process_map,
            camera_names=kuavo.DEFAULT_CAMERA_NAMES,
            train_hz=kuavo.TRAIN_HZ,
            main_timeline_fps=kuavo.MAIN_TIMELINE_FPS,
            sample_drop=kuavo.SAMPLE_DROP,
            only_half_up_body=kuavo.ONLY_HALF_UP_BODY
        )
        
        main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)
        
        end_time = time.time()
        monitor.sample()
        
        metrics.total_time_sec = end_time - start_time
        metrics.scan_time_sec = metrics.total_time_sec
        metrics.num_frames = len(main_timestamps)
        metrics.peak_memory_mb = monitor.get_peak()
        metrics.memory_increase_mb = monitor.get_increase()
        metrics.success = True
        
        # 统计各话题的消息数
        print(f"✓ Scan completed in {metrics.total_time_sec:.2f}s")
        print(f"  Main timeline: {main_timeline}")
        print(f"  Aligned frames: {len(main_timestamps)}")
        print(f"  Memory increase: {metrics.memory_increase_mb:.2f} MB")
        print("  Topic message counts:")
        for topic, timestamps in sorted(all_timestamps.items()):
            print(f"    {topic}: {len(timestamps)} messages")
        
    except Exception as e:
        metrics.success = False
        metrics.error_msg = str(e)
        print(f"✗ Failed: {e}")
        traceback.print_exc()
    
    gc.collect()
    
    return metrics


def print_comparison_table(results: List[PerformanceMetrics]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    # 表头
    headers = ["Method", "Time(s)", "Frames", "FPS", "Peak Mem(MB)", "Mem Inc(MB)", "Status"]
    widths = [35, 10, 8, 8, 14, 14, 8]
    
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    
    # 数据行
    for r in results:
        status = "✓ OK" if r.success else "✗ FAIL"
        row = [
            r.method_name[:35],
            f"{r.total_time_sec:.2f}",
            str(r.num_frames),
            f"{r.frames_per_sec:.1f}",
            f"{r.peak_memory_mb:.1f}",
            f"{r.memory_increase_mb:.1f}",
            status
        ]
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    
    print("="*80)
    
    # 计算改进比例
    if len(results) >= 2:
        original = results[0]
        chunked = results[1]
        if original.success and chunked.success and original.memory_increase_mb > 0:
            mem_reduction = (1 - chunked.memory_increase_mb / original.memory_increase_mb) * 100
            print(f"\nMemory Reduction: {mem_reduction:.1f}%")
        if original.success and chunked.success and original.total_time_sec > 0:
            time_ratio = chunked.total_time_sec / original.total_time_sec
            print(f"Time Ratio: {time_ratio:.2f}x")


def get_file_size_mb(file_path: str) -> float:
    """获取文件大小(MB)"""
    return os.path.getsize(file_path) / 1024 / 1024


def init_default_config():
    """
    初始化默认配置，用于测试环境
    
    创建一个默认配置对象，包含测试所需的基本参数
    支持两种方式：
    1. 使用OmegaConf（如果已安装）
    2. 直接使用Config类（备用方案）
    """
    try:
        import kuavo_data.common.kuavo_dataset as kuavo
        from kuavo_data.common.config_dataset import Config, ResizeConfig
        
        # 方法1：尝试使用OmegaConf（如果可用）
        try:
            from omegaconf import OmegaConf
            
            # 创建默认配置字典
            default_cfg = {
                "dataset": {
                    "only_arm": True,
                    "eef_type": "leju_claw",  # 默认使用leju_claw
                    "which_arm": "both",  # 默认使用双臂
                    "use_depth": False,  # 测试时默认不使用深度
                    "depth_range": [0, 1000],
                    "dex_dof_needed": 1,
                    "train_hz": 10,
                    "main_timeline": "head_cam_h",
                    "main_timeline_fps": 30,
                    "sample_drop": 0,
                    "is_binary": False,
                    "delta_action": False,
                    "relative_start": False,
                    "resize": {
                        "width": 640,
                        "height": 480
                    },
                    "task_description": "Test Task"
                }
            }
            
            # 转换为OmegaConf对象
            cfg = OmegaConf.create(default_cfg)
            kuavo.init_parameters(cfg)
            return True
            
        except ImportError:
            # 方法2：直接使用Config类（不依赖OmegaConf）
            resize_config = ResizeConfig(width=640, height=480)
            config = Config(
                only_arm=True,
                eef_type="leju_claw",
                which_arm="both",
                use_depth=False,
                depth_range=(0, 1000),
                dex_dof_needed=1,
                train_hz=10,
                main_timeline="head_cam_h",
                main_timeline_fps=30,
                sample_drop=0,
                is_binary=False,
                delta_action=False,
                relative_start=False,
                resize=resize_config,
                task_description="Test Task"
            )
            
            # 手动初始化全局变量（模拟init_parameters的逻辑）
            kuavo.DEFAULT_CAMERA_NAMES = config.default_camera_names
            kuavo.TRAIN_HZ = config.train_hz
            kuavo.MAIN_TIMELINE_FPS = config.main_timeline_fps
            kuavo.SAMPLE_DROP = config.sample_drop
            kuavo.CONTROL_HAND_SIDE = config.which_arm
            kuavo.SLICE_ROBOT = config.slice_robot
            kuavo.SLICE_DEX = config.dex_slice
            kuavo.SLICE_CLAW = config.claw_slice
            kuavo.IS_BINARY = config.is_binary
            kuavo.DELTA_ACTION = config.delta_action
            kuavo.RELATIVE_START = config.relative_start
            kuavo.RESIZE_W = config.resize.width
            kuavo.RESIZE_H = config.resize.height
            kuavo.ONLY_HALF_UP_BODY = config.only_half_up_body
            kuavo.USE_LEJU_CLAW = config.use_leju_claw
            kuavo.USE_QIANGNAO = config.use_qiangnao
            kuavo.USE_DEPTH = config.use_depth
            kuavo.DEPTH_RANGE = config.depth_range
            kuavo.TASK_DESCRIPTION = config.task_description
            
            return True
            
    except Exception as e:
        print(f"Warning: Failed to initialize default config: {e}")
        import traceback
        traceback.print_exc()
        print("Some features may not work correctly.")
        return False


def check_dependencies():
    """
    检查必需的依赖是否已安装
    
    注意：项目根目录已自动添加到sys.path，无需pip install -e .
    """
    missing = []
    
    # 检查ROS相关依赖
    try:
        import rosbag
    except ImportError:
        missing.append("rosbag")
    
    try:
        import rospy
    except ImportError:
        missing.append("rospy")
    
    # 检查项目内部模块
    # 注意：_project_root已在脚本开头添加到sys.path
    try:
        import kuavo_data.common.kuavo_dataset
    except ImportError as e:
        print(f"Error: Cannot import kuavo_data module: {e}")
        print(f"Project root: {_project_root}")
        print(f"kuavo_data directory: {_project_root / 'kuavo_data'}")
        print(f"kuavo_data exists: {(_project_root / 'kuavo_data').exists()}")
        print("\nPossible solutions:")
        print("  1. Ensure you are running from the project root directory")
        print("  2. Or install the project: pip install -e .")
        return False
    
    # 初始化默认配置
    if not init_default_config():
        print("Warning: Failed to initialize default configuration.")
        print("The test may fail if configuration is required.")
    
    if missing:
        print("Error: Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag roslz4")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rosbag转换性能测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本测试
  python test_rosbag_conversion.py /path/to/test.bag
  
  # 跳过原始方法（推荐用于大文件）
  python test_rosbag_conversion.py /path/to/test.bag --skip-original
  
  # 测试多个chunk大小
  python test_rosbag_conversion.py /path/to/test.bag --chunk-sizes 50 100 200

详细安装说明请参考: kuavo_data/TEST_SETUP.md
        """
    )
    parser.add_argument("bag_file", nargs="?", default="/home/jsl/Downloads/A02-A02-H-K-02-06_12-leju_claw-20251011124317-v1.bag", help="要测试的rosbag文件路径")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[100, 50],
                        help="要测试的chunk大小列表 (默认: 100 50)")
    parser.add_argument("--skip-original", action="store_true",
                        help="跳过原始方法测试（避免内存不足，推荐用于大文件）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认使用临时目录）")
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    bag_file = args.bag_file
    
    if not os.path.exists(bag_file):
        print(f"Error: File not found: {bag_file}")
        sys.exit(1)
    
    file_size_mb = get_file_size_mb(bag_file)
    
    print("="*80)
    print("ROSBAG CONVERSION PERFORMANCE TEST")
    print("="*80)
    print(f"Test file: {bag_file}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
    print(f"Chunk sizes to test: {args.chunk_sizes}")
    print(f"Skip original method: {args.skip_original}")
    if HAS_PSUTIL:
        print(f"System memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print("="*80)
    
    # 创建临时输出目录
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="rosbag_test_")
    
    print(f"Output directory: {output_dir}")
    
    results = []
    
    # 1. 测试时间戳扫描
    results.append(test_timestamp_scan_only(bag_file))
    
    # 2. 测试原始方法（可选）
    if not args.skip_original:
        results.append(test_original_method(bag_file, output_dir))
    
    # 3. 测试不同chunk大小的分块方法
    for chunk_size in args.chunk_sizes:
        results.append(test_chunked_method(bag_file, output_dir, chunk_size))
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 清理临时目录
    if not args.output_dir and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"\nCleaned up temporary directory: {output_dir}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()



