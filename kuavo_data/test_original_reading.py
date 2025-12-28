#!/usr/bin/env python3
"""
测试原本的数据读取方式（不使用测试框架）

直接调用 KuavoRosbagReader.process_rosbag() 方法，统计：
1. 内存占用（峰值、增量）
2. 读取时间（总时间、各阶段时间）
3. 读取的数据量（帧数、各话题的消息数）

使用方法：
    python test_original_reading.py /path/to/your/rosbag.bag
"""

import os
import sys
import gc
import time
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 内存监控（可选）
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory monitoring disabled.")
    print("Install with: pip install psutil")

# 初始化配置（必须）
def init_default_config():
    """初始化默认配置"""
    try:
        import kuavo_data.common.kuavo_dataset as kuavo
        from kuavo_data.common.config_dataset import Config, ResizeConfig
        
        try:
            from omegaconf import OmegaConf
            default_cfg = {
                "dataset": {
                    "only_arm": True,
                    "eef_type": "leju_claw",
                    "which_arm": "both",
                    "use_depth": False,
                    "depth_range": [0, 1000],
                    "dex_dof_needed": 1,
                    "train_hz": 10,
                    "main_timeline": "head_cam_h",
                    "main_timeline_fps": 30,
                    "sample_drop": 0,
                    "is_binary": False,
                    "delta_action": False,
                    "relative_start": False,
                    "resize": {"width": 640, "height": 480},
                    "task_description": "Test Task"
                }
            }
            cfg = OmegaConf.create(default_cfg)
            kuavo.init_parameters(cfg)
            return True
        except ImportError:
            # 备用方案：直接使用Config类
            resize_config = ResizeConfig(width=640, height=480)
            config = Config(
                only_arm=True, eef_type="leju_claw", which_arm="both",
                use_depth=False, depth_range=(0, 1000), dex_dof_needed=1,
                train_hz=10, main_timeline="head_cam_h", main_timeline_fps=30,
                sample_drop=0, is_binary=False, delta_action=False,
                relative_start=False, resize=resize_config, task_description="Test Task"
            )
            # 手动设置全局变量
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
        print(f"Error: Failed to initialize config: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_memory_mb():
    """获取当前内存使用(MB)"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def test_original_reading(bag_file: str):
    """
    测试原本的数据读取方式
    
    原本的方式：
    bag_reader = KuavoRosbagReader()
    bag_data = bag_reader.process_rosbag(bag_file)
    
    这个方法会：
    1. 加载rosbag文件
    2. 遍历所有话题，读取所有消息
    3. 处理消息（图像解码、数据提取等）
    4. 对齐数据（align_frame_data）
    5. 返回对齐后的数据字典
    """
    print("="*80)
    print("TESTING ORIGINAL DATA READING METHOD")
    print("="*80)
    print(f"Rosbag file: {bag_file}")
    
    # 检查文件
    if not os.path.exists(bag_file):
        print(f"Error: File not found: {bag_file}")
        return
    
    file_size_mb = os.path.getsize(bag_file) / 1024 / 1024
    print(f"File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
    
    if HAS_PSUTIL:
        mem_info = psutil.virtual_memory()
        print(f"System memory: {mem_info.total / 1024**3:.1f} GB")
        print(f"Available memory: {mem_info.available / 1024**3:.1f} GB")
    
    print("="*80)
    
    # 初始化配置
    if not init_default_config():
        print("Error: Failed to initialize configuration")
        return
    
    # 导入模块
    try:
        import kuavo_data.common.kuavo_dataset as kuavo
    except ImportError as e:
        print(f"Error: Cannot import kuavo_data module: {e}")
        return
    
    # 内存监控
    gc.collect()
    initial_memory = get_memory_mb()
    peak_memory = initial_memory
    
    print("\n[Step 1] Creating KuavoRosbagReader...")
    start_time = time.time()
    
    try:
        bag_reader = kuavo.KuavoRosbagReader()
        create_time = time.time() - start_time
        current_memory = get_memory_mb()
        peak_memory = max(peak_memory, current_memory)
        print(f"  ✓ Created in {create_time:.2f}s")
        print(f"  Memory: {current_memory:.2f} MB (increase: {current_memory - initial_memory:.2f} MB)")
        
        print("\n[Step 2] Loading and processing rosbag (this may take a while)...")
        print("  This step will:")
        print("    1. Load rosbag file")
        print("    2. Read all messages from all topics")
        print("    3. Process messages (decode images, extract data)")
        print("    4. Align frames by timestamp")
        
        process_start = time.time()
        bag_data = bag_reader.process_rosbag(bag_file)
        process_end = time.time()
        
        process_time = process_end - process_start
        current_memory = get_memory_mb()
        peak_memory = max(peak_memory, current_memory)
        
        print(f"\n  ✓ Processing completed in {process_time:.2f}s")
        print(f"  Memory: {current_memory:.2f} MB (increase: {current_memory - initial_memory:.2f} MB)")
        
        # 统计数据
        print("\n[Step 3] Data Statistics:")
        print("-" * 80)
        
        total_frames = 0
        main_timeline = None
        
        # 找到主时间线（最长的相机数据）
        camera_keys = [k for k in bag_data.keys() if k in kuavo.DEFAULT_CAMERA_NAMES]
        if camera_keys:
            main_timeline = max(camera_keys, key=lambda k: len(bag_data.get(k, [])))
            total_frames = len(bag_data.get(main_timeline, []))
            print(f"Main timeline: {main_timeline}")
            print(f"Total aligned frames: {total_frames}")
        
        # 统计各话题的数据量
        print("\nTopic data counts:")
        for key in sorted(bag_data.keys()):
            count = len(bag_data.get(key, []))
            if count > 0:
                print(f"  {key:30s}: {count:6d} frames")
                # 显示第一个数据的时间戳
                first_item = bag_data[key][0]
                if isinstance(first_item, dict) and 'timestamp' in first_item:
                    print(f"    First timestamp: {first_item['timestamp']:.3f}")
        
        # 计算FPS
        if process_time > 0 and total_frames > 0:
            fps = total_frames / process_time
            print(f"\nProcessing speed: {fps:.2f} frames/second")
        
        # 内存统计
        memory_increase = peak_memory - initial_memory
        print("\n" + "="*80)
        print("MEMORY STATISTICS")
        print("="*80)
        print(f"Initial memory:  {initial_memory:.2f} MB")
        print(f"Peak memory:    {peak_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB ({memory_increase/1024:.2f} GB)")
        
        # 时间统计
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("TIME STATISTICS")
        print("="*80)
        print(f"Reader creation: {create_time:.2f}s")
        print(f"Data processing:  {process_time:.2f}s")
        print(f"Total time:       {total_time:.2f}s")
        
        # 数据大小估算
        print("\n" + "="*80)
        print("DATA SIZE ESTIMATION")
        print("="*80)
        print(f"Rosbag file size:     {file_size_mb:.2f} MB")
        print(f"Memory after loading: {memory_increase:.2f} MB")
        print(f"Expansion ratio:      {memory_increase/file_size_mb:.2f}x")
        print("(Note: Images are decompressed in memory, so memory usage is higher than file size)")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # 清理
        del bag_data
        gc.collect()
        final_memory = get_memory_mb()
        print(f"\nAfter cleanup: {final_memory:.2f} MB (freed: {peak_memory - final_memory:.2f} MB)")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    parser = argparse.ArgumentParser(
        description="Test original rosbag reading method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_original_reading.py /path/to/your/rosbag.bag
        """
    )
    parser.add_argument(
        "bag_file",
        nargs="?",
        help="Path to rosbag file to test"
    )
    
    args = parser.parse_args()
    
    if not args.bag_file:
        parser.print_help()
        sys.exit(1)
    
    test_original_reading(args.bag_file)


if __name__ == "__main__":
    main()




