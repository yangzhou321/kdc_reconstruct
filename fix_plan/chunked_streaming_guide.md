# 分块流式Rosbag处理指南

## 核心优化思路

参考 Diffusion Policy 的按需读取方式，将原来的"一次性全部加载"改为"分块读取+即时对齐+即时写入"。

### 原始方式的问题
```
一次性加载整个rosbag (7-10GB) 
    → 全部解析到内存 (60GB+) 
    → 对齐 
    → 写入dataset
```
**问题**: 内存峰值巨大，7GB的rosbag需要60GB+内存

### 新方式（分块流式）
```
第一遍扫描: 只读取时间戳 (几MB)
第二遍扫描: 
    → 按时间窗口分块读取 (chunk_size帧)
    → 即时对齐
    → 即时写入dataset
    → 释放内存
    → 下一个chunk...
```
**优势**: 内存占用可控，与chunk_size成正比

## 使用方法

### 方法1: 使用分块流式转换脚本（推荐）

```bash
cd kuavo_data_challenge

# 使用分块流式版本（默认chunk_size=100）
python kuavo_data/CvtRosbag2Lerobot_chunked.py \
    rosbag.rosbag_dir=/path/to/your/rosbag \
    rosbag.lerobot_dir=/path/to/output \
    rosbag.chunk_size=100

# 对于超大rosbag，使用更小的chunk_size
python kuavo_data/CvtRosbag2Lerobot_chunked.py \
    rosbag.rosbag_dir=/path/to/large_rosbag \
    rosbag.lerobot_dir=/path/to/output \
    rosbag.chunk_size=50
```

### 方法2: 在代码中使用分块处理API

```python
from kuavo_data.common.kuavo_dataset import KuavoRosbagReader

reader = KuavoRosbagReader()

# 定义帧回调函数
def on_frame(aligned_frame: dict, frame_idx: int):
    """处理每帧对齐后的数据"""
    state = aligned_frame['observation.state']['data']
    action = aligned_frame['action']['data']
    # ... 添加到dataset
    dataset.add_frame(frame, task=task)

# 定义chunk完成回调
def on_chunk_done():
    """每个chunk处理完后保存并释放内存"""
    dataset.save_episode()
    dataset.hf_dataset = dataset.create_hf_dataset()
    gc.collect()

# 分块流式处理
reader.process_rosbag_chunked(
    bag_file="large.bag",
    frame_callback=on_frame,
    chunk_size=100,
    save_callback=on_chunk_done
)
```

## 新增文件说明

| 文件 | 说明 |
|------|------|
| `kuavo_data/common/kuavo_dataset_chunked.py` | 分块流式处理核心实现 |
| `kuavo_data/CvtRosbag2Lerobot_chunked.py` | 分块流式转换入口脚本 |

## 新增API

### `KuavoRosbagReader.process_rosbag_chunked()`

```python
def process_rosbag_chunked(
    self,
    bag_file: str,
    frame_callback: Callable[[dict, int], None],
    chunk_size: int = 100,
    save_callback: Optional[Callable[[], None]] = None
) -> int:
    """
    分块流式处理rosbag
    
    Args:
        bag_file: rosbag文件路径
        frame_callback: 处理每帧的回调函数 (aligned_frame, frame_idx) -> None
        chunk_size: 每个chunk包含的帧数
        save_callback: 每个chunk处理完后的回调
    
    Returns:
        处理的总帧数
    """
```

## 内存占用对比

| 方法 | 7GB rosbag | 10GB rosbag |
|------|------------|-------------|
| 原始方式 | ~60GB | ~100GB |
| 分块流式 (chunk=100) | ~2-3GB | ~2-3GB |
| 分块流式 (chunk=50) | ~1-2GB | ~1-2GB |

## 注意事项

1. **Episode分割**: 由于LeRobotDataset的设计，每次`save_episode()`会创建新episode。使用分块处理时，大rosbag会被分成多个小episode。

2. **chunk_size选择**:
   - 默认100帧，适合大多数场景
   - 内存紧张时减小到50
   - 内存充足时可增大到200-500

3. **第一遍扫描**: 只读取时间戳，非常快且内存占用极小（只有时间戳列表）

4. **第二遍扫描**: 按时间窗口读取消息，避免一次性加载全部数据






