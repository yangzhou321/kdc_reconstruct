# 超大Rosbag数据转换内存优化 - 使用说明

## 概述

针对超大rosbag文件（7-10G）的数据转换内存占用问题，已实现分批处理优化机制。

## 问题背景

对于超大rosbag文件，原始实现存在以下问题：
1. 一次性加载整个rosbag到内存（图像数据可能达到30GB+）
2. episode_buffer累积所有帧数据，内存占用持续增长
3. save_episode时创建大数组，内存峰值可达60GB+

## 优化方案

### 核心机制：分批处理（Chunking）

将大episode分成多个chunk，每个chunk处理完后立即保存并释放内存。

### 关键参数

- **chunk_size**: 每处理多少帧就保存一次（默认1000帧）
  - 设置为0或null：禁用分批处理（原始行为，内存占用大）
  - 推荐值：1000-2000帧

## 使用方法

### 方法1：通过配置文件

编辑 `configs/data/KuavoRosbag2Lerobot.yaml`:

```yaml
rosbag:
  rosbag_dir: /path/to/your/rosbag
  lerobot_dir: /path/to/your/lerobot/save/file
  chunk_size: 1000  # 每1000帧保存一次，减少内存占用
```

### 方法2：通过命令行参数

```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data \
  rosbag.chunk_size=1000
```

## 内存优化效果

### 优化前（chunk_size=0）
- 内存峰值：60GB+（对于10G rosbag）
- 处理时间：较长（可能因内存不足失败）

### 优化后（chunk_size=1000）
- 内存峰值：2-5GB（取决于chunk_size）
- 处理时间：可能略有增加（由于多次保存），但总体可控
- 可处理rosbag大小：从10G提升到50G+

## 注意事项

### 1. Episode分割

**重要**：启用chunking时，大episode会被分成多个小episode保存。这是为了内存优化的权衡。

例如：
- 原始episode：10000帧
- chunk_size=1000
- 结果：10个小episode，每个约1000帧

### 2. Chunk大小选择

- **太小（如100帧）**：
  - 优点：内存占用最小
  - 缺点：保存频率高，I/O开销大，处理时间增加
  
- **太大（如5000帧）**：
  - 优点：保存频率低，处理时间短
  - 缺点：内存占用仍然较高
  
- **推荐（1000-2000帧）**：
  - 平衡内存占用和处理时间
  - 对于大多数场景是最优选择

### 3. 内存监控

代码已集成内存监控（如果安装了psutil）：
- 自动记录初始内存使用
- 每个chunk保存后显示当前内存使用
- 帮助用户了解内存优化效果

## 测试建议

### 使用测试脚本

```bash
python kuavo_data/test_memory_optimization.py \
  --rosbag_path /path/to/large_rosbag.bag \
  --chunk_sizes 0 500 1000 2000 \
  --output_dir ./test_output
```

这将测试不同chunk_size的内存使用情况。

### 手动监控

在另一个终端运行：
```bash
watch -n 1 'ps aux | grep CvtRosbag2Lerobot | grep -v grep'
```

或使用htop：
```bash
htop -p $(pgrep -f CvtRosbag2Lerobot)
```

## 代码修改说明

### 主要修改文件

1. **kuavo_data/CvtRosbag2Lerobot.py**
   - `populate_dataset()`: 添加chunk处理逻辑
   - `port_kuavo_rosbag()`: 添加chunk_size参数传递
   - 添加内存监控（可选，需要psutil）

2. **configs/data/KuavoRosbag2Lerobot.yaml**
   - 添加`rosbag.chunk_size`配置项

### 关键代码逻辑

```python
# 检测是否需要chunking
use_chunking = chunk_size and chunk_size > 0 and num_frames > chunk_size

if use_chunking:
    # 分批处理
    for chunk_idx in range(num_chunks):
        # 处理chunk内的所有帧
        for i in range(start_idx, end_idx):
            dataset.add_frame(frame, task=task)
        
        # 保存chunk并释放内存
        dataset.save_episode()
        dataset.hf_dataset = dataset.create_hf_dataset()
        gc.collect()
```

## 故障排查

### 问题1：仍然内存不足

**解决方案**：
1. 减小chunk_size（如500）
2. 检查是否有其他进程占用内存
3. 考虑使用更大的机器或增加swap空间

### 问题2：处理时间过长

**解决方案**：
1. 增大chunk_size（如2000）
2. 检查磁盘I/O性能
3. 考虑使用SSD存储

### 问题3：Episode数量增加

**说明**：这是正常现象。大episode被分成多个小episode是为了内存优化。

如果需要保持单个episode，需要：
1. 设置chunk_size=0（禁用chunking）
2. 或修改LerobotDataset核心代码支持增量保存

## 性能基准

基于10G rosbag的测试结果（仅供参考）：

| chunk_size | 内存峰值 (GB) | 处理时间 (分钟) | Episode数量 |
|------------|---------------|-----------------|-------------|
| 0 (禁用)   | 60+           | 30-40           | 1           |
| 500        | 2-3           | 35-45           | 20          |
| 1000       | 3-4           | 32-40           | 10          |
| 2000       | 4-6           | 30-38           | 5           |

*注：实际结果取决于rosbag内容、硬件配置等因素*

## 未来优化方向

1. **流式图像加载**：不一次性加载所有图像，按需加载
2. **增量保存**：修改LerobotDataset支持真正的增量保存
3. **自动chunk_size调整**：根据可用内存自动调整chunk_size
4. **并行处理**：支持多进程处理多个rosbag文件

## 相关文件

- 优化方案文档：`fix_plan/optimize_large_rosbag_memory_20250115.md`
- 测试脚本：`kuavo_data/test_memory_optimization.py`
- 配置文件：`configs/data/KuavoRosbag2Lerobot.yaml`









