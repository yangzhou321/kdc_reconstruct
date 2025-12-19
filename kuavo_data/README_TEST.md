# Rosbag转换性能测试 - 快速参考

## 快速开始

### 1. 安装依赖

**方法A：使用安装脚本（推荐）**
```bash
./install_test_deps.sh
```

**方法B：手动安装**
```bash
# 安装ROS包
pip install --extra-index-url https://rospypi.github.io/simple/ \
    rospy rosbag roslz4 cv-bridge sensor-msgs rospkg

# 安装测试依赖
pip install -r requirements_test.txt
```

### 2. 运行测试

```bash
# 基本测试
python kuavo_data/test_rosbag_conversion.py /path/to/test.bag

# 测试大文件（推荐）
python kuavo_data/test_rosbag_conversion.py /path/to/large.bag --skip-original
```

## 测试维度

测试脚本会测量以下性能指标：

1. **内存占用**
   - 峰值内存使用（MB）
   - 内存增长量（MB）
   - 内存使用曲线

2. **转换速度**
   - 总耗时（秒）
   - 各阶段耗时（扫描、处理、对齐、写入）
   - 帧处理速度（FPS）

3. **方法对比**
   - 原始方法 vs 分块流式方法
   - 不同chunk大小的性能对比

## 输出示例

```
================================================================================
ROSBAG CONVERSION PERFORMANCE TEST
================================================================================
Test file: /path/to/test.bag
File size: 7500.00 MB (7.32 GB)
Chunk sizes to test: [100, 50]
Skip original method: True
System memory: 32.0 GB
Available memory: 16.5 GB
================================================================================

============================================================
Testing: Timestamp Scan Only
============================================================
✓ Scan completed in 45.23s
  Main timeline: kuavo_arm_traj
  Aligned frames: 12500
  Memory increase: 125.50 MB

============================================================
Testing: Chunked Streaming Method (chunk_size=100)
============================================================
Phase 1: Scanning timestamps...
  Found 12500 frames, took 45.12s
  Memory after scan: 125.30 MB
Phase 2: Chunked processing...
  Chunk 1 done, memory: 450.20 MB
  Chunk 2 done, memory: 455.10 MB
  ...
✓ Completed: 12500 frames in 180.45s
  Peak memory: 480.30 MB
  Memory increase: 354.80 MB

================================================================================
PERFORMANCE COMPARISON
================================================================================
Method                              | Time(s)   | Frames  | FPS     | Peak Mem(MB)   | Mem Inc(MB)   | Status
------------------------------------|-----------|--------|---------|----------------|---------------|--------
Timestamp Scan Only                 | 45.23     | 12500  | 276.4   | 125.5          | 125.5         | ✓ OK
Chunked Streaming (chunk=100)       | 180.45    | 12500  | 69.3    | 480.3          | 354.8         | ✓ OK
Chunked Streaming (chunk=50)        | 195.20    | 12500  | 64.0    | 350.2          | 224.7         | ✓ OK
================================================================================
```

## 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `bag_file` | rosbag文件路径（必需） | - |
| `--chunk-sizes` | 要测试的chunk大小列表 | `100 50` |
| `--skip-original` | 跳过原始方法测试 | `False` |
| `--output-dir` | 输出目录 | 临时目录 |

## 常见问题

**Q: 可以在无GPU设备上运行吗？**  
A: 可以！测试脚本不需要GPU，只需要CPU和基本Python环境。

**Q: 需要安装完整的ROS系统吗？**  
A: 不需要。只需要ROS Python包，可以通过rospypi安装。

**Q: 测试大文件时内存不足怎么办？**  
A: 使用`--skip-original`选项，并减小chunk大小（如`--chunk-sizes 50`）。

**Q: 如何只测试时间戳扫描？**  
A: 运行测试后查看"Timestamp Scan Only"的结果，这是最快的测试。

## 更多信息

- 详细安装说明：`kuavo_data/TEST_SETUP.md`
- 依赖文件：`requirements_test.txt`
- 测试脚本：`kuavo_data/test_rosbag_conversion.py`




