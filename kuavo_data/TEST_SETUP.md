# Rosbag转换性能测试 - 安装和使用指南

## 概述

本测试脚本用于评估rosbag转换的性能，包括内存占用、转换速度等指标。**可以在无GPU设备上运行**，只需要CPU和基本的Python环境。

## 系统要求

- Python 3.10+
- 至少 4GB 可用内存（推荐 8GB+）
- 无需GPU
- Linux/macOS/Windows（推荐Linux，ROS支持最好）

## 安装步骤

### 方法1：使用rospypi（推荐，适合无ROS系统环境）

```bash
# 1. 安装ROS Python包（从rospypi）
pip install --extra-index-url https://rospypi.github.io/simple/ \
    rospy==1.17.4 \
    rosbag==1.17.4 \
    roslz4==1.17.4 \
    cv-bridge==1.16.2 \
    sensor-msgs==1.13.2 \
    rospkg==1.6.0

# 2. 安装测试依赖
pip install -r requirements_test.txt
```

### 方法2：使用系统ROS Noetic（如果已安装ROS）

```bash
# 1. 确保ROS环境已source
source /opt/ros/noetic/setup.bash

# 2. 安装测试依赖（ROS包会使用系统包）
pip install -r requirements_test.txt
```

### 方法3：最小化安装（仅测试时间戳扫描，不处理图像）

如果只需要测试时间戳扫描和对齐逻辑，可以跳过图像处理相关依赖：

```bash
# 只安装核心依赖
pip install --extra-index-url https://rospypi.github.io/simple/ \
    rospy==1.17.4 \
    rosbag==1.17.4 \
    roslz4==1.17.4 \
    rospkg==1.6.0

pip install numpy psutil tqdm
pip install -e .
```

**注意**：如果跳过`cv-bridge`，测试脚本中涉及图像处理的部分可能会失败，但时间戳扫描测试仍可运行。

## 验证安装

运行以下命令验证环境：

```bash
python -c "import rosbag; import numpy; import psutil; print('✓ All dependencies installed')"
```

## 使用方法

### 基本用法

```bash
# 测试单个rosbag文件
python kuavo_data/test_rosbag_conversion.py /path/to/your/test.bag
```

### 常用选项

```bash
# 跳过原始方法测试（避免内存不足，推荐用于大文件）
python kuavo_data/test_rosbag_conversion.py /path/to/test.bag --skip-original

# 测试不同chunk大小
python kuavo_data/test_rosbag_conversion.py /path/to/test.bag --chunk-sizes 50 100 200

# 指定输出目录（默认使用临时目录）
python kuavo_data/test_rosbag_conversion.py /path/to/test.bag --output-dir ./test_output
```

### 完整示例

```bash
# 测试大rosbag文件（7-10GB），跳过原始方法，测试多个chunk大小
python kuavo_data/test_rosbag_conversion.py \
    /path/to/large_rosbag.bag \
    --skip-original \
    --chunk-sizes 50 100 200 \
    --output-dir ./test_results
```

## 测试输出说明

测试脚本会输出以下信息：

1. **文件信息**：rosbag文件大小、系统内存信息
2. **各方法测试结果**：
   - 时间戳扫描测试
   - 原始方法测试（可选）
   - 分块流式方法测试（多个chunk大小）
3. **性能对比表格**：包含以下指标
   - **Time(s)**: 总耗时（秒）
   - **Frames**: 处理的帧数
   - **FPS**: 帧处理速度（帧/秒）
   - **Peak Mem(MB)**: 内存峰值（MB）
   - **Mem Inc(MB)**: 内存增长（MB）
   - **Status**: 测试状态（✓ OK 或 ✗ FAIL）

## 故障排除

### 问题1：找不到rosbag模块

```
ModuleNotFoundError: No module named 'rosbag'
```

**解决方案**：
```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rosbag roslz4
```

### 问题2：找不到cv_bridge

```
ModuleNotFoundError: No module named 'cv_bridge'
```

**解决方案**：
```bash
pip install --extra-index-url https://rospypi.github.io/simple/ cv-bridge
# 或者使用opencv-python-headless
pip install opencv-python-headless
```

### 问题3：内存不足

如果测试大文件时出现内存不足，可以：

1. 使用`--skip-original`跳过原始方法测试
2. 减小chunk大小（如`--chunk-sizes 50`）
3. 关闭其他占用内存的程序

### 问题4：ROS环境变量未设置

如果使用系统ROS，确保已source环境：

```bash
source /opt/ros/noetic/setup.bash
```

## 性能测试建议

1. **小文件测试**（<1GB）：可以运行所有测试方法，包括原始方法
2. **中等文件**（1-5GB）：建议使用`--skip-original`，只测试分块方法
3. **大文件**（>5GB）：必须使用`--skip-original`，并选择合适的chunk大小

## 注意事项

- 测试脚本会创建临时目录，测试完成后会自动清理（除非指定了`--output-dir`）
- 内存监控需要`psutil`，如果未安装会显示警告但测试仍可继续
- 测试过程中会显示实时内存使用情况
- 如果rosbag文件损坏或格式不正确，测试会失败并显示错误信息

## 依赖说明

测试脚本的核心依赖：

- **numpy**: 数据处理
- **rosbag**: ROS bag文件读取
- **psutil**: 内存监控（可选但推荐）
- **tqdm**: 进度条显示（可选）

**不需要的依赖**：
- PyTorch（不需要GPU）
- CUDA相关库
- LeRobot训练相关库
- 其他深度学习框架

这使得测试可以在任何有Python环境的设备上运行。




