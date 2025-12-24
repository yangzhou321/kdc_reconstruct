#!/bin/bash
# Rosbag转换性能测试 - 快速安装脚本
# 此脚本用于在无GPU设备上安装测试所需的依赖

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Rosbag转换性能测试 - 依赖安装脚本"
echo "=========================================="
echo ""

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip3，请先安装pip"
    exit 1
fi

echo ""
echo "步骤1: 安装ROS Python包（从rospypi）..."
pip3 install --extra-index-url https://rospypi.github.io/simple/ \
    rospy==1.17.4 \
    rosbag==1.17.4 \
    roslz4==1.17.4 \
    cv-bridge==1.16.2 \
    sensor-msgs==1.13.2 \
    rospkg==1.6.0

if [ $? -ne 0 ]; then
    echo "警告: ROS包安装可能失败，请检查网络连接或使用系统ROS"
    echo "如果系统已安装ROS Noetic，可以跳过此步骤"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "步骤2: 安装测试依赖..."
pip3 install -r requirements_test.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo ""
echo "步骤3: 验证安装..."
python3 -c "
import sys
errors = []
try:
    import rosbag
    print('✓ rosbag')
except ImportError as e:
    errors.append('rosbag')
    print('✗ rosbag:', e)

try:
    import numpy
    print('✓ numpy')
except ImportError as e:
    errors.append('numpy')
    print('✗ numpy:', e)

try:
    import psutil
    print('✓ psutil')
except ImportError:
    print('⚠ psutil (可选)')

try:
    from kuavo_data.common import kuavo_dataset
    print('✓ kuavo_data')
except ImportError as e:
    errors.append('kuavo_data')
    print('✗ kuavo_data:', e)

if errors:
    print(f'\n错误: 以下模块未正确安装: {errors}')
    sys.exit(1)
else:
    print('\n✓ 所有依赖已正确安装!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "安装完成！"
    echo "=========================================="
    echo ""
    echo "现在可以运行测试:"
    echo "  python3 kuavo_data/test_rosbag_conversion.py /path/to/your/test.bag"
    echo ""
    echo "详细使用说明请参考: kuavo_data/TEST_SETUP.md"
else
    echo ""
    echo "=========================================="
    echo "安装验证失败"
    echo "=========================================="
    echo "请检查错误信息并手动安装缺失的依赖"
    exit 1
fi




