# 流式Rosbag处理优化方案

## 问题分析

### 当前实现的问题

1. **一次性加载**：`process_rosbag()` 一次性读取所有消息到内存
   ```python
   for _, msg, t in bag.read_messages(topics=topic):
       data[key].append(msg_data)  # 所有数据都在内存中
   ```

2. **后处理对齐**：需要先读取所有数据，才能确定主时间线和对齐
   ```python
   data_aligned = self.align_frame_data(data)  # 需要完整数据
   ```

3. **内存占用**：对于10G rosbag，所有图像数据（30GB+）都在内存中

### 帧对齐逻辑位置

**文件**：`kuavo_data/common/kuavo_dataset.py`
**函数**：`KuavoRosbagReader.align_frame_data()` (第477行)

**对齐逻辑**：
1. 确定主时间线（main_timeline）：选择消息数量最多的相机
2. 生成主时间戳序列：`main_img_timestamps = [t['timestamp'] for t in data[main_timeline]][SAMPLE_DROP:-SAMPLE_DROP][::jump]`
3. 对每个主时间戳，找到其他话题最接近的消息（使用`np.argmin(np.abs(time_array - stamp_sec))`）

## 流式处理方案

### 方案设计

**核心思想**：使用滑动窗口缓冲，边读取边对齐边处理

**关键挑战**：
1. 主时间线需要先确定（需要知道哪个相机消息最多）
2. 对齐需要知道所有时间戳范围
3. 需要处理时间戳间隙（999标志）

**解决方案**：
1. **两遍扫描**：
   - 第一遍：快速扫描确定主时间线和时间戳范围（不加载图像数据）
   - 第二遍：流式读取，按主时间线对齐

2. **滑动窗口对齐**：
   - 维护一个时间窗口的缓冲
   - 当窗口足够大时，对齐并处理
   - 使用`searchsorted`高效查找最近时间戳

### 实现步骤

1. 创建流式处理类 `StreamingRosbagProcessor`
2. 第一遍扫描：确定主时间线和时间戳序列
3. 第二遍扫描：流式读取和对齐
4. 集成到 `populate_dataset`








