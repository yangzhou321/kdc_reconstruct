# 超大Rosbag数据转换内存优化方案

## 问题分析

### 当前内存占用瓶颈

1. **一次性加载整个rosbag** (`load_raw_episode_data`)
   - 所有图像数据（可能几GB）一次性加载到内存
   - 所有状态和动作数据也在内存中
   - 对于7-10G的rosbag，内存占用可能达到15-20G

2. **episode_buffer累积** (`add_frame`)
   - 每帧数据都append到`episode_buffer`的列表中
   - 对于超大episode（数万帧），buffer会占用大量内存
   - 非图像数据（state, action等）的numpy数组一直保存在内存中

3. **save_episode时的内存峰值** (`save_episode`)
   - `np.stack(episode_buffer[key])`会将所有帧堆叠成一个大数组
   - 创建HuggingFace Dataset时会复制数据
   - `ep_dataset.with_format("arrow")[:]`会将整个数据集加载到内存

### 内存占用估算

对于一个10G的rosbag，假设：
- 10000帧数据
- 3个相机，每个图像640x480x3 (约1MB/帧)
- 状态和动作数据约1KB/帧

内存占用：
- 图像数据：10000 * 3 * 1MB = 30GB（原始加载）
- episode_buffer：30GB + 状态动作数据
- save_episode时：30GB * 2（stack + Dataset创建）= 60GB峰值

## 优化方案

### 方案1：分批处理 + 定期保存（已实现）

**核心思想**：将大episode分成多个chunk，每个chunk处理完后立即保存，然后清空buffer。

**实现要点**：
1. 在`populate_dataset`中，每处理N帧（如1000帧）就调用一次`save_episode`
2. 使用`save_episode(episode_data=chunk_buffer)`保存chunk数据
3. 保存后清空chunk buffer，继续处理下一批
4. 每次保存后调用`gc.collect()`释放内存

**优点**：
- 内存占用可控（最多N帧的数据在内存中）
- 实现相对简单
- 不需要修改LerobotDataset核心代码

**缺点**：
- 一个episode会被分成多个小episode保存（这是为了内存优化的权衡）
- 需要处理chunk之间的索引连续性（LerobotDataset自动处理）

### 方案2：流式处理 + 增量保存（未来优化）

**核心思想**：逐帧处理，每帧立即写入临时文件，最后合并。

**实现要点**：
1. 修改`add_frame`，支持增量写入
2. 使用临时parquet文件存储chunk数据
3. 最后合并所有chunk文件

**优点**：
- 内存占用最小
- 可以处理任意大小的episode

**缺点**：
- 需要修改LerobotDataset核心代码
- 实现复杂度较高

### 方案3：优化数据加载（未来优化）

**核心思想**：不要一次性加载所有图像到内存，而是按需加载。

**实现要点**：
1. 修改`load_raw_images_per_camera`，返回图像路径而不是图像数据
2. 在`populate_dataset`中，按需加载图像
3. 或者使用生成器，逐帧加载

**优点**：
- 大幅减少初始内存占用
- 可以处理更大的rosbag

**缺点**：
- 需要修改数据加载逻辑
- 可能影响处理速度（I/O开销）

## 已实现：方案1

### 实现细节

1. **修改`populate_dataset`函数**：
   - 添加`chunk_size`参数（默认1000帧）
   - 检测大episode（帧数 > chunk_size）
   - 自动启用分批处理
   - 每个chunk处理完后立即保存并释放内存

2. **修改`port_kuavo_rosbag`函数**：
   - 添加`chunk_size`参数
   - 从配置文件读取`chunk_size`

3. **配置文件支持**：
   - 在`configs/data/KuavoRosbag2Lerobot.yaml`中添加`rosbag.chunk_size`参数

### 使用方法

在配置文件中设置：
```yaml
rosbag:
  chunk_size: 1000  # 每1000帧保存一次，减少内存占用
```

或者通过命令行：
```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data \
  rosbag.chunk_size=1000
```

### 关键代码修改点

1. `CvtRosbag2Lerobot.py`:
   - `populate_dataset`: 添加chunk处理逻辑
   - `port_kuavo_rosbag`: 添加chunk_size参数传递

2. `configs/data/KuavoRosbag2Lerobot.yaml`:
   - 添加`rosbag.chunk_size`配置项

## 预期效果

- **内存占用**：从60GB峰值降低到约2-5GB（取决于chunk_size）
- **处理时间**：可能略有增加（由于多次保存），但总体可控
- **可处理rosbag大小**：从当前10G提升到50G+

## 注意事项

1. **Episode分割**：启用chunking时，大episode会被分成多个小episode。这是为了内存优化的权衡。
2. **Chunk大小选择**：
   - 太小（如100）：保存频率高，I/O开销大
   - 太大（如5000）：内存占用仍然较高
   - 推荐：1000-2000帧
3. **内存监控**：建议在处理大rosbag时监控内存使用情况

## 测试建议

1. 使用提供的测试rosbag（7-10G）
2. 设置不同的chunk_size值（500, 1000, 2000）
3. 监控内存使用情况
4. 比较处理时间和内存占用

## 未来优化方向

1. 实现方案3：优化数据加载，避免一次性加载所有图像
2. 实现方案2：流式处理，进一步减少内存占用
3. 添加内存监控和自动调整chunk_size的机制
4. 优化`save_episode`中的`np.stack`操作，使用更高效的内存管理
