"""
分块流式处理rosbag模块

核心思路（参考Diffusion Policy的按需读取方式和内存管理策略）：
1. 第一遍扫描：只读取时间戳，确定主时间线（不加载图像数据，内存占用极小）
2. 第二遍扫描：按时间窗口分块读取，边读取边对齐边写入dataset

内存管理策略（参照内存管理实现详解.md）：
- 显式内存释放：使用del立即释放不需要的变量
- 流式处理：逐帧处理，不一次性加载
- 及时释放中间变量：在循环中及时释放
- 线程池限制：避免过度订阅

与原始方法的区别：
- 原始：一次性加载所有数据到内存 → 对齐 → 写入dataset（内存峰值巨大）
- 新方法：分块读取 → 即时对齐 → 即时写入 → 释放内存（内存占用可控）
"""

import numpy as np
import rosbag
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Tuple, Generator
import logging
import bisect
import gc

logger = logging.getLogger(__name__)

# 尝试导入线程池限制工具（如果可用）
try:
    from threadpoolctl import threadpool_limits
    THREADPOOL_AVAILABLE = True
except ImportError:
    THREADPOOL_AVAILABLE = False
    logger.warning("threadpoolctl not available, thread pool limiting disabled")

# 尝试导入OpenCV线程限制（如果可用）
try:
    import cv2
    cv2.setNumThreads(1)  # 限制OpenCV线程数
except ImportError:
    pass


class ChunkedRosbagProcessor:
    """
    分块流式处理rosbag，实现边读取边对齐边处理
    
    工作流程：
    1. scan_timestamps(): 第一遍扫描，只读取时间戳（内存占用极小）
    2. process_chunks(): 第二遍扫描，按时间窗口分块处理
    """
    
    def __init__(self, msg_processer, topic_process_map: dict, 
                 camera_names: list, train_hz: int, main_timeline_fps: int, 
                 sample_drop: int, only_half_up_body: bool):
        self._msg_processer = msg_processer
        self._topic_process_map = topic_process_map
        self.camera_names = camera_names
        self.train_hz = train_hz
        self.main_timeline_fps = main_timeline_fps
        self.sample_drop = sample_drop
        self.only_half_up_body = only_half_up_body
    
    def scan_timestamps_only(self, bag_file: str) -> Tuple[str, List[float], Dict[str, List[float]]]:
        """
        第一遍扫描：只读取时间戳，不加载数据
        
        内存占用：只有时间戳列表（几MB），不包含图像数据
        
        Returns:
            main_timeline: 主时间线话题key
            main_timestamps: 对齐后的主时间戳序列（降采样后）
            all_timestamps: 每个话题的原始时间戳列表
        """
        bag = self._load_bag(bag_file)
        
        # 统计每个话题的时间戳（不加载消息内容）
        all_timestamps = defaultdict(list)
        topic_to_key = {v["topic"]: k for k, v in self._topic_process_map.items()}
        
        logger.info(f"[Phase 1] Scanning timestamps from {bag_file}...")
        
        # 一次遍历，收集所有话题的时间戳
        all_topics = [v["topic"] for v in self._topic_process_map.values()]
        for topic, msg, t in bag.read_messages(topics=all_topics):
            key = topic_to_key.get(topic)
            if key:
                all_timestamps[key].append(t.to_sec())
        
        bag.close()
        
        # 确定主时间线：消息最多的相机
        camera_counts = {k: len(all_timestamps.get(k, [])) for k in self.camera_names}
        if not any(camera_counts.values()):
            raise ValueError("No camera data found in rosbag")
        
        main_timeline = max(camera_counts, key=lambda k: camera_counts[k])
        logger.info(f"Main timeline: {main_timeline} ({camera_counts[main_timeline]} frames)")
        
        # 生成对齐后的主时间戳序列
        jump = self.main_timeline_fps // self.train_hz
        raw_timestamps = all_timestamps[main_timeline]

        if len(raw_timestamps) < 2 * self.sample_drop + 1:
            raise ValueError(f"Not enough frames: {len(raw_timestamps)}")

        # 丢弃首尾帧，降采样
        # 注意：当 sample_drop 为 0 时，不能使用 [self.sample_drop:-self.sample_drop]，
        # 因为 [-0] 等价于 [0]，会导致切片结果为空。
        if self.sample_drop > 0:
            main_timestamps = raw_timestamps[self.sample_drop:-self.sample_drop][::jump]
        else:
            main_timestamps = raw_timestamps[::jump]
        
        logger.info(f"Generated {len(main_timestamps)} aligned timestamps "
                   f"(from {len(raw_timestamps)} raw frames, "
                   f"dropped {self.sample_drop} frames at each end, jump={jump})")
        
        return main_timeline, main_timestamps, dict(all_timestamps)
    
    def process_in_chunks(
        self,
        bag_file: str,
        main_timestamps: List[float],
        all_timestamps: Dict[str, List[float]],
        frame_callback: Callable[[dict, int], None],
        chunk_size: int = 100,
        save_callback: Optional[Callable[[int], None]] = None
    ) -> int:
        """
        第二遍扫描：按时间窗口分块处理
        
        策略：
        1. 将main_timestamps分成多个chunk
        2. 对于每个chunk，只读取该时间范围内的消息
        3. 对齐后立即调用frame_callback
        4. 每个chunk处理完后调用save_callback释放内存
        
        Args:
            bag_file: rosbag文件路径
            main_timestamps: 对齐后的主时间戳序列
            all_timestamps: 每个话题的原始时间戳列表（用于快速查找）
            frame_callback: 处理每帧的回调函数 (aligned_frame, frame_idx) -> None
            chunk_size: 每个chunk包含的帧数
            save_callback: 每个chunk处理完后的回调（用于保存和释放内存）
        
        Returns:
            处理的总帧数
        """
        bag = self._load_bag(bag_file)
        
        # 为每个话题构建时间戳索引（用于快速查找最近帧）
        timestamp_arrays = {k: np.array(v) for k, v in all_timestamps.items()}
        
        # 预计算每个主时间戳对应的各话题索引（避免重复查找）
        alignment_indices = self._precompute_alignment_indices(
            main_timestamps, timestamp_arrays
        )
        
        # 检测kuavo_arm_traj的时间戳间隙
        arm_traj_gaps = self._detect_arm_traj_gaps(all_timestamps)
        
        num_chunks = (len(main_timestamps) + chunk_size - 1) // chunk_size
        total_frames = 0
        
        logger.info(f"[Phase 2] Processing {len(main_timestamps)} frames in {num_chunks} chunks...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(main_timestamps))
            chunk_timestamps = main_timestamps[start_idx:end_idx]
            
            if not chunk_timestamps:
                continue
            
            # 确定该chunk的时间范围（扩展一点以确保对齐数据可用）
            time_margin = 1.0 / self.train_hz  # 一帧的时间
            chunk_start_time = chunk_timestamps[0] - time_margin
            chunk_end_time = chunk_timestamps[-1] + time_margin
            
            logger.debug(f"Chunk {chunk_idx+1}/{num_chunks}: "
                        f"frames {start_idx}-{end_idx-1}, "
                        f"time range [{chunk_start_time:.3f}, {chunk_end_time:.3f}]")
            
            # 读取该时间范围内的消息
            chunk_data = self._read_chunk_data(bag, chunk_start_time, chunk_end_time)
            
            # 对齐并处理每帧（边读取边对齐边处理，处理完立即删除）
            # 参照内存管理实现详解.md：显式释放策略
            for local_idx, (global_idx, main_stamp) in enumerate(
                zip(range(start_idx, end_idx), chunk_timestamps)
            ):
                # 限制线程池大小（避免过度订阅，参照文档策略）
                if THREADPOOL_AVAILABLE:
                    with threadpool_limits(limits=1, user_api='blas'):
                        aligned_frame = self._align_single_frame(
                            main_stamp=main_stamp,
                            global_idx=global_idx,
                            chunk_data=chunk_data,
                            timestamp_arrays=timestamp_arrays,
                            alignment_indices=alignment_indices,
                            arm_traj_gaps=arm_traj_gaps
                        )
                else:
                    aligned_frame = self._align_single_frame(
                        main_stamp=main_stamp,
                        global_idx=global_idx,
                        chunk_data=chunk_data,
                        timestamp_arrays=timestamp_arrays,
                        alignment_indices=alignment_indices,
                        arm_traj_gaps=arm_traj_gaps
                    )
                
                # 立即调用回调处理（不存储，假设回调会立即处理数据）
                frame_callback(aligned_frame, global_idx)
                total_frames += 1
                
                # 立即清理对齐后的帧数据（释放引用）
                # 参照文档：在数据加载时释放（del data[key]）
                # 先清理帧中的大对象（图像数组），确保彻底释放
                if isinstance(aligned_frame, dict):
                    for k in list(aligned_frame.keys()):
                        v = aligned_frame[k]
                        if isinstance(v, dict):
                            # 如果是字典，检查是否有numpy数组
                            if "data" in v and isinstance(v["data"], np.ndarray):
                                # 显式删除numpy数组（参照文档：del data[key]）
                                arr = v["data"]
                                del v["data"]
                                # 尝试释放numpy数组的底层内存
                                if hasattr(arr, 'base') and arr.base is not None:
                                    del arr.base
                                del arr
                            # 删除整个字典值
                            del aligned_frame[k]
                        elif isinstance(v, np.ndarray):
                            # 如果是直接的numpy数组，直接删除
                            arr = v
                            del aligned_frame[k]
                            if hasattr(arr, 'base') and arr.base is not None:
                                del arr.base
                            del arr
                        else:
                            # 其他类型，直接删除
                            del aligned_frame[k]
                # 删除整个aligned_frame字典
                del aligned_frame
                
                # 注意：不再调用gc.collect()，完全依赖del机制
                # Python的引用计数会在del时立即删除对象（如果没有循环引用）
                # 如果有循环引用，会在下次自动GC时回收，不需要手动调用
            
            # 彻底释放chunk数据（包含所有图像和消息数据）
            # 参照文档：在训练循环中释放（del batch, del obs_dict等）
            # 策略：先删除所有numpy数组，再删除字典结构
            chunk_keys = list(chunk_data.keys())
            for key in chunk_keys:
                if key not in chunk_data:
                    continue
                ts_dict = chunk_data[key]
                ts_keys = list(ts_dict.keys())
                for timestamp in ts_keys:
                    if timestamp not in ts_dict:
                        continue
                    msg_data = ts_dict[timestamp]
                    
                    # 删除图像数组等大对象
                    if isinstance(msg_data, dict):
                        # 如果是字典，先删除其中的numpy数组
                        if "data" in msg_data and isinstance(msg_data["data"], np.ndarray):
                            arr = msg_data["data"]
                            del msg_data["data"]
                            # 尝试释放numpy数组的底层内存
                            if hasattr(arr, 'base') and arr.base is not None:
                                del arr.base
                            del arr
                        # 删除整个消息数据字典
                        del msg_data
                    elif isinstance(msg_data, np.ndarray):
                        # 如果是直接的numpy数组，直接删除
                        arr = msg_data
                        if hasattr(arr, 'base') and arr.base is not None:
                            del arr.base
                        del arr
                    
                    # 从字典中删除该时间戳的条目
                    del ts_dict[timestamp]
                
                # 删除该key的整个字典
                del chunk_data[key]
            
            # 清空并删除整个chunk_data字典
            chunk_data.clear()
            del chunk_data
            
            # 清理临时变量（避免保留引用）
            del chunk_keys
            del ts_dict
            del ts_keys
            
            # 注意：完全依赖del机制删除对象，不再调用gc.collect()
            # 
            # Python的引用计数机制：
            # - 当对象的引用计数降为0时，立即被删除（不需要gc.collect()）
            # - 如果对象有循环引用，会在Python的自动GC时回收（通常很快）
            # - del语句会立即减少引用计数，对象会被立即删除
            #
            # 优势：
            # - 更快的删除速度（不需要等待GC扫描）
            # - 更可预测的行为（del后立即删除）
            # - 减少GC开销（让Python自动管理GC）
            #
            # 验证删除是否生效：
            # - 查看内存变化（+X MB from start）
            # - 查看是否有"freed X MB"的提示
            # - 如果内存稳定或下降，说明删除生效
            
            # 调用保存回调（不再传递collected数量，因为不再调用gc.collect()）
            if save_callback:
                # 传递0表示完全依赖del机制（不依赖gc.collect()）
                save_callback(0)
                logger.info(f"Chunk {chunk_idx+1}/{num_chunks} processed and saved. "
                           f"Frames: {start_idx}-{end_idx-1} (using del mechanism, no gc.collect())")
        
        bag.close()
        logger.info(f"Total frames processed: {total_frames}")
        return total_frames
    
    def _precompute_alignment_indices(
        self, 
        main_timestamps: List[float], 
        timestamp_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, List[int]]:
        """
        预计算每个主时间戳对应的各话题索引
        使用二分查找，比每帧都查找快很多
        """
        alignment_indices = {}
        
        for key, ts_array in timestamp_arrays.items():
            if len(ts_array) == 0:
                alignment_indices[key] = []
                continue
            
            indices = []
            for stamp in main_timestamps:
                # 二分查找最近的时间戳
                idx = bisect.bisect_left(ts_array, stamp)
                if idx == 0:
                    closest_idx = 0
                elif idx == len(ts_array):
                    closest_idx = len(ts_array) - 1
                else:
                    # 选择更接近的
                    if abs(ts_array[idx] - stamp) < abs(ts_array[idx-1] - stamp):
                        closest_idx = idx
                    else:
                        closest_idx = idx - 1
                indices.append(closest_idx)
            
            alignment_indices[key] = indices
        
        return alignment_indices
    
    def _detect_arm_traj_gaps(self, all_timestamps: Dict[str, List[float]]) -> List[Tuple[float, float]]:
        """检测kuavo_arm_traj的时间戳间隙"""
        gaps = []
        if not self.only_half_up_body and "action.kuavo_arm_traj" in all_timestamps:
            timestamps = all_timestamps["action.kuavo_arm_traj"]
            if len(timestamps) > 1:
                gap_threshold = 0.15 * 10 / self.train_hz
                for i in range(1, len(timestamps)):
                    if timestamps[i] - timestamps[i-1] > gap_threshold:
                        gaps.append((timestamps[i-1], timestamps[i]))
        if gaps:
            logger.info(f"Detected {len(gaps)} gaps in action.kuavo_arm_traj")
        return gaps
    
    def _read_chunk_data(self, bag: rosbag.Bag, start_time: float, end_time: float) -> Dict[str, Dict[float, dict]]:
        """
        读取指定时间范围内的消息数据（边读取边处理，不保留原始消息对象）
        
        参照文档：流式处理策略，逐帧处理，不一次性加载
        
        Returns:
            {topic_key: {timestamp: msg_data}}
        """
        import rospy
        chunk_data = defaultdict(dict)
        topic_to_key = {v["topic"]: k for k, v in self._topic_process_map.items()}
        
        # 限制线程池大小（避免过度订阅）
        if THREADPOOL_AVAILABLE:
            with threadpool_limits(limits=1, user_api='blas'):
                # 使用时间范围过滤
                try:
                    start_ros_time = rospy.Time.from_sec(start_time)
                    end_ros_time = rospy.Time.from_sec(end_time)
                    
                    for topic, msg, t in bag.read_messages(
                        topics=list(topic_to_key.keys()),
                        start_time=start_ros_time,
                        end_time=end_ros_time
                    ):
                        key = topic_to_key.get(topic)
                        if key:
                            msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                            # 立即处理消息，转换为数据格式（释放原始msg对象）
                            # 参照文档：立即处理，不保留原始对象
                            msg_data = msg_process_fn(msg)
                            msg_data["timestamp"] = t.to_sec()
                            chunk_data[key][t.to_sec()] = msg_data
                            # 立即删除原始消息对象（参照文档：del msg）
                            del msg
                except Exception as e:
                    logger.warning(f"Time-range filtering failed: {e}, falling back to full scan")
                    # 回退到全量扫描+手动过滤
                    for topic, msg, t in bag.read_messages(topics=list(topic_to_key.keys())):
                        ts = t.to_sec()
                        if start_time <= ts <= end_time:
                            key = topic_to_key.get(topic)
                            if key:
                                msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                                # 立即处理消息，转换为数据格式
                                msg_data = msg_process_fn(msg)
                                msg_data["timestamp"] = ts
                                chunk_data[key][ts] = msg_data
                                # 立即删除原始消息对象
                                del msg
        else:
            # 不使用线程池限制（如果threadpoolctl不可用）
            try:
                start_ros_time = rospy.Time.from_sec(start_time)
                end_ros_time = rospy.Time.from_sec(end_time)
                
                for topic, msg, t in bag.read_messages(
                    topics=list(topic_to_key.keys()),
                    start_time=start_ros_time,
                    end_time=end_ros_time
                ):
                    key = topic_to_key.get(topic)
                    if key:
                        msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                        msg_data = msg_process_fn(msg)
                        msg_data["timestamp"] = t.to_sec()
                        chunk_data[key][t.to_sec()] = msg_data
                        del msg
            except Exception as e:
                logger.warning(f"Time-range filtering failed: {e}, falling back to full scan")
                for topic, msg, t in bag.read_messages(topics=list(topic_to_key.keys())):
                    ts = t.to_sec()
                    if start_time <= ts <= end_time:
                        key = topic_to_key.get(topic)
                        if key:
                            msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                            msg_data = msg_process_fn(msg)
                            msg_data["timestamp"] = ts
                            chunk_data[key][ts] = msg_data
                            del msg
        
        # 清理临时变量（参照文档：及时释放中间变量）
        del topic_to_key
        
        return dict(chunk_data)
    
    def _align_single_frame(
        self,
        main_stamp: float,
        global_idx: int,
        chunk_data: Dict[str, Dict[float, dict]],
        timestamp_arrays: Dict[str, np.ndarray],
        alignment_indices: Dict[str, List[int]],
        arm_traj_gaps: List[Tuple[float, float]]
    ) -> dict:
        """
        对齐单帧数据（创建数据副本，避免保留对chunk_data的引用）
        
        注意：为了确保内存能够被释放，这里会创建数据的副本。
        对于numpy数组，使用.copy()；对于字典，使用浅拷贝。
        """
        aligned_frame = {"timestamp": main_stamp}
        
        for key in self._topic_process_map.keys():
            # 特殊处理kuavo_arm_traj的间隙
            if key == "action.kuavo_arm_traj" and arm_traj_gaps:
                in_gap = any(gap_start < main_stamp < gap_end 
                            for gap_start, gap_end in arm_traj_gaps)
                if in_gap:
                    # 在间隙中，使用999填充
                    sample_data = next(iter(chunk_data.get(key, {}).values()), None)
                    if sample_data and "data" in sample_data:
                        data_dim = len(sample_data["data"])
                        aligned_frame[key] = {
                            "data": np.full(data_dim, 999.0, dtype=np.float32),
                            "timestamp": main_stamp
                        }
                    continue
            
            # 获取预计算的索引
            if key not in alignment_indices or global_idx >= len(alignment_indices[key]):
                aligned_frame[key] = None
                continue
            
            closest_idx = alignment_indices[key][global_idx]
            
            # 从timestamp_arrays获取对应的时间戳
            if key not in timestamp_arrays or len(timestamp_arrays[key]) == 0:
                aligned_frame[key] = None
                continue
            
            target_ts = timestamp_arrays[key][closest_idx]
            
            # 从chunk_data中查找数据（创建副本，避免保留引用）
            if key in chunk_data:
                chunk_key_data = chunk_data[key]
                if chunk_key_data:
                    # 查找最接近target_ts的数据（避免创建临时列表）
                    closest_chunk_ts = None
                    min_diff = float('inf')
                    for ts in chunk_key_data.keys():
                        diff = abs(ts - target_ts)
                        if diff < min_diff:
                            min_diff = diff
                            closest_chunk_ts = ts
                        # 清理循环中的临时变量
                        del diff
                    # 清理临时变量
                    del min_diff
                    
                    if closest_chunk_ts is not None:
                        source_data = chunk_key_data[closest_chunk_ts]
                        
                        # 创建数据副本（避免保留对chunk_data的引用）
                        if source_data is None:
                            aligned_frame[key] = None
                        elif isinstance(source_data, dict):
                            # 对于字典，创建浅拷贝，但对其中的numpy数组创建副本
                            aligned_frame[key] = {}
                            for k, v in source_data.items():
                                if isinstance(v, np.ndarray):
                                    aligned_frame[key][k] = v.copy()
                                else:
                                    aligned_frame[key][k] = v
                        elif isinstance(source_data, np.ndarray):
                            aligned_frame[key] = source_data.copy()
                        else:
                            # 其他类型，直接赋值（通常是标量）
                            aligned_frame[key] = source_data
                        
                        # 清理临时变量
                        del source_data
                    else:
                        aligned_frame[key] = None
                    
                    # 清理临时变量
                    del closest_chunk_ts
                else:
                    aligned_frame[key] = None
                
                # 清理临时变量
                del chunk_key_data
            else:
                aligned_frame[key] = None
        
        return aligned_frame
    
    def _load_bag(self, bag_file: str) -> rosbag.Bag:
        """加载rosbag文件"""
        try:
            return rosbag.Bag(bag_file)
        except rosbag.bag.ROSBagUnindexedException:
            logger.warning(f"Bag file {bag_file} is unindexed, attempting to reindex...")
            from .utils import reindex_rosbag
            reindexed_file = reindex_rosbag(bag_file)
            if reindexed_file:
                return rosbag.Bag(reindexed_file)
            else:
                return rosbag.Bag(bag_file, 'r', allow_unindexed=True)






