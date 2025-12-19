"""
流式处理rosbag的模块，避免一次性加载所有数据到内存

核心思想：
1. 第一遍扫描：只读取时间戳，确定主时间线和时间戳序列（不加载图像数据）
2. 第二遍扫描：按主时间线流式读取，边对齐边处理边添加到dataset

注意：虽然叫"流式处理"，但由于对齐的需要，第二遍扫描仍需要加载所有消息数据。
但相比原始方法，第一遍扫描只读取时间戳，内存占用大幅减少。
"""

import numpy as np
import rosbag
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Generator, Tuple
import logging

logger = logging.getLogger(__name__)


class StreamingRosbagProcessor:
    """
    流式处理rosbag，避免一次性加载所有数据到内存
    """
    
    def __init__(self, msg_processer, topic_process_map, camera_names, 
                 train_hz, main_timeline_fps, sample_drop, only_half_up_body):
        """
        Args:
            msg_processer: 消息处理器实例
            topic_process_map: 话题到处理函数的映射
            camera_names: 相机名称列表
            train_hz: 训练频率
            main_timeline_fps: 主时间线帧率
            sample_drop: 丢弃的帧数
            only_half_up_body: 是否只使用上半身
        """
        self._msg_processer = msg_processer
        self._topic_process_map = topic_process_map
        self.camera_names = camera_names
        self.train_hz = train_hz
        self.main_timeline_fps = main_timeline_fps
        self.sample_drop = sample_drop
        self.only_half_up_body = only_half_up_body
    
    def scan_timestamps(self, bag_file: str) -> Tuple[str, List[float], Dict[str, List[float]]]:
        """
        第一遍扫描：只读取时间戳，确定主时间线和时间戳序列
        
        Returns:
            main_timeline: 主时间线话题名称
            main_timestamps: 主时间戳序列
            topic_timestamps: 每个话题的时间戳列表
        """
        bag = self._load_bag(bag_file)
        
        # 统计每个话题的消息数量（只读取时间戳，不加载数据）
        topic_counts = {}
        topic_timestamps = defaultdict(list)
        
        logger.info("Scanning rosbag to determine main timeline...")
        
        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            count = 0
            for _, msg, t in bag.read_messages(topics=topic):
                timestamp = t.to_sec()
                topic_timestamps[key].append(timestamp)
                count += 1
            topic_counts[key] = count
            logger.debug(f"Topic {key}: {count} messages")
        
        bag.close()
        
        # 确定主时间线：选择消息数量最多的相机
        camera_counts = {k: topic_counts.get(k, 0) for k in self.camera_names if k in topic_counts}
        if not camera_counts:
            raise ValueError("No camera topics found in rosbag")
        
        main_timeline = max(camera_counts, key=lambda k: camera_counts[k])
        logger.info(f"Main timeline: {main_timeline} ({camera_counts[main_timeline]} messages)")
        
        # 生成主时间戳序列
        jump = self.main_timeline_fps // self.train_hz
        main_timestamps_raw = topic_timestamps[main_timeline]
        
        if len(main_timestamps_raw) < 2 * self.sample_drop:
            raise ValueError(f"Not enough frames in main timeline: {len(main_timestamps_raw)}")
        
        main_timestamps = main_timestamps_raw[self.sample_drop:-self.sample_drop][::jump]
        
        # 找到所有话题的最小结束时间
        min_end = min([
            timestamps[-1] 
            for timestamps in topic_timestamps.values() 
            if len(timestamps) > 0
        ])
        
        # 过滤掉超过最小结束时间的时间戳
        main_timestamps = [t for t in main_timestamps if t < min_end]
        
        logger.info(f"Generated {len(main_timestamps)} main timestamps (from {len(main_timestamps_raw)} raw frames)")
        
        return main_timeline, main_timestamps, topic_timestamps
    
    def stream_process_rosbag(
        self, 
        bag_file: str,
        main_timeline: str,
        main_timestamps: List[float],
        frame_callback: Callable[[Dict], None],
        chunk_size: Optional[int] = None
    ) -> None:
        """
        第二遍扫描：流式读取，按主时间线对齐并处理
        
        Args:
            bag_file: rosbag文件路径
            main_timeline: 主时间线话题名称
            main_timestamps: 主时间戳序列
            frame_callback: 对齐后的帧数据回调函数
            chunk_size: 每处理多少帧调用一次回调（None表示处理完所有帧）
        """
        bag = self._load_bag(bag_file)
        
        # 为每个话题创建时间戳索引（用于快速查找）
        topic_data_buffers = defaultdict(list)
        topic_timestamp_arrays = {}
        
        # 读取所有消息到缓冲区（但只保留必要的窗口大小）
        logger.info("Loading messages into buffers...")
        
        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            msg_process_fn = topic_info["msg_process_fn"]
            
            timestamps = []
            data_list = []
            
            for _, msg, t in bag.read_messages(topics=topic):
                msg_data = msg_process_fn(msg)
                timestamp = t.to_sec()
                timestamps.append(timestamp)
                data_list.append(msg_data)
            
            if len(timestamps) > 0:
                topic_timestamp_arrays[key] = np.array(timestamps)
                topic_data_buffers[key] = data_list
                logger.debug(f"Loaded {len(timestamps)} messages for {key}")
        
        bag.close()
        
        # 处理kuavo_arm_traj的特殊情况（检测间隙）
        arm_traj_gaps = []
        if not self.only_half_up_body and "action.kuavo_arm_traj" in topic_timestamp_arrays:
            arm_traj_timestamps = topic_timestamp_arrays["action.kuavo_arm_traj"]
            if len(arm_traj_timestamps) > 1:
                gap_threshold = 0.15 * 10 / self.train_hz
                for i in range(1, len(arm_traj_timestamps)):
                    if arm_traj_timestamps[i] - arm_traj_timestamps[i-1] > gap_threshold:
                        arm_traj_gaps.append((arm_traj_timestamps[i-1], arm_traj_timestamps[i]))
        
        # 按主时间戳流式处理
        logger.info(f"Processing {len(main_timestamps)} frames...")
        
        processed_count = 0
        for frame_idx, main_stamp in enumerate(main_timestamps):
            aligned_frame = {}
            
            # 对齐每个话题的数据
            for key in self._topic_process_map.keys():
                if key == "action.kuavo_arm_traj" and len(arm_traj_gaps) > 0:
                    # 特殊处理：检查是否在间隙中
                    in_gap = False
                    for gap_start, gap_end in arm_traj_gaps:
                        if gap_start < main_stamp < gap_end:
                            in_gap = True
                            break
                    
                    if in_gap:
                        # 在间隙中，使用999填充
                        data_dim = len(topic_data_buffers["action.kuavo_arm_traj"][0]["data"])
                        aligned_frame[key] = {
                            "data": np.full(data_dim, 999.0, dtype=np.float32),
                            "timestamp": main_stamp
                        }
                    else:
                        # 正常对齐
                        aligned_frame[key] = self._align_single_topic(
                            key, main_stamp, topic_timestamp_arrays, topic_data_buffers
                        )
                else:
                    aligned_frame[key] = self._align_single_topic(
                        key, main_stamp, topic_timestamp_arrays, topic_data_buffers
                    )
            
            # 调用回调函数处理对齐后的帧
            frame_callback(aligned_frame)
            processed_count += 1
            
            # 如果设置了chunk_size，每处理chunk_size帧后可以做一些清理
            if chunk_size and processed_count % chunk_size == 0:
                logger.debug(f"Processed {processed_count} frames")
        
        logger.info(f"Finished processing {processed_count} frames")
    
    def _align_single_topic(
        self,
        key: str,
        target_timestamp: float,
        topic_timestamp_arrays: Dict[str, np.ndarray],
        topic_data_buffers: Dict[str, List]
    ) -> Optional[Dict]:
        """
        对齐单个话题到目标时间戳
        
        Returns:
            对齐后的消息数据，如果话题为空则返回None
        """
        if key not in topic_timestamp_arrays or len(topic_timestamp_arrays[key]) == 0:
            return None
        
        timestamps = topic_timestamp_arrays[key]
        data_list = topic_data_buffers[key]
        
        # 使用searchsorted找到最接近的时间戳
        idx = np.searchsorted(timestamps, target_timestamp)
        
        # 处理边界情况
        if idx == 0:
            closest_idx = 0
        elif idx == len(timestamps):
            closest_idx = len(timestamps) - 1
        else:
            # 选择更接近的时间戳
            if abs(timestamps[idx] - target_timestamp) < abs(timestamps[idx-1] - target_timestamp):
                closest_idx = idx
            else:
                closest_idx = idx - 1
        
        aligned_data = data_list[closest_idx].copy()
        aligned_data["timestamp"] = target_timestamp  # 使用目标时间戳
        
        return aligned_data
    
    def _load_bag(self, bag_file: str) -> rosbag.Bag:
        """加载rosbag文件，处理未索引的情况"""
        try:
            return rosbag.Bag(bag_file)
        except rosbag.bag.ROSBagUnindexedException:
            logger.warning(f"Bag file {bag_file} is unindexed, attempting to reindex...")
            from common.utils import reindex_rosbag
            reindexed_file = reindex_rosbag(bag_file)
            if reindexed_file:
                try:
                    return rosbag.Bag(reindexed_file)
                except Exception as e:
                    logger.error(f"Error loading reindexed bag file: {e}")
                    raise RuntimeError(f"Failed to load reindexed bag file: {reindexed_file}")
            else:
                logger.warning("Reindexing failed, trying to open with allow_unindexed=True")
                try:
                    return rosbag.Bag(bag_file, 'r', allow_unindexed=True)
                except Exception as e:
                    logger.error(f"Failed to open unindexed bag: {e}")
                    raise RuntimeError(f"Failed to reindex and load bag file: {bag_file}")
        except Exception as e:
            logger.error(f"Error loading bag file {bag_file}: {e}")
            raise

