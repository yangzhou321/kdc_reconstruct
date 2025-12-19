#!/usr/bin/env python3
"""
分块流式rosbag转换器 - 低内存版本

核心优化（参考Diffusion Policy的按需读取方式）：
1. 第一遍扫描：只读取时间戳（内存占用几MB）
2. 第二遍扫描：按时间窗口分块读取+对齐+写入dataset

与原始CvtRosbag2Lerobot.py的区别：
- 原始：一次性加载整个rosbag到内存 → 对齐 → 写入（内存峰值巨大）
- 本版：分块读取 → 即时对齐 → 即时写入 → 释放内存（内存可控）

使用方法：
    python CvtRosbag2Lerobot_chunked.py --config-name=KuavoRosbag2Lerobot \
        rosbag.rosbag_dir=/path/to/rosbag \
        rosbag.lerobot_dir=/path/to/output \
        rosbag.chunk_size=100
"""

import os
import gc
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME

from kuavo_data.common import kuavo_dataset as kuavo
from kuavo_data.common.config_dataset import DEFAULT_DATASET_CONFIG, DatasetConfig

# 复用原始脚本的工具函数
from kuavo_data.CvtRosbag2Lerobot import (
    get_cameras,
    get_motors,
    state_dim,
    action_dim,
    state_name,
    action_name,
    log_print,
)


def create_empty_dataset_chunked(
    repo_id: str,
    robot_type: str = "kuavo4pro",
    mode: Literal["video", "image"] = "image",
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str = None,
) -> LeRobotDataset:
    """创建空数据集（与原始函数相同）"""
    cameras = kuavo.DEFAULT_CAMERA_NAMES
    motors = get_motors()

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_dim,
            "names": {
                "state_names": state_name
            }
        },
        "action": {
            "dtype": "float32",
            "shape": action_dim,
            "names": {
                "action_names": action_name
            }
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.{cam}"] = {
                "dtype": "uint16",
                "shape": (1, kuavo.RESIZE_H, kuavo.RESIZE_W),
                "names": ["channels", "height", "width"],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
                "names": ["channels", "height", "width"],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def populate_dataset_chunked(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    chunk_size: int = 100,
) -> LeRobotDataset:
    """
    使用分块流式处理填充数据集
    
    核心优化：
    1. 第一遍扫描只读取时间戳（内存几MB）
    2. 第二遍扫描按时间窗口分块读取+对齐+写入
    3. 每个chunk处理完立即保存并释放内存
    
    Args:
        dataset: LeRobotDataset实例
        bag_files: rosbag文件路径列表
        task: 任务描述
        episodes: 要处理的episode索引列表
        chunk_size: 每个chunk包含的帧数（默认100帧）
    """
    if episodes is None:
        episodes = range(len(bag_files))
    
    failed_bags = []
    bag_reader = kuavo.KuavoRosbagReader()
    
    # 内存监控
    process = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        pass
    
    def log_memory(prefix: str):
        if process:
            mem_mb = process.memory_info().rss / 1024 / 1024
            log_print.info(f"{prefix} Memory: {mem_mb:.2f} MB")
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        log_print.info(f"Processing {ep_path}")
        log_memory("Before processing")
        
        try:
            # 收集当前episode的所有帧
            frames_buffer = []
            frame_count = [0]
            
            def on_frame(aligned_frame: dict, frame_idx: int):
                """处理单帧对齐数据"""
                # 提取state和action
                state_data = aligned_frame.get('observation.state')
                action_data = aligned_frame.get('action')
                
                if state_data is None or action_data is None:
                    return
                
                state = np.array(state_data.get('data', []), dtype=np.float32)
                action = np.array(action_data.get('data', []), dtype=np.float32)
                
                if len(state) == 0 or len(action) == 0:
                    return
                
                # 处理手部数据
                claw_state_data = aligned_frame.get('observation.claw')
                claw_action_data = aligned_frame.get('action.claw')
                qiangnao_state_data = aligned_frame.get('observation.qiangnao')
                qiangnao_action_data = aligned_frame.get('action.qiangnao')
                rq2f85_state_data = aligned_frame.get('observation.rq2f85')
                rq2f85_action_data = aligned_frame.get('action.rq2f85')
                
                claw_state = np.array(claw_state_data.get('data', []), dtype=np.float64) if claw_state_data else np.array([])
                claw_action = np.array(claw_action_data.get('data', []), dtype=np.float64) if claw_action_data else np.array([])
                qiangnao_state = np.array(qiangnao_state_data.get('data', []), dtype=np.float64) if qiangnao_state_data else np.array([])
                qiangnao_action = np.array(qiangnao_action_data.get('data', []), dtype=np.float64) if qiangnao_action_data else np.array([])
                rq2f85_state = np.array(rq2f85_state_data.get('data', []), dtype=np.float64) if rq2f85_state_data else np.array([])
                rq2f85_action = np.array(rq2f85_action_data.get('data', []), dtype=np.float64) if rq2f85_action_data else np.array([])
                
                # 手部数据归一化
                if kuavo.IS_BINARY:
                    qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
                    qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
                    claw_state = np.where(claw_state > 50, 1, 0)
                    claw_action = np.where(claw_action > 50, 1, 0)
                    rq2f85_state = np.where(rq2f85_state > 0.4, 1, 0)
                    rq2f85_action = np.where(rq2f85_action > 70, 1, 0)
                else:
                    claw_state = claw_state / 100 if len(claw_state) > 0 else claw_state
                    claw_action = claw_action / 100 if len(claw_action) > 0 else claw_action
                    qiangnao_state = qiangnao_state / 100 if len(qiangnao_state) > 0 else qiangnao_state
                    qiangnao_action = qiangnao_action / 100 if len(qiangnao_action) > 0 else qiangnao_action
                    rq2f85_state = rq2f85_state / 0.8 if len(rq2f85_state) > 0 else rq2f85_state
                    rq2f85_action = rq2f85_action / 140 if len(rq2f85_action) > 0 else rq2f85_action
                
                if len(claw_action) == 0 and len(qiangnao_action) == 0:
                    claw_action = rq2f85_action
                    claw_state = rq2f85_state
                
                # 构建输出state和action
                if kuavo.USE_LEJU_CLAW or kuavo.USE_QIANGNAO:
                    hand_type = "LEJU" if kuavo.USE_LEJU_CLAW else "QIANGNAO"
                    s_list, a_list = [], []
                    
                    def get_hand_slice(hand_side, hand_type):
                        if hand_type == "LEJU":
                            s_slice = kuavo.SLICE_ROBOT[hand_side]
                            c_slice = kuavo.SLICE_CLAW[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]], claw_state[c_slice[0]:c_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]], claw_action[c_slice[0]:c_slice[-1]]))
                        else:
                            s_slice = kuavo.SLICE_ROBOT[hand_side]
                            d_slice = kuavo.SLICE_DEX[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]], qiangnao_state[d_slice[0]:d_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]], qiangnao_action[d_slice[0]:d_slice[-1]]))
                        return s, a
                    
                    if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
                        s, a = get_hand_slice(0, hand_type)
                        s_list.append(s)
                        a_list.append(a)
                    if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
                        s, a = get_hand_slice(1, hand_type)
                        s_list.append(s)
                        a_list.append(a)
                    
                    output_state = np.concatenate(s_list).astype(np.float32)
                    output_action = np.concatenate(a_list).astype(np.float32)
                else:
                    output_state = state.astype(np.float32)
                    output_action = action.astype(np.float32)
                
                final_state = output_state
                final_action = output_action
                
                # 处理cmd_pos_world和gap_flag
                if not kuavo.ONLY_HALF_UP_BODY:
                    cmd_pos_world_data = aligned_frame.get('action.cmd_pos_world')
                    action_kuavo_arm_traj_data = aligned_frame.get('action.kuavo_arm_traj')
                    
                    cmd_pos_world = np.array(cmd_pos_world_data.get('data', [0, 0, 0]), dtype=np.float32) if cmd_pos_world_data else np.zeros(3, dtype=np.float32)
                    action_kuavo_arm_traj = np.array(action_kuavo_arm_traj_data.get('data', []), dtype=np.float32) if action_kuavo_arm_traj_data else np.array([])
                    
                    gap_flag = 1.0 if len(action_kuavo_arm_traj) > 0 and np.any(action_kuavo_arm_traj == 999.0) else 0.0
                    final_action = np.concatenate([
                        final_action,
                        cmd_pos_world,
                        np.array([gap_flag], dtype=np.float32)
                    ], axis=0)
                
                # 构建帧数据
                frame = {
                    "observation.state": torch.from_numpy(final_state).type(torch.float32),
                    "action": torch.from_numpy(final_action).type(torch.float32),
                }
                
                # 处理图像
                for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
                    cam_data = aligned_frame.get(cam_key)
                    if cam_data and 'data' in cam_data:
                        img = cam_data['data']
                        if "depth" in cam_key:
                            min_depth, max_depth = kuavo.DEPTH_RANGE[0], kuavo.DEPTH_RANGE[1]
                            frame[f"observation.{cam_key}"] = np.clip(img, min_depth, max_depth)
                        else:
                            frame[f"observation.images.{cam_key}"] = img
                
                frames_buffer.append(frame)
                frame_count[0] += 1
            
            def on_chunk_done():
                """每个chunk处理完后的回调：保存并释放内存"""
                if len(frames_buffer) == 0:
                    return
                
                # 将所有缓存的帧添加到dataset
                for frame in frames_buffer:
                    dataset.add_frame(frame, task=task)
                
                # 保存当前chunk
                dataset.save_episode()
                dataset.hf_dataset = dataset.create_hf_dataset()
                
                # 清空buffer并释放内存
                frames_buffer.clear()
                gc.collect()
                
                log_memory(f"After saving chunk (total frames: {frame_count[0]})")
            
            # 使用分块流式处理
            bag_reader.process_rosbag_chunked(
                bag_file=str(ep_path),
                frame_callback=on_frame,
                chunk_size=chunk_size,
                save_callback=on_chunk_done
            )
            
            # 处理剩余的帧
            if len(frames_buffer) > 0:
                for frame in frames_buffer:
                    dataset.add_frame(frame, task=task)
                dataset.save_episode()
                dataset.hf_dataset = dataset.create_hf_dataset()
                frames_buffer.clear()
                gc.collect()
            
            log_print.info(f"Episode {ep_idx} completed: {frame_count[0]} frames")
            
        except Exception as e:
            log_print.error(f"Error processing {ep_path}: {e}")
            import traceback
            traceback.print_exc()
            failed_bags.append(str(ep_path))
            continue
        
        log_memory("After episode")
        gc.collect()
    
    if failed_bags:
        with open("error.txt", "w") as f:
            for bag in failed_bags:
                f.write(bag + "\n")
        log_print.error(f"{len(failed_bags)} failed bags written to error.txt")
    
    return dataset


def port_kuavo_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
    chunk_size: int = 100,
):
    """
    分块流式转换rosbag到LeRobot格式
    
    Args:
        raw_dir: rosbag目录
        repo_id: 输出数据集ID
        task: 任务描述
        chunk_size: 每个chunk的帧数（默认100）
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    if isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(f"Requested {n} bags, but only {num_available_bags} available.")
            n = num_available_bags
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]
    
    dataset = create_empty_dataset_chunked(
        repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
        root=root,
    )
    
    dataset = populate_dataset_chunked(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
        chunk_size=chunk_size,
    )
    
    return dataset


@hydra.main(
    config_path="../configs/data/",
    config_name="KuavoRosbag2Lerobot",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    分块流式转换入口
    
    使用方法：
        python CvtRosbag2Lerobot_chunked.py \
            rosbag.rosbag_dir=/path/to/rosbag \
            rosbag.lerobot_dir=/path/to/output \
            rosbag.chunk_size=100
    """
    chunk_size = cfg.rosbag.get("chunk_size", 100)
    
    log_print.info(f"=== Chunked Streaming Rosbag Converter ===")
    log_print.info(f"Rosbag dir: {cfg.rosbag.rosbag_dir}")
    log_print.info(f"Output dir: {cfg.rosbag.lerobot_dir}")
    log_print.info(f"Chunk size: {chunk_size}")
    
    port_kuavo_rosbag_chunked(
        raw_dir=cfg.rosbag.rosbag_dir,
        repo_id="kuavo_lerobot_data",
        task=cfg.task,
        mode="image",
        root=cfg.rosbag.lerobot_dir,
        n=cfg.rosbag.num_used,
        chunk_size=chunk_size,
    )
    
    log_print.info("Conversion completed!")


if __name__ == "__main__":
    main()






