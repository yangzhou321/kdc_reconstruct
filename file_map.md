# Kuavo Data Challenge 项目文件映射

## 项目结构树

```
kuavo_data_challenge/
├── configs/                          # 配置文件目录
│   ├── __init__.py                   # 配置模块初始化
│   ├── data/                         # 数据转换配置
│   │   └── KuavoRosbag2Lerobot.yaml # Rosbag转Lerobot配置
│   ├── deploy/                       # 部署配置
│   │   ├── config_inference.py      # 推理配置加载
│   │   ├── config_kuavo_env.py      # Kuavo环境配置加载
│   │   ├── kuavo_real_env.yaml      # 真实机器人环境配置
│   │   ├── kuavo_sim_env.yaml       # 仿真环境配置
│   │   └── others_env.yaml          # 其他环境配置
│   └── policy/                       # 策略配置
│       ├── act_config.yaml           # ACT策略配置
│       └── diffusion_config.yaml     # Diffusion策略配置
├── docker/                           # Docker相关
│   ├── readme.md                     # Docker说明文档
│   └── run_with_gpu.sh              # GPU运行脚本
├── Dockerfile                        # Docker镜像定义
├── kuavo_data/                       # 数据转换模块
│   ├── common/                       # 通用工具
│   │   ├── __init__.py               # 模块初始化
│   │   ├── config_dataset.py         # 数据集配置加载
│   │   ├── key_listener.py           # 键盘监听器
│   │   ├── kuavo_dataset.py          # Kuavo数据集处理
│   │   ├── logger.py                 # 日志工具
│   │   ├── ros_handler.py            # ROS消息处理
│   │   └── utils.py                  # 工具函数
│   └── CvtRosbag2Lerobot.py         # Rosbag转Lerobot主程序
├── kuavo_deploy/                     # 部署模块
│   ├── __init__.py                   # 模块初始化
│   ├── eval_kuavo.sh                 # 评估脚本
│   ├── eval_others.py                # 其他环境评估
│   ├── examples/                     # 示例代码
│   │   ├── eval/                     # 评估示例
│   │   │   ├── auto_test/            # 自动测试
│   │   │   │   ├── eval_kuavo.py     # 评估脚本
│   │   │   │   └── eval_kuavo_autotest.py # 自动测试评估
│   │   │   └── eval_kuavo.py         # 主评估脚本
│   │   └── scripts/                  # 脚本示例
│   │       ├── controller.py         # 控制器
│   │       ├── script.py              # 脚本
│   │       └── script_auto_test.py   # 自动测试脚本
│   ├── kuavo_env/                    # Kuavo环境
│   │   ├── __init__.py               # 环境注册
│   │   ├── KuavoBaseRosEnv.py        # ROS环境基类
│   │   ├── kuavo_real_env/           # 真实机器人环境
│   │   │   ├── __init__.py
│   │   │   └── KuavoRealEnv.py       # 真实环境实现
│   │   └── kuavo_sim_env/            # 仿真环境
│   │       ├── __init__.py
│   │       └── KuavoSimEnv.py        # 仿真环境实现
│   ├── readme/                       # 文档目录
│   │   ├── inference.md              # 推理文档
│   │   ├── setup_bridge.sh           # 桥接设置脚本
│   │   ├── setup_env.md              # 环境设置文档
│   │   ├── setup_env.sh              # 环境设置脚本
│   │   └── setup_robot_connection.md # 机器人连接设置
│   ├── readme.md                     # 部署模块说明
│   └── utils/                        # 工具函数
│       └── logging_utils.py          # 日志工具
├── kuavo_train/                      # 训练模块
│   ├── __init__.py                   # 模块初始化
│   ├── README.md                     # 训练模块说明
│   ├── train_policy.py               # 策略训练主程序
│   ├── utils/                        # 工具函数
│   │   ├── __init__.py
│   │   ├── augmenter.py              # 数据增强器
│   │   ├── transforms.py             # 图像变换
│   │   └── utils.py                  # 通用工具
│   └── wrapper/                      # 包装器
│       ├── dataset/                  # 数据集包装器
│       │   └── LeRobotDatasetWrapper.py # LeRobot数据集包装
│       └── policy/                   # 策略包装器
│           └── diffusion/           # Diffusion策略
│               ├── __init__.py
│               ├── DiffusionConfigWrapper.py # 配置包装
│               ├── DiffusionModelWrapper.py # 模型包装
│               ├── DiffusionPolicyWrapper.py # 策略包装
│               ├── DiT_1D_model.py   # 1D DiT模型
│               ├── DiT_model.py      # DiT模型
│               └── transformer_diffusion.py # Transformer Diffusion
├── lerobot_patches/                  # Lerobot补丁
│   ├── __init__.py                   # 模块初始化
│   └── custom_patches.py             # 自定义补丁
├── README.md                         # 项目主文档
├── README_ZH.md                      # 项目中文文档
├── requirements_ilcode.txt           # 模仿学习代码依赖
├── requirements_total.txt            # 完整依赖列表
├── setup.py                          # 安装配置
└── third_party/                      # 第三方库
    └── lerobot/                      # Lerobot框架
```

---

## 文件详细说明

### 根目录文件

#### `setup.py`
- **作用**: Python包安装配置文件
- **主要变量/函数**:
  - `name`: 包名 "kuavo_data_challenge"
  - `version`: 版本号 "0.1"
  - `packages`: 自动发现的包列表

#### `README.md` / `README_ZH.md`
- **作用**: 项目说明文档（英文/中文）
- **内容**: 项目概述、安装指南、使用方法、ROS话题说明等

#### `requirements_ilcode.txt` / `requirements_total.txt`
- **作用**: Python依赖包列表
- **区别**: 
  - `requirements_ilcode.txt`: 仅模仿学习训练所需依赖（不含ROS）
  - `requirements_total.txt`: 完整依赖（包含ROS相关）

#### `Dockerfile`
- **作用**: Docker镜像构建文件

---

### configs/ 配置目录

#### `configs/__init__.py`
- **作用**: 配置模块初始化文件（空文件）

#### `configs/data/KuavoRosbag2Lerobot.yaml`
- **作用**: Rosbag转Lerobot数据格式的配置文件
- **配置项**: 数据集参数、相机选择、深度图像使用等

#### `configs/deploy/config_inference.py`
- **作用**: 推理配置加载模块
- **主要类**:
  - `Config_Inference`: 推理配置数据类
    - `go_bag_path`: bag文件路径
    - `policy_type`: 策略类型（diffusion/act）
    - `use_delta`: 是否使用delta动作
    - `eval_episodes`: 评估回合数
    - `seed`: 随机种子
    - `device`: 设备（cuda/cpu）
    - `task`: 任务名称
    - `method`: 方法名称
    - `timestamp`: 时间戳
    - `epoch`: 模型epoch
    - `max_episode_steps`: 最大步数
    - `env_name`: 环境名称
    - `depth_range`: 深度图像范围
- **主要函数**:
  - `load_inference_config(config_path)`: 从YAML加载推理配置

#### `configs/deploy/config_kuavo_env.py`
- **作用**: Kuavo环境配置加载模块
- **主要类**:
  - `Config_Kuavo_Env`: Kuavo环境配置数据类
    - `real`: 是否真实机器人
    - `only_arm`: 是否仅手臂控制
    - `eef_type`: 末端执行器类型（qiangnao/leju_claw/rq2f85）
    - `control_mode`: 控制模式（joint/eef）
    - `which_arm`: 使用的手臂（left/right/both）
    - `qiangnao_dof_needed`: 灵巧手所需自由度
    - `leju_claw_dof_needed`: 夹爪所需自由度
    - `rq2f85_dof_needed`: rq2f85夹爪所需自由度
    - `is_binary`: 是否二值化
    - `ros_rate`: ROS发布频率
    - `image_size`: 图像尺寸
    - `head_init`: 头部初始位置
    - `arm_init`: 手臂初始位置
    - `arm_min/max`: 手臂关节范围
    - `base_min/max`: 底盘移动范围
    - `eef_min/max`: 末端执行器范围
    - `input_images`: 输入图像列表
    - **属性方法**:
      - `use_leju_claw`: 是否使用leju夹爪
      - `use_qiangnao`: 是否使用灵巧手
      - `only_half_up_body`: 是否仅上半身
      - `default_camera_names`: 默认相机名称列表
      - `slice_robot`: 机器人切片配置
      - `qiangnao_slice`: 灵巧手切片配置
      - `claw_slice`: 夹爪切片配置
- **主要函数**:
  - `load_kuavo_env_config(config_path)`: 从YAML加载环境配置

#### `configs/policy/act_config.yaml` / `diffusion_config.yaml`
- **作用**: 策略训练配置文件
- **配置项**: 模型参数、训练参数、优化器、调度器等

---

### kuavo_data/ 数据转换模块

#### `kuavo_data/CvtRosbag2Lerobot.py`
- **作用**: 将Kuavo rosbag数据转换为Lerobot parquet格式的主程序
- **主要函数**:
  - `get_cameras(bag_data)`: 获取相机列表
  - `create_empty_dataset(...)`: 创建空数据集
  - `load_raw_images_per_camera(bag_data)`: 加载原始图像
  - `load_raw_episode_data(ep_path)`: 加载原始episode数据
  - `diagnose_frame_data(data)`: 诊断帧数据
  - `populate_dataset(...)`: 填充数据集
  - `port_kuavo_rosbag(...)`: 转换rosbag数据
  - `main(cfg)`: Hydra主函数
- **主要变量**:
  - `DEFAULT_DATASET_CONFIG`: 默认数据集配置
  - `DEFAULT_JOINT_NAMES_LIST`: 默认关节名称列表

#### `kuavo_data/common/config_dataset.py`
- **作用**: 数据集配置加载模块
- **主要类**:
  - `ResizeConfig`: 图像缩放配置
    - `width`: 宽度
    - `height`: 高度
  - `Config`: 数据集配置
    - `only_arm`: 是否仅手臂
    - `eef_type`: 末端执行器类型
    - `which_arm`: 使用的手臂
    - `use_depth`: 是否使用深度
    - `depth_range`: 深度范围
    - `dex_dof_needed`: 灵巧手自由度
    - `train_hz`: 训练频率
    - `main_timeline`: 主时间线
    - `main_timeline_fps`: 主时间线FPS
    - `sample_drop`: 采样丢弃数
    - `is_binary`: 是否二值化
    - `delta_action`: 是否delta动作
    - `relative_start`: 是否相对起始
    - `resize`: 缩放配置
    - `task_description`: 任务描述
- **主要函数**:
  - `load_config(cfg)`: 从配置加载Config对象

#### `kuavo_data/common/key_listener.py`
- **作用**: 键盘监听器，用于交互式控制
- **主要类**:
  - `KeyListener`: 键盘监听器类
    - `exit_program`: 退出标志
    - `key_callbacks`: 按键回调字典
    - `crtk_c_callback`: Ctrl-C回调
    - `old_settings`: 终端原始设置
    - **方法**:
      - `register_ctrlC_callback(callback)`: 注册Ctrl-C回调
      - `register_callbacks(keys, callback)`: 批量注册回调
      - `register_callback(key, callback)`: 注册单个回调
      - `unregister_callback(key)`: 注销回调
      - `getKey()`: 获取按键
      - `on_press(key)`: 按键处理
      - `stop()`: 停止监听
      - `loop_control()`: 控制循环

#### `kuavo_data/common/kuavo_dataset.py`
- **作用**: Kuavo数据集处理核心模块
- **主要常量**:
  - `DEFAULT_LEG_JOINT_NAMES`: 腿部关节名称
  - `DEFAULT_ARM_JOINT_NAMES`: 手臂关节名称
  - `DEFAULT_HEAD_JOINT_NAMES`: 头部关节名称
  - `DEFAULT_DEXHAND_JOINT_NAMES`: 灵巧手关节名称
  - `DEFAULT_LEJUCLAW_JOINT_NAMES`: 夹爪关节名称
  - `DEFAULT_JOINT_NAMES_LIST`: 完整关节名称列表
- **主要函数**:
  - `init_parameters(cfg)`: 初始化全局参数
- **主要类**:
  - `KuavoMsgProcesser`: ROS消息处理器
    - **静态方法**:
      - `process_color_image(msg)`: 处理彩色图像
      - `process_depth_image(msg)`: 处理深度图像
      - `process_joint_state(msg)`: 处理关节状态
      - `process_joint_cmd(msg)`: 处理关节命令
      - `process_kuavo_arm_traj(msg)`: 处理手臂轨迹
      - `process_cmd_pos_world(msg)`: 处理世界坐标命令
      - `process_claw_state(msg)`: 处理夹爪状态
      - `process_claw_cmd(msg)`: 处理夹爪命令
      - `process_rq2f85_cmd(msg)`: 处理rq2f85命令
      - `process_rq2f85_state(msg)`: 处理rq2f85状态
      - `process_qiangnao_state(msg)`: 处理灵巧手状态
      - `process_dex_state(msg)`: 处理灵巧手状态
      - `process_qiangnao_cmd(msg)`: 处理灵巧手命令
      - `process_sensors_data_raw_extract_imu(msg)`: 提取IMU数据
      - `process_sensors_data_raw_extract_arm(msg)`: 提取手臂数据
      - `process_joint_cmd_extract_arm(msg)`: 提取手臂命令
      - `process_sensors_data_raw_extract_arm_head(msg)`: 提取手臂和头部数据
      - `process_joint_cmd_extract_arm_head(msg)`: 提取手臂和头部命令
  - `KuavoRosbagReader`: Rosbag读取器
    - `_msg_processer`: 消息处理器实例
    - `_topic_process_map`: 话题处理映射
    - **方法**:
      - `load_raw_rosbag(bag_file)`: 加载原始rosbag
      - `print_bag_info(bag)`: 打印bag信息
      - `process_rosbag(bag_file)`: 处理rosbag文件
      - `align_frame_data(data)`: 对齐帧数据
      - `list_bag_files(bag_dir)`: 列出bag文件
      - `process_rosbag_dir(bag_dir)`: 处理bag目录

#### `kuavo_data/common/logger.py`
- **作用**: 简单日志工具
- **主要类**:
  - `Logger`: 日志类
    - `log_file`: 日志文件路径
    - **方法**:
      - `log(text)`: 记录日志

#### `kuavo_data/common/ros_handler.py`
- **作用**: ROS消息处理器，用于多进程ROS通信
- **主要类**:
  - `ROSHandler`: ROS处理器类
    - `queue`: 消息队列
    - `topics_config`: 话题配置列表
    - **方法**:
      - `_extract_data(msg, topic_name)`: 提取数据
      - `_get_timestamp(msg)`: 获取时间戳
      - `_generic_callback(msg, topic_name)`: 通用回调
      - `run()`: 运行ROS节点
- **主要函数**:
  - `start_ros_handler(queue)`: 启动ROS处理器
  - `get_ros_queue(maxsize)`: 获取ROS队列

#### `kuavo_data/common/utils.py`
- **作用**: 通用工具函数
- **主要函数**:
  - `load_json(fpath)`: 加载JSON文件
  - `write_json(data, fpath)`: 写入JSON文件
  - `reindex_rosbag(bag_file)`: 重新索引rosbag文件

---

### kuavo_deploy/ 部署模块

#### `kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py`
- **作用**: Kuavo ROS环境基类，实现Gymnasium接口
- **主要类**:
  - `KuavoBaseRosEnv`: ROS环境基类（继承gym.Env）
    - `real`: 是否真实机器人
    - `ros_rate`: ROS频率
    - `control_mode`: 控制模式
    - `image_size`: 图像尺寸
    - `only_arm`: 是否仅手臂
    - `eef_type`: 末端执行器类型
    - `which_arm`: 使用的手臂
    - `qiangnao_dof_needed`: 灵巧手自由度
    - `leju_claw_dof_needed`: 夹爪自由度
    - `rq2f85_dof_needed`: rq2f85自由度
    - `arm_init`: 手臂初始位置
    - `slice_robot`: 机器人切片
    - `qiangnao_slice`: 灵巧手切片
    - `claw_slice`: 夹爪切片
    - `is_binary`: 是否二值化
    - `head_init`: 头部初始位置
    - `arm_min/max`: 手臂范围
    - `base_min/max`: 底盘范围
    - `eef_min/max`: 末端范围
    - `input_images`: 输入图像
    - `bridge`: CvBridge实例
    - `robot`: KuavoRobot实例
    - `robot_state`: KuavoRobotState实例
    - `action_space`: 动作空间
    - `observation_space`: 观测空间
    - **方法**:
      - `init_kuavo_sdk()`: 初始化SDK
      - `initial_topics()`: 初始化ROS话题
      - `check_rostopics()`: 检查ROS话题可用性
      - `reset(**kwargs)`: 重置环境
      - `check_action(action, mode)`: 检查动作
      - `step(action)`: 执行一步
      - `exec_action(action)`: 执行动作
      - `get_obs()`: 获取观测
      - `process_rgb_img(msg)`: 处理RGB图像
      - `process_depth_img(msg)`: 处理深度图像
      - `cam_h_callback(msg)`: 头部相机回调
      - `cam_l_callback(msg)`: 左手腕相机回调
      - `cam_r_callback(msg)`: 右手腕相机回调
      - `cam_h_depth_callback(msg)`: 头部深度相机回调
      - `cam_l_depth_callback(msg)`: 左手腕深度相机回调
      - `cam_r_depth_callback(msg)`: 右手腕深度相机回调
      - `gripper_state_callback(msg)`: 夹爪状态回调
      - `F_state_callback(msg)`: F状态回调（仿真用）
      - `compute_reward()`: 计算奖励
  - `LejuClaw`: Leju夹爪控制类
    - `_pub_leju_claw_cmd`: 夹爪命令发布器
    - **方法**:
      - `control(...)`: 控制夹爪
      - `control_left(...)`: 控制左手夹爪
      - `control_right(...)`: 控制右手夹爪
- **全局变量**:
  - `stop_flag`: 停止标志（threading.Event）
  - `pause_flag`: 暂停标志（threading.Event）
  - `pause_sub`: 暂停订阅者
  - `stop_sub`: 停止订阅者
- **全局函数**:
  - `pause_callback(msg)`: 暂停回调
  - `stop_callback(msg)`: 停止回调
  - `check_control_signals()`: 检查控制信号

#### `kuavo_deploy/kuavo_env/kuavo_real_env/KuavoRealEnv.py`
- **作用**: 真实机器人环境实现
- **主要类**:
  - `KuavoRealEnv`: 继承自KuavoBaseRosEnv
    - **方法**:
      - `compute_reward()`: 计算奖励（返回0）

#### `kuavo_deploy/kuavo_env/kuavo_sim_env/KuavoSimEnv.py`
- **作用**: 仿真环境实现
- **主要类**:
  - `KuavoSimEnv`: 继承自KuavoBaseRosEnv
    - **方法**:
      - `compute_reward()`: 计算奖励（返回0）

#### `kuavo_deploy/kuavo_env/__init__.py`
- **作用**: 注册Gymnasium环境
- **内容**: 注册'Kuavo-Sim'和'Kuavo-Real'环境

#### `kuavo_deploy/examples/eval/eval_kuavo.py`
- **作用**: Kuavo评估主程序
- **主要函数**:
  - `img_preprocess(image, device)`: 图像预处理
  - `depth_preprocess(depth, device, depth_range)`: 深度图像预处理
  - `setup_policy(pretrained_path, policy_type, device)`: 设置策略
  - `main(config_path, env)`: 主评估函数
  - `kuavo_eval(config_path, env)`: Kuavo评估入口
- **全局变量**:
  - `stop_flag`: 停止标志
  - `pause_flag`: 暂停标志
  - `pause_sub`: 暂停订阅者
  - `stop_sub`: 停止订阅者
- **全局函数**:
  - `pause_callback(msg)`: 暂停回调
  - `stop_callback(msg)`: 停止回调
  - `check_control_signals()`: 检查控制信号

#### `kuavo_deploy/utils/logging_utils.py`
- **作用**: 日志工具模块，提供彩色日志和文件日志功能
- **主要类**:
  - `ColoredFormatter`: 彩色日志格式化器
    - `DEFAULT_STYLE_CONFIG`: 默认样式配置
    - `style_config`: 样式配置
    - `is_console`: 是否控制台输出
    - **方法**:
      - `format(record)`: 格式化日志记录
  - `LoggerManager`: 日志管理器
    - `log_level`: 日志级别
    - `log_dir`: 日志目录
    - `loggers`: 日志记录器字典
    - `style_config`: 样式配置
    - `file_handler`: 文件处理器
    - **方法**:
      - `_setup_log_dir(log_dir)`: 设置日志目录
      - `get_logger(name)`: 获取或创建logger
- **主要函数**:
  - `get_log_manager(...)`: 获取全局日志管理器
  - `setup_logger(name, level, log_file, save_to_file)`: 设置日志记录器
  - `highlight_message(logger, message, color, attrs)`: 高亮消息
  - `test_logging()`: 测试日志功能

---

### kuavo_train/ 训练模块

#### `kuavo_train/train_policy.py`
- **作用**: 策略训练主程序
- **主要函数**:
  - `build_augmenter(cfg)`: 构建数据增强器
  - `build_delta_timestamps(dataset_metadata, policy_cfg)`: 构建delta时间戳
  - `build_optimizer_and_scheduler(policy, cfg, total_frames)`: 构建优化器和调度器
  - `build_policy_config(cfg, input_features, output_features)`: 构建策略配置
  - `build_policy(name, policy_cfg, dataset_stats)`: 构建策略
  - `main(cfg)`: Hydra主训练函数
- **主要变量**:
  - `device`: 训练设备
  - `output_directory`: 输出目录
  - `writer`: TensorBoard写入器
  - `dataset_metadata`: 数据集元数据
  - `input_features`: 输入特征
  - `output_features`: 输出特征
  - `policy_cfg`: 策略配置
  - `policy`: 策略模型
  - `optimizer`: 优化器
  - `lr_scheduler`: 学习率调度器
  - `scaler`: AMP梯度缩放器
  - `start_epoch`: 起始epoch
  - `steps`: 训练步数
  - `best_loss`: 最佳损失

#### `kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py`
- **作用**: LeRobot数据集包装器，扩展支持深度图像和RGB图像同步裁剪
- **主要类**:
  - `CustomLeRobotDataset`: 自定义LeRobot数据集（继承LeRobotDataset）
    - **方法**:
      - `__getitem__(idx)`: 获取数据项，支持深度图像和RGB图像同步处理

#### `kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py`
- **作用**: Diffusion策略包装器，支持RGB和深度图像
- **主要类**:
  - `CustomDiffusionPolicyWrapper`: 自定义Diffusion策略（继承DiffusionPolicy）
    - `diffusion`: CustomDiffusionModelWrapper实例
    - `_queues`: 观测和动作队列
    - **方法**:
      - `reset()`: 重置队列
      - `select_action(batch)`: 选择动作（推理时）
      - `forward(batch)`: 前向传播（训练时）
      - `from_pretrained(...)`: 从预训练模型加载
- **常量**:
  - `OBS_DEPTH`: 深度观测键名

#### `kuavo_train/utils/utils.py`
- **作用**: 训练工具函数
- **主要函数**:
  - `save_rng_state(filepath)`: 保存随机数生成器状态
  - `load_rng_state(filepath)`: 加载随机数生成器状态
  - `worker_init_fn(worker_id)`: DataLoader worker初始化函数

#### `kuavo_train/utils/augmenter.py`
- **作用**: 数据增强器模块
- **主要类**:
  - `DeterministicAugmenterColor`: 确定性颜色增强器
    - `augmentations`: 增强类型列表
    - `params`: 参数字典
    - `aug_type`: 当前增强类型
    - **方法**:
      - `set_random_params()`: 设置随机参数
      - `apply_augment_sequence(images)`: 应用增强序列
  - `DeterministicAugmenterGeo4Rgbds`: RGB+深度几何增强器（5通道）
    - **方法**:
      - `set_random_params()`: 设置随机参数
      - `apply_augment(image)`: 应用增强
      - `apply_augment_sequence(images)`: 应用增强序列
  - `DeterministicAugmenterGeo4Rgbdss`: RGB+深度几何增强器（6通道）
  - `DeterministicAugmenterGeo4Rgbd`: RGB+深度几何增强器（4通道）
  - `NoiseAdder_AfterNorm`: 归一化后噪声添加器
    - `state_noise_scale`: 状态噪声尺度
    - `action_noise_scale`: 动作噪声尺度
    - **方法**:
      - `add_noise(nsample, keys)`: 添加噪声
  - `NoiseAdder_AfterNorm2`: 归一化后噪声添加器（概率版本）
  - `Augmenter`: 增强器容器
    - `RGB_Augmenter`: RGB增强器
    - `crop_shape`: 裁剪形状
    - `resize_shape`: 缩放形状
- **主要函数**:
  - `resize_image(image, target_size, image_type)`: 缩放图像
  - `crop_image(image, target_range, random_crop)`: 裁剪图像

#### `kuavo_train/utils/transforms.py`
- **作用**: 图像变换模块，基于torchvision.transforms.v2
- **主要类**:
  - `RandomMask`: 随机掩码变换
    - `mask_size`: 掩码大小
    - **方法**:
      - `make_params(flat_inputs)`: 生成参数
      - `transform(inpt, params)`: 应用变换
  - `RandomBorderCutout`: 随机边界裁剪
    - `cut_ratio`: 裁剪比例
  - `GaussianNoise`: 高斯噪声
    - `mean`: 均值
    - `std`: 标准差
  - `GammaCorrection`: 伽马校正
    - `gamma`: 伽马值范围
  - `RandomSubsetApply`: 随机子集应用变换
    - `transforms`: 变换列表
    - `p`: 概率列表
    - `n_subset`: 子集大小
    - `random_order`: 是否随机顺序
  - `SharpnessJitter`: 锐度抖动
    - `sharpness`: 锐度范围
  - `ImageTransformConfig`: 图像变换配置
    - `weight`: 权重
    - `type`: 变换类型
    - `kwargs`: 关键字参数
  - `ImageTransformsConfig`: 图像变换集合配置
    - `enable`: 是否启用
    - `max_num_transforms`: 最大变换数
    - `random_order`: 是否随机顺序
    - `tfs`: 变换字典
  - `ImageTransforms`: 图像变换组合
    - `_cfg`: 配置
    - `weights`: 权重列表
    - `transforms`: 变换字典
    - `tf`: 最终变换
- **主要函数**:
  - `make_transform_from_config(cfg)`: 从配置创建变换

---

### lerobot_patches/ Lerobot补丁

#### `lerobot_patches/custom_patches.py`
- **作用**: Lerobot框架的自定义补丁，扩展支持RGB和深度图像
- **主要修改**:
  - 扩展`FeatureType`枚举，添加`RGB`和`DEPTH`类型
  - 修改`compute_episode_stats`函数，支持深度图像统计
  - 修改`dataset_to_policy_features`函数，正确映射深度特征
  - 修改`create_stats_buffers`函数，支持深度图像归一化
- **主要函数**:
  - `custom_sample_images(image_paths, sampled_indices)`: 自定义图像采样
  - `custom_sample_depth(data, sampled_indices)`: 自定义深度采样
  - `compute_episode_stats(episode_data, features)`: 计算episode统计
  - `dataset_to_policy_features(features)`: 数据集到策略特征映射
  - `create_stats_buffers(features, norm_map, stats)`: 创建统计缓冲区

---

## 关键数据流

### 数据转换流程
1. `CvtRosbag2Lerobot.py` → 读取rosbag文件
2. `KuavoRosbagReader.process_rosbag()` → 处理rosbag消息
3. `KuavoMsgProcesser` → 处理各类ROS消息
4. `align_frame_data()` → 对齐时间戳
5. `populate_dataset()` → 填充LeRobot数据集
6. 保存为parquet格式

### 训练流程
1. `train_policy.py` → 加载配置和数据集
2. `build_policy()` → 构建策略模型
3. `build_optimizer_and_scheduler()` → 构建优化器
4. 训练循环 → 前向传播、反向传播、更新参数
5. 保存检查点

### 部署流程
1. `eval_kuavo.py` → 加载模型和环境
2. `setup_policy()` → 设置策略
3. 评估循环 → 获取观测、选择动作、执行动作
4. 记录结果和视频

---

## 主要依赖关系

- **kuavo_data**: 依赖ROS、rosbag、cv2、numpy、torch
- **kuavo_train**: 依赖lerobot、torch、hydra、diffusers
- **kuavo_deploy**: 依赖gymnasium、kuavo_humanoid_sdk、rospy、torch
- **lerobot_patches**: 依赖lerobot内部模块

---

## 注意事项

1. **ROS依赖**: `kuavo_data`和`kuavo_deploy`需要ROS Noetic环境
2. **深度图像**: 项目支持RGB和深度图像，需要相应的相机配置
3. **末端执行器**: 支持三种类型：qiangnao（灵巧手）、leju_claw（夹爪）、rq2f85（仿真夹爪）
4. **控制模式**: 目前仅支持关节角度控制（joint），不支持末端执行器控制（eef）
5. **手臂选择**: 支持left、right、both三种模式
6. **补丁应用**: 必须在导入lerobot相关模块前导入`lerobot_patches.custom_patches`


