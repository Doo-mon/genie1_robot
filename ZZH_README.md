# G1 远程推理运行手册

本文描述当前工作区的 EE、joint、RTC 和 rollout 推理链路。默认部署方式是：机器人控制机只连接本机回环端口，SSH `-L` 把该端口转发到远端模型服务器。所有示例中的 `<REMOTE_USER>`、`<SSH_HOST>`、`<REMOTE_MODEL_PORT>`、`<POLICY_ARGS>` 都是占位符，必须替换为当前部署值。

## 1. 模块图与文件职责

```text
9 个 run_*.sh launcher
├── 6 个通用 launcher ───────────────┐
├── 1 个 5659 同步 EE preset ──> 通用 EE launcher
└── 2 个 EE rollout preset ─────> 通用 EE rollout launcher
                                      │
                                      v
6 个 desk Python 入口 + inference.py 兼容入口
├── policy_prompts.py       prompt 目录、key 解析、人工确认
├── inference_common.py     URI、超时/退避、WebSocket、图像与运行时工厂
│   ├── websockets.sync.client
│   └── openpi-client/.../msgpack_numpy.py
├── policy_ee.py            EE14 / 30D 数据合同与动作安全处理
├── policy_joint.py         joint16 数据合同与动作安全处理
├── policy_ee_rtc.py        EE 双线程 RTC、前缀硬锁、停止/取消
├── policy_joint_rtc.py     joint 双线程 RTC、前缀模拟/硬锁、停止/取消
├── action_logging.py       策略 I/O 与原始/有效/实际动作逐行日志、写盘耗时监测
├── plot_action_log.py      从动作 JSONL 离线生成逐维曲线和全维度总览 SVG
├── rollout_schemas.py      运行时观测/命令到数据集 state/action 的映射
├── rollout_recorder.py     三路视频、Parquet、LeRobot 风格 metadata
└── robot_env.py
    └── robot_interface_wrapper.py -> robot_interface.py -> a2d_sdk
```

主要文件如下：

| 文件 | 职责 |
|---|---|
| `inference_common.py` | 所有活跃入口共用的可靠同步 WebSocket 客户端；在没有安装包时把 vendored `openpi-client/src` 加入导入路径；延迟创建 `RobotEnv`。 |
| `policy_ee.py` | 构造 EE 请求、校验 `H x 14` 响应、位置/旋转单步限幅、夹爪二值化、可选冻结左 EE，以及同步 EE 的 SO(3) chunk 时间重采样。 |
| `policy_joint.py` | 构造 joint 请求、校验 `H x 16` 响应、关节单步限幅和夹爪二值化；当前模型与机器人顺序相同，不做旧式重排。 |
| `policy_ee_rtc.py` / `policy_joint_rtc.py` | 推理线程与执行线程并行；低水位补 chunk、按已经执行的步数对齐响应、硬锁已承诺前缀。 |
| `policy_prompts.py` | joint、EE、bread prompt 分组，named prompt，literal/key/菜单选择与最终人工确认。 |
| `rollout_schemas.py` | 定义 EE 和 joint 数据集维度、顺序，以及“实际下发命令”如何进入 action；同步 EE 与 RTC EE 使用不同 processing mode。 |
| `rollout_recorder.py` | 以独立帧率抓取三路 RGB；后台写 MP4、Parquet 和 `meta/*`；只有启用录制时才加载 OpenCV/PyArrow。 |
| `action_logging.py` | 为每个进程保存 run 配置、策略数值输入/原始响应，以及每条成功执行的原始服务端输出、有效目标和实际命令；同时监测同步写盘对控制周期的占用。 |
| `plot_action_log.py` | 纯标准库离线绘图；EE 生成 14 张、joint 生成 16 张逐维 SVG，并额外生成一张全维度总览；同一维度叠加原始输出、有效目标和实际命令。 |
| `inference_log_helpers.sh` | 六个通用 launcher 共用的时间戳目录、console 捕获、动作日志开关、退出绘图和退出码保留函数。 |
| `inference_pi05_g1_desk_{ee,joint}.py` | 同步 chunk 推理入口；EE 入口另支持可选的 chunk 重采样和同步 rollout。 |
| `inference_pi05_g1_desk_{ee,joint}_rtc.py` | RTC 执行但不录制。 |
| `inference_pi05_g1_desk_{ee,joint}_rollout_rtc.py` | RTC 执行并可录制 rollout。 |
| `inference.py` | 兼容 bread/旧 18D、20D 控制链路；也使用同一 WebSocket 工厂，但不要把它的维度定义混入 desk EE/joint 合同。 |
| `robot_env.py` | Gym 风格真实机器人封装；`reset()` 可能运动，`step()` 下发命令，`get_obs()` 读取三路图像和状态，并以统一锁串行化硬件访问。 |
| `robot_interface.py` | a2d_sdk 底层接口。构造时等待 head 图像、14D 手臂和双夹爪数据；等待过久只告警，不自动退出。 |
| `env_for_g1.sh` | 激活机器人 SDK/conda 环境并进入工作区；应使用 `source`。 |

活跃的 9 个 Shell launcher：

| launcher | 模式 | prompt profile / 默认 key | 默认本地端口 |
|---|---|---|---:|
| `run_inference_pi05_g1_desk_ee.sh` | EE 同步 | `ee` | 2379 |
| `run_inference_pi05_g1_desk_ee_5659_camera_h50.sh` | 5659 同步 EE + 重采样 + rollout preset | `named:5659-all` | 继承通用 EE |
| `run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm.sh` | 5659 同步 EE 单臂隔离测试；冻结指定手臂 | `named:5659-all` | 继承 5659 同步 preset |
| `run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh` | 5659 RTC + rollout 单臂隔离测试 | `named:5659-all` | 继承 5659 RTC preset |
| `run_inference_pi05_g1_desk_ee_rtc.sh` | EE RTC | `ee` | 2379 |
| `run_inference_pi05_g1_desk_ee_rollout_rtc.sh` | EE RTC + rollout | `ee` | 2379 |
| `run_inference_pi05_g1_desk_ee_3017_3040_h50_rollout_rtc.sh` | EE rollout preset | `named:bread-all`，`breads` 初始姿态 | 继承通用 EE |
| `run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh` | EE rollout preset | `named:5659-all` | 继承通用 EE |
| `run_inference_pi05_g1_desk_joint.sh` | joint 同步 | `joint` | 2379 |
| `run_inference_pi05_g1_desk_joint_rtc.sh` | joint RTC | `joint` | 2379 |
| `run_inference_pi05_g1_desk_joint_rollout_rtc.sh` | joint RTC + rollout | `joint` | 2379 |

三个 preset 不直接运行 Python；它们用命令级环境变量补充默认值，再 `exec` 对应的通用 EE 或 EE rollout launcher。调用者已经设置的同名变量优先，`"$@"` 最后继续传给 Python。5659 同步 preset 不启用 RTC，但默认开启同步 rollout 和 chunk 重采样。它当前以 30 Hz 选取 H50 全部模型行，然后展开为相邻位置/SO(3) 旋转不超过 `0.002 m / 0.01 rad`的可变长执行队列。夹爪仅在对应模型 segment 的最后一个插值步切换；`MAX_TIMESTEPS=1500` 统计实际下发的重采样命令。

### 1.1 每次运行的输入/动作日志与自动绘图

九个 launcher 最终都会经过同一套日志入口。每次启动立即在默认的
`./logs/` 下创建一个独立目录：

```text
logs/YYYYMMDD_HHMMSS_<python入口>_pid<PID>/
├── console.log          # 终端 stdout/stderr
├── run.json             # 模式、维度名、参数、开始/结束时间
├── policy_io.jsonl      # 数值输入、RTC prefix、prompt、原始模型响应
├── actions.jsonl        # 原始模型输出、有效目标与实际命令
└── plots/
    ├── overview.svg     # 所有维度汇总在同一画布，每个子图独立纵轴
    ├── overview.html    # 单独打开总览
    ├── index.html       # 顶部总览，下面浏览逐维大图
    └── 00_*.svg ...     # 每个维度一张独立曲线
```

EE 日志固定 14 维，joint 日志固定 16 维
`[left_arm7,left_hand,right_arm7,right_hand]`。`actions.jsonl` 只在
`env.step()` 成功（fake 模式则在等价执行点）后追加，所以未真正下发的目标不会
伪装成实际动作。逐维图和总览中的每个子图都包含原始服务端输出、完成
freeze/prefix hard-lock/时间对齐后的有效目标，以及实际命令三条曲线；总览的
各维度使用独立纵轴，避免夹爪、位置和旋转的量纲互相压缩。原始 action chunk
也完整保存在 `policy_io.jsonl`。

默认在 Python 正常退出、报错退出或 Ctrl-C 最终结束进程后自动读取已经 flush 的
`actions.jsonl` 绘图。日志没有任何成功动作时会明确跳过，不生成空图；绘图失败
也不会覆盖 Python 原退出码。开关和目录均用单次命令变量，不会注册到父终端：

```bash
# 默认自动绘图
PLOT_ACTIONS=true bash ./run_inference_pi05_g1_desk_joint_rtc.sh

# 只记日志，不在退出时绘图
PLOT_ACTIONS=false bash ./run_inference_pi05_g1_desk_ee_rtc.sh

# 完全关闭动作/策略 JSONL（仍保存 console.log，也自动跳过绘图）
ACTION_LOGGING=false bash ./run_inference_pi05_g1_desk_ee_rtc.sh

# 修改日志根目录
ACTION_LOG_ROOT=/data/g1_inference_logs \
  bash ./run_inference_pi05_g1_desk_joint_rollout_rtc.sh

# 之后也可手动重画任意一次日志
python ./plot_action_log.py ./logs/<本次时间戳目录>
```

直接运行 Python（不经过 Shell）时仍会自动创建日志；需要手动调用绘图脚本。
`SIGKILL`、掉电和系统崩溃无法执行 Shell EXIT trap，但此前已经完整 flush 的
JSONL 行仍可在恢复后手动绘制。

动作日志会测量每次 JSON 序列化、写盘和 flush 的实际耗时。记录至少 30 步后，
若平均耗时达到控制周期的10%，或单次峰值达到25%，终端会提示下次用
`ACTION_LOGGING=false` 关闭，避免日志明显侵占20/30Hz控制周期。

提示只是一项基于真机当前磁盘性能的建议，不会在运行中自动关闭日志或改变动作
执行。需要关闭时应先正常结束当前进程，再在下一次启动命令前设置开关。直接运行
Python 而不经过 Shell 时，对应开关是
`INFERENCE_ACTION_LOG_ENABLED=false`；Shell 推荐始终使用更短的
`ACTION_LOGGING=false`。

## 2. 数据合同与安全处理

### 2.1 EE：EE14 请求/响应与 30D 状态

机器人观测必须包含：

- `image/head`、`image/left_wrist`、`image/right_wrist`；
- `state/joint`：16D；
- `state/pose`：14D EE pose。

发给 EE 模型的 `observation/state` 为：

```text
[joint16, ee14] = 30D
```

EE14 的固定顺序为：

```text
[L_xyz3, L_axis_angle3, L_hand1,
 R_xyz3, R_axis_angle3, R_hand1]
```

夹爪在 state 与 action 中使用不同数值空间：机器人反馈 state 是 0--120 的编码器位置，模型 action 是连续 0--1；当前训练数据约定二者同方向变化，`0=张开、1/120=闭合`。模型响应 `actions` 必须是有限值 `H x 14`。下发前会按左右臂分别限制位置和轴角的单步变化。通用 Shell 默认开启夹爪二值化，并使用双阈值滞回：action 从张开状态上升到 `0.90` 才闭合，从闭合状态下降到 `0.10` 才张开，中间区间保持上次命令；这样不会像单阈值那样在中间值附近提前切换或抖动。设置 `BINARIZE_GRIPPER=false` 可原样保留连续 action。启用 `freeze_left_arm` 时，完整左侧 7D EE 目标固定为 episode 起点，其中夹爪 state 会先除以 120 转换到 action 空间，避免把 0--120 state 写进动作队列。

EE rollout 与线上命令维度不同：

```text
observation.state = [joint16_feedback, ee14_feedback] = 30D
action            = [joint16_feedback, executed_ee14] = 30D
```

真正发给 `RobotEnv.step()` 的仍是 14D。数据集 action 的前 16D 是同步关节反馈，训练侧 EE transform 会忽略它；后 14D 是经过安全处理、实际下发的命令。

### 2.2 joint：统一 joint16 顺序

机器人观测、模型输入/输出、RTC 队列、prefix、实际命令及 rollout 始终采用：

```text
[left_arm7, left_hand1, right_arm7, right_hand1] = 16D
```

夹爪索引固定为 7 和 15。当前 `robot`、`model/openpi` 两个名称表示同一个顺序，所有转换函数仅做严格维度、有限值检查和复制，不再重排索引。joint rollout 的 state/action 都是 16D，action 是限幅、夹爪二值化后实际下发的命令。

Joint RTC 必须使用直接固定频率命令，会明确拒绝 `use_smooth_trajectory=true`；Ruckig 平滑轨迹在一条目标内部会执行多个带 sleep 的插值点，只能用于同步 joint 入口，否则会破坏 RTC prefix 的固定时间线。

不要把旧脚本或旧 checkpoint 的其他顺序直接接入当前入口；必须先确认服务端 checkpoint 的 16D 合同一致。

## 3. 从远端模型到本机 SSH `-L` 的完整启动顺序

### 第 0 步：真机现场准备

先完成本文末尾的真机安全清单。远端服务和隧道可以提前启动，但在确认 prompt 之前不要允许人员进入机器人运动范围。

### 第 1 步：在远端模型机启动策略服务

在远端终端或受管理的 `tmux`/服务进程中，激活模型环境并运行当前 checkpoint 对应的 server 命令。例如 OpenPI 风格模板：

```bash
cd <REMOTE_OPENPI_WORKSPACE>
source <REMOTE_MODEL_ENV_ACTIVATION>

REMOTE_MODEL_PORT=<REMOTE_MODEL_PORT>
python scripts/serve_policy.py --port "${REMOTE_MODEL_PORT}" <POLICY_ARGS>
```

`<POLICY_ARGS>` 应替换成部署已经验证过的 config/checkpoint 参数，不要从本文猜测。确认日志显示模型加载完成或 WebSocket server 已监听。若服务端提供 `/healthz`，可在远端检查：

```bash
curl --fail --max-time 2 "http://127.0.0.1:${REMOTE_MODEL_PORT}/healthz"
```

健康检查不存在时，以服务端监听日志或 `ss -ltn` 为准。远端服务若与 SSH 登录主机在同一台机器，隧道目标使用远端的 `127.0.0.1`；若不在同一台机器，隧道目标必须改为 SSH 登录主机能够访问的模型地址。

### 第 2 步：在机器人控制机建立持久 SSH 隧道

另开一个终端，选择未被占用的本地端口。EE 与 joint 通用 launcher 当前都默认 2379；这只是本地约定，不要求等于远端模型端口。

```bash
LOCAL_PORT=2379                 # 也可改成其他未占用的本地端口
REMOTE_MODEL_PORT=<REMOTE_MODEL_PORT>

ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_MODEL_PORT}" \
  <REMOTE_USER>@<SSH_HOST>
```

说明：

- 本地只绑定 `127.0.0.1`，不要无意中把机器人策略端口暴露到局域网。
- `ExitOnForwardFailure=yes` 能发现本地端口占用，但不保证远端模型已经就绪。
- SSH 进程必须持续存活；公共 WebSocket 客户端能等待或重连，但不能替你重启已经退出的 SSH 进程。
- 使用隧道时，Python 的 `REMOTE_HOST` 应是本地 `127.0.0.1`，`REMOTE_PORT` 应是 `LOCAL_PORT`，不是远端 GPU 的 IP/端口。

可选检查：

```bash
lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN
curl --fail --max-time 2 "http://127.0.0.1:${LOCAL_PORT}/healthz"
```

第二条仅适用于服务端实现了 `/healthz`。即使健康检查失败，推理客户端也会打印具体的握手/metadata 重试日志。

### 第 3 步：在机器人控制机激活环境

新开运行终端：

```bash
cd <LOCAL_WORKSPACE>
source ./env_for_g1.sh
```

确认激活后仍位于本工作区。launcher 会设置其 Python 子进程需要的路径；不要同时运行另一个会控制同一台机器人的客户端。

### 第 4 步：先用假观测验证完整网络与模型链路

假观测会跳过 `RobotEnv` 和真实动作，但仍会连接远端模型、完成 metadata 握手并实际调用推理。

EE 同步示例：

```bash
REMOTE_HOST=127.0.0.1 \
REMOTE_PORT=2379 \
PROMPT_KEY='ee-5659:1' \
WS_SERVER_WAIT_TIMEOUT=120 \
WS_RESPONSE_TIMEOUT=600 \
bash ./run_inference_pi05_g1_desk_ee.sh \
  --use-fake-obs \
  --max-timesteps=5
```

joint RTC 示例：

```bash
REMOTE_HOST=127.0.0.1 \
REMOTE_PORT=2379 \
PROMPT_KEY='joint-desk:1' \
USE_FAKE_OBS=1 \
MAX_TIMESTEPS=10 \
bash ./run_inference_pi05_g1_desk_joint_rtc.sh
```

仍需在终端输入 `y` 或 `yes` 确认 prompt。先检查首个请求打印出的图像/state shape、服务端响应维度和 warmup 是否完成。

### 第 5 步：启动真机任务

去掉 `--use-fake-obs` 或令 `USE_FAKE_OBS=0`，保留已验证的 host、port、checkpoint、prompt 和安全限幅。例如：

```bash
REMOTE_HOST=127.0.0.1 \
REMOTE_PORT=2379 \
PROMPT_KEY='joint-desk:1' \
WS_SERVER_WAIT_TIMEOUT=120 \
WS_RECONNECT_WAIT_TIMEOUT=30 \
WS_INFERENCE_RETRIES=0 \
bash ./run_inference_pi05_g1_desk_joint_rtc.sh
```

EE 5659 rollout preset 示例：

```bash
REMOTE_HOST=127.0.0.1 \
REMOTE_PORT=2379 \
ROLLOUT_OUTPUT_DIR='./rollout_datasets/5659_verified_run' \
bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh
```

新增的 5659 纯同步 EE + rollout 示例（不启用 RTC）：

```bash
REMOTE_HOST=127.0.0.1 \
REMOTE_PORT=2379 \
ROLLOUT_OUTPUT_DIR='./rollout_datasets/5659_sync_verified_run' \
bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50.sh
```

该 preset 默认 `ROLLOUT_ENABLED=true`。只关闭 rollout，但保留动作日志和退出画图：

```bash
ROLLOUT_ENABLED=false \
  bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50.sh
```

该 preset 还默认 `RETIME_ACTION_CHUNK=true`。如需用同一入口对照旧的“每个模型 row 只限幅一次就丢弃”行为：

```bash
RETIME_ACTION_CHUNK=false \
  bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50.sh
```

单臂隔离测试中，`FREEZE_ARM` 表示需要固定的手臂；另一只手臂正常执行：

```bash
# 固定左臂，只测试右臂
FREEZE_ARM=left \
  bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm.sh

# 固定右臂，只测试左臂
FREEZE_ARM=right \
  bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm.sh
```

冻结首先覆盖所选手臂的完整 EE7（位置、轴角旋转和夹爪），随后在 IK 完成后再次把该臂发送给 SDK 的 6 维 `ABS_POSE` 和夹爪命令锁定到 episode reset 后的实测锚点。双层冻结可避免 IK 分支变化让测试中本应静止的手臂产生物理运动。未冻结手臂继续使用 5659 同步重采样和原有限幅；动作日志仍保留服务端原始双臂输出，因此可以对照模型是否在被冻结的一侧产生了异常目标。

同步入口只在执行队列为空时请求下一块，因此服务端推理期间会保持最后命令；插值只能平滑 chunk 内部，不能消除 chunk 之间的等待。需要连续执行时应使用对应的单臂 RTC 入口：

```bash
# 固定左臂，只让右臂走 RTC；改成 right 即只测试左臂
FREEZE_ARM=left \
  bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh
```

RTC 对服务端响应先应用同一侧 EE7 冻结，再生成 committed prefix 和 hard-lock，因此模型时间线、实际 EE 命令与硬件侧 `ABS_POSE` 冻结保持一致。RTC 不使用同步版的整块可变长度重采样：服务端 prefix 仍按原始模型 timestep 对齐，未冻结手臂继续使用每 tick 位置/旋转限幅。二值夹爪发生切换时是唯一例外：如果对应手臂因为限幅尚未到达该模型 row 的绝对 EE 位姿，RTC 会把这一行保留在队首，保持旧夹爪并逐 tick 继续靠近；到达终点的命令先单独下发并保留旧夹爪，经过一个完整控制周期后，下一 tick 才切换夹爪、弹出该行并增加模型时间线的 pop count。门控默认最多持续 2 秒，超时会保持旧夹爪并终止本 episode，而不会为了继续运行强制抓取或释放。

需要 literal prompt 时可设置 `PROMPT='完整文本'`。通用同步 EE launcher 的 rollout 默认关闭，可用 `ROLLOUT_ENABLED=true`开启；通用 RTC-rollout launcher 也可用 `ROLLOUT_ENABLED=0` 关闭。`ROLLOUT_FPS` 设置采集频率，`ROLLOUT_OUTPUT_DIR` 指定数据集根目录。

## 4. WebSocket metadata + infer 协议

### 4.1 连接和 metadata

客户端先规范化 URI，再执行：

```text
TCP/WebSocket connect
  -> 服务端发送第一条 binary frame
  -> msgpack_numpy.unpackb(frame)
  -> 保存为最近一次 server metadata
```

metadata 必须是可由 msgpack 解码的字典；调用方可通过 `get_server_metadata()` 读取最近一次成功握手的值。当前协议没有远端 `reset` 消息，客户端的 `reset()` 是兼容 no-op。

`remote_host` 可为裸主机、IPv4/IPv6，或 `ws://` / `wss://` URI。完整 URI 中已有端口时必须与 `remote_port` 相同；path/query 会保留；不允许嵌入用户名、密码或 fragment。SSH 转发场景优先使用裸 `127.0.0.1`。

### 4.2 一次 infer

```text
observation dict
  -> msgpack_numpy.Packer.pack（每个逻辑请求只打包一次）
  -> WebSocket binary send
  -> recv(timeout=WS_RESPONSE_TIMEOUT)
  -> binary msgpack dict
  -> policy_ee / policy_joint 校验 response["actions"]
  -> 安全处理
  -> RobotEnv.step
```

客户端用 I/O lock 保证同一连接上的 infer 严格串行。服务端可以返回额外字段，例如 timing；执行侧只读取 `actions`。

### 4.3 初连退避、总等待和超时

首次连接发生在真实 `RobotEnv` 创建之前。一次尝试包括 WebSocket 握手和第一帧 metadata：

1. 握手最多等待 `WS_OPEN_TIMEOUT`；若总等待预算更少，则使用剩余预算。
2. metadata 最多等待 `WS_RESPONSE_TIMEOUT`；同样不会超过有限的总等待剩余值。
3. 可恢复传输错误会关闭失败 socket：连接拒绝/重置、超时、EOF、运行中关闭、SSH 远端端口未监听造成的无效握手消息，以及 HTTP 408/425/429/5xx。
4. 等待 `WS_RETRY_INITIAL_DELAY`，随后每次翻倍，封顶 `WS_RETRY_MAX_DELAY`。
5. 累计达到 `WS_SERVER_WAIT_TIMEOUT` 后抛出 `PolicyConnectionError`；值为 0 时不设累计上限。

非法 URI、TLS 证书/代理/认证错误、大多数 HTTP 4xx、协议协商错误、文本 metadata、无法解码或解码后不是字典的 metadata 会立即退出，不进入传输重试。`WS_SERVER_WAIT_TIMEOUT` / `WS_RECONNECT_WAIT_TIMEOUT` 约束连接、metadata 和退避；失败 socket 的 graceful close 最多还可额外占用 `WS_CLOSE_TIMEOUT`，因此它们是重试预算而不是包含清理时间的严格墙钟 deadline。

### 4.4 运行中重连与重发

`infer()` 遇到超时、连接关闭、EOF 或其他被识别为传输错误时，可以按配置执行以下恢复流程：

1. 关闭旧 socket 并清空当前连接。
2. 在 `WS_RECONNECT_WAIT_TIMEOUT` 预算内重连。
3. 重新接收并保存 metadata。
4. 重发同一份已经打包的请求 bytes。
5. 最多重发 `WS_INFERENCE_RETRIES` 次，即总发送尝试最多为 `1 + WS_INFERENCE_RETRIES`。

默认 `WS_INFERENCE_RETRIES=0`，所以请求一旦发送后发生传输错误，会立即终止当前 episode，不会自动重发；首次连接时等待隧道和模型就绪的机制不受影响。只有显式把该值设为正数时，才启用 **at-least-once 的策略推理语义**：如果服务端已经处理了第一次请求、但响应在途中丢失，服务端可能再次计算同一请求。策略请求本身不会直接驱动机器人，客户端只执行最终收到的一份响应；但 RTC 时间线在重连期间仍可能前进，重发使用的观测/prefix 是原请求快照。

不要在一个仍运行的客户端背后把隧道切到不同 checkpoint。RTC warmup 每个客户端进程通常只做一次；如果远端 server 重启或换模型，应退出并重启本地客户端，使 no-prefix 和 prefix 两个变体重新 warmup。

### 4.5 不重试的错误边界

- 推理响应是文本：视为服务端 traceback，关闭连接并立即抛 `RuntimeError`，不重连、不重发。
- metadata 是文本、非法 msgpack 或不是字典：抛 `PolicyProtocolError`，不重试。
- 推理响应不是合法 msgpack 字典：关闭连接并抛 `PolicyProtocolError`，不重试。
- URI/config 错误或并发 I/O `RuntimeError`：视为配置/生命周期 bug，不用重连掩盖。
- TLS 证书、代理、认证/大多数 4xx、无效 header 或协议协商错误：视为永久配置错误，立即失败；HTTP 408/425/429/5xx 可重试。
- 本地 observation 打包、动作 shape、NaN/Inf、安全校验错误：属于客户端数据错误，不是网络抖动。

当前连接禁用压缩并允许任意大小的入站 WebSocket message；这适合图像/动作数据，但也意味着服务端和 SSH 隧道必须位于可信边界。

## 5. 八个 `WS_*` 参数及 `0` 语义

除重试次数外，单位均为秒。Shell 环境变量被转换为对应的 Tyro CLI 参数和 `WebsocketPolicyConfig` 字段。

| 环境变量 / CLI | 默认值 | 约束 | `0` 的精确语义 |
|---|---:|---|---|
| `WS_OPEN_TIMEOUT` / `--ws-open-timeout` | 10 | 有限且 `> 0` | 非法；不能用 0 关闭握手超时。 |
| `WS_CLOSE_TIMEOUT` / `--ws-close-timeout` | 10 | 有限且 `> 0` | 非法；不能用 0 跳过 close。 |
| `WS_RESPONSE_TIMEOUT` / `--ws-response-timeout` | 300 | 有限且 `>= 0` | 不为 metadata/infer 单独设接收超时；metadata 仍受有限的 server/reconnect 总预算约束。 |
| `WS_RETRY_INITIAL_DELAY` / `--ws-retry-initial-delay` | 1 | 有限且 `> 0` | 非法。 |
| `WS_RETRY_MAX_DELAY` / `--ws-retry-max-delay` | 10 | 有限、`> 0` 且不小于 initial | 非法。 |
| `WS_SERVER_WAIT_TIMEOUT` / `--ws-server-wait-timeout` | 0 | 有限且 `>= 0` | 初次启动等待不设累计上限；程序可一直等隧道/模型，但此时尚未创建真实 `RobotEnv`。 |
| `WS_RECONNECT_WAIT_TIMEOUT` / `--ws-reconnect-wait-timeout` | 30 | 有限且 `>= 0` | 每次运行中重连不设累计上限；RTC 队列耗尽后只会保持最后命令，必须由现场人员监督。 |
| `WS_INFERENCE_RETRIES` / `--ws-inference-retries` | 0 | 非负整数 | 默认行为：不重连重发原 infer；首次发送失败就抛错。 |

真机运行通常应给 server wait、response 和 reconnect 设置有限值。时间参数的 `0` 表示不设累计上限或关闭对应接收超时；重试次数的 `0` 则表示禁止重发，二者不要混淆。

`WS_RESPONSE_TIMEOUT` 从 `recv()` 开始计时；`websockets` 的同步 `send()` 没有独立 timeout 参数。正常本地隧道下请求发送很短，但极端 TCP 背压仍可能先阻塞在 send；SSH keepalive、有限的外层任务监督和现场 Ctrl-C 仍是必要的最后保护。一次逻辑 infer 的最长时间还会随 `1 + WS_INFERENCE_RETRIES` 次响应等待及中间重连累计，不能把 300 秒误解为整次 infer 的总墙钟上限。

## 6. 同步、RTC 与 rollout 的差异

| 模式 | 推理/执行关系 | chunk 使用 | 网络中断时的运行表现 | 录制 |
|---|---|---|---|---|
| 同步 | 同一线程先 infer，再执行 | 可用 `chunk_skip` 丢开头、用 `execution_horizon` 选窗口 | 等待 infer 时没有新的命令下发；失败传播到 episode/入口 | EE 可选；5659 同步 preset 默认开启 |
| RTC | 推理线程和固定频率执行线程并行 | `chunk_skip=0`；队列低于 `rtc_refill_at` 时补充；锁定最多 `rtc_prefix_len` 个承诺动作 | 仍消费已有队列；队列空时不再下发新目标，底层保持最后命令；结束时关闭阻塞 socket | 无 |
| RTC + rollout | 与 RTC 完全相同 | 与 RTC 完全相同 | 与 RTC 完全相同 | 独立采样线程按 `rollout_fps` 记录三路 RGB、反馈和实际命令 |

### 6.1 同步 EE chunk 重采样

该功能只属于 `inference_pi05_g1_desk_ee.py`，不会修改 EE/joint RTC 时间线，也不要把相同开关加到 RTC launcher。开启后，客户端按以下顺序处理：

```text
服务端绝对 EE14 chunk
-> freeze（如启用）
-> chunk_skip / execution_horizon 选取模型 row
-> 首个 chunk 从 reset 后 feedback 起步；后续从上一条成功下发 command 衔接
-> 按 control_frequency 逐条下发
```

每个原始 segment 的子步数取左右手位置距离和 SO(3) 相对旋转所需步数的最大值，因此双臂仍处于同一 segment 相位。位置线性插值，旋转使用 quaternion SLERP 最短路径；夹爪在前面插入的子步中保持旧值，仅在该 segment endpoint 切换。

`execution_horizon` 统计采用的模型 row；`max_timesteps` 统计重采样后成功下发的 command。单 chunk 超过 `MAX_RETIMED_CHUNK_STEPS` 会在第一条真机命令之前整体拒绝，不会截断 endpoint 或夹爪事件。终端每个 chunk 打印 `model_rows -> commands`、最大 segment 展开倍数和预计执行时间；展开过大时应减少 `EXECUTION_HORIZON`或重新评估限幅，不应只把安全上限抬高。

重采样模式不会在 chunk 边界改用滞后的反馈重置 command 基线，因此这些上限连续约束相邻成功下发 EE target command（包括两个 chunk 之间）；`actions.jsonl` 中的 `actual_action` 也是这个命令，不是传感器实测位移。策略请求仍使用每次新读取的观测。真实机械臂还受 SDK 轨迹控制、`TRAJECTORY_TIME`、跟踪误差和反馈频率影响；rollout state 才是观测反馈。

RTC 第一个请求没有 prefix。后续请求携带：

```text
remaining_action_chunk
prefix_len
```

响应返回时，运行时先在完整响应上把承诺前缀硬锁回原命令，再丢弃推理期间已经执行所对应的行，避免近似模型输出改变已经承诺的轨迹。EE 与 joint 都会从最近实际下发命令开始，顺序模拟位置/旋转或关节限幅及夹爪处理后的真实承诺命令，再把该命令序列发送给服务端并用于硬锁；EE 模拟遇到需要等待手臂的夹爪切换行时，会在该行之前截断 committed prefix，因为这一行尚不再满足“一模型 row 对应一物理 tick”。门控过程中虽然手臂命令继续推进，但 `_pop_count` 不增加；timeline revision 会拒绝恰好跨越门控返回的旧响应，门控结束后再用新观测和剩余队列发起请求。`rtc_prefix_space=openpi` 和 `robot` 目前都是统一 joint16 顺序，仅为旧启动配置保留名称。EE 的 prefix 夹爪固定使用模型 action 的 `0-1` 空间；`0-120` 只属于机器人反馈 state，不能作为服务端 RTC action prefix。该合同不再通过 Shell 或 CLI 暴露切换参数。原始目标队列保留模型输出；committed prefix、hard-lock、机器人命令和 rollout action 也始终处于 0--1 动作空间。通用 Shell 默认用滞回阈值把实际承诺/执行命令处理成端点，关闭二值化后则保持连续值。EE 或 joint 响应若晚到且执行进度已经超过本次 `prefix_len`，运行时不会接入这段缺少完整承诺历史的结果，而是保留当前队列并发起新请求。

两个 H50 EE rollout preset 当前默认使用 `RTC_PREFIX_LEN=34`、
`RTC_REFILL_AT=40`。在 30 Hz 下，34 步约为 1.13 秒，用于覆盖本机一次真机
测试中约 1.09 秒的最大推理延迟；旧的 8 步仅约 0.27 秒，会让多数响应越过
承诺前缀而被拒绝。该默认只属于 H50 preset，不能照搬到默认 H20 的通用入口。
34/40 能显著减少迟到拒绝，但连续多次接近最坏延迟时 H50 仍可能短暂耗尽；
应继续观察 `infer_ms`、handoff reject 和 queue starvation，而不是把该配置理解为
无条件的实时保证。

Shell 示例（均为单次命令变量，不会写入父终端环境）：

```bash
# 默认：prefix 使用0--1；夹爪以 close=0.90/open=0.10 双阈值切换
bash run_inference_pi05_g1_desk_ee_rtc.sh

# 更晚闭合、也更晚张开（1=闭合，0=张开）
GRIPPER_CLOSE_THRESHOLD=0.95 \
GRIPPER_OPEN_THRESHOLD=0.05 \
  bash run_inference_pi05_g1_desk_ee_rtc.sh

# 如需执行模型的连续夹爪 action，不做阈值二值化
BINARIZE_GRIPPER=false \
  bash run_inference_pi05_g1_desk_ee_rtc.sh

# 兼容旧命令：显式设置后退回无记忆单阈值，并优先于上面的双阈值
GRIPPER_THRESHOLD=0.98 \
  bash run_inference_pi05_g1_desk_ee_rtc.sh

```

首次被确认的 RTC task 会在 runner 启动前依次 warmup 无 prefix 和有 prefix 两个请求；同一进程后续 task 不重复 warmup。EE 与 joint 都会在首次 warmup 后重新读取当前观测，用它建立 last-command（EE 还会建立 freeze anchor），避免长时间编译期间姿态变化造成基准陈旧。RTC rollout 的 `EpisodeRecorder` 在确认、reset、warmup 和这次观测刷新后才由 `begin_episode()` 创建；同步 EE rollout 没有 warmup，在 prompt 确认和 reset 后创建。两者都在首个实际命令下发成功后才开始形成有效 action 帧。

录制帧率可以不同于控制频率。writer 队列满时会丢录制帧而不是阻塞控制线程，并在日志/metadata 中记录 dropped frames；这不表示控制动作被丢弃。真机录制由 `RobotEnv` 在同一硬件锁内返回观测与最近一条成功下发的命令，避免 state/action 跨控制步错配。开始录制前还会校验已有数据集的处理模式、state/action 维度、视频格式与 FPS，不符时在写入前失败。

`RobotEnv` 用统一 `RLock` 串行化 SDK 与 retargeter 访问；RTC 另用 observation gate 把“观测+队列快照”与“出队+真机下发”保持在同一物理时序。网络推理和 rollout callback 不持有这些锁。这样不再依赖底层 SDK 自身的并发安全性，但图像/状态读取过慢仍可能造成 control overrun，真机时应观察该日志，并严格保证没有第二个客户端操作同一机器人。

## 7. Prompt 覆盖、选择与确认时序

覆盖优先级：

```text
literal PROMPT / --prompt
  > PROMPT_KEY / --prompt-key
  > 交互菜单
```

Shell 中 `PROMPT` 为空时兼容读取 `FIXED_PROMPT`。Python 层只要同时收到 literal 和 key，literal 始终优先；若环境中已有 `PROMPT`，仅在命令尾追加 `--prompt-key` 不能压过它，应先清空 `PROMPT` 或明确传新的 `--prompt`。

Key 形式：

- 裸数字：profile 主分组中的编号；joint 主分组是 `joint-desk`，EE 主分组是 `ee-5659`，bread 主分组是 `bread`。
- `group:key`：例如 `joint-dressing-table:15`、`ee-legacy-desk:2`。
- `named:key`：例如 `named:bread-all`、`named:5659-all`、`named:tidy-desk`。
- 不像 key 的普通文本：作为自定义 literal prompt。

无论 prompt 来自 Shell、CLI、key 还是现场输入，都会打印完整候选并询问：

```text
Use this prompt and start execution?
```

只有 `y` / `yes` 放行；`q` 或 EOF/Ctrl-C 退出；其他回答拒绝候选并返回完整菜单。

当前真实时序是：

```text
参数验证
-> 等待并连接策略服务器
-> 创建 RobotEnv（fake 模式跳过；此时尚未 reset）
-> 选择并确认 prompt
-> RobotEnv.reset（从这里开始可能运动）
-> RTC 首次 warmup（同步模式没有显式 warmup）
-> EE / joint 首次 warmup 后重新读取当前观测
-> rollout begin_episode（同步 EE 或 RTC 显式开启时）
-> runner / 控制循环
-> rollout end_episode
```

因此 prompt 确认严格发生在 reset、warmup、RTC worker 和实际 recorder 启动之前，但发生在网络连接和 `RobotEnv` 构造之后。

### 7.1 RobotEnv 构造阶段的硬件就绪告警

真实模式创建 `RobotEnv` 时，底层默认认为 SDK 的状态接口符合当前合同，并等待：

```text
head_camera + arm_joint_states + gripper_states
```

正常情况下三项会很快到达，不产生额外输出。若等待超过 5 秒，终端打印第一次
`[RobotNode] warning`，列出仍为 `None` 的通道；之后每 10 秒重复一次。该告警
不会自动终止初始化，也不会绕过缺失数据开始执行，程序仍继续等待。现场确认只是
模型或 DDS 启动较慢时可以继续观察；确认相机、DDS 或 SDK 异常时按 Ctrl-C 退出，
修复后重新启动。不要因为 warning 而直接进入 reset 或手工下发动作。

当前代码按已确认的 SDK 合同使用 14D 手臂序列和双夹爪序列，并组合成
`[L7,Lhand,R7,Rhand]` 16D，不额外猜测或重排 SDK 返回格式。

## 8. 正常关闭顺序

推荐顺序：

1. 在推理终端按 Ctrl-C；RTC 会设置 stop，反复用 `cancel_pending_inference()` 打断正在阻塞或恰好启动的 socket 请求，再等待执行与推理线程退出；stop 会阻止新的命令出队，但在 stop 取得 timeline 锁前已经选中的一条命令仍可能在 observation gate 保护下完成 `env.step()` 和随后的 callback，因此 Ctrl-C 不是硬件急停。线程未在 join 时限内退出会令本 episode 失败，不能进入下一轮。
2. 若入口询问是否继续 task，选择 `n` 退出进程。
3. rollout 入口先 `end_episode()`：原子关闭新帧入队闸门，再停止采样、清空 writer 队列、释放视频 writer、写 Parquet/metadata；即使此步报错，外层 `finally` 仍会继续关闭 policy 和 RobotEnv。
4. 入口调用 `policy.close()`：永久关闭 WebSocket，并唤醒所有退避等待。
5. 最后调用 `env.close()` 关闭机器人接口。
6. 确认本地 Python 已退出后，再终止 SSH 隧道。
7. 确认没有其他客户端需要服务后，最后停止远端模型 server。

不要先杀 SSH 隧道再处理机器人进程；默认会让当前 episode 因连接错误退出，显式启用重发时还会触发重连。不要用 `kill -9` 作为常规关闭方式，否则 rollout 可能缺少尾部视频、Parquet 或 metadata。

## 9. 四类排障

### A. 一直显示 `Policy server not ready` / 初连耗尽

- 确认 SSH 进程仍在，`lsof` 显示本地 `REMOTE_PORT` 被 ssh 监听。
- 检查本地端口是否与 launcher 的 `REMOTE_PORT` 一致；EE 与 joint 当前都默认 2379。
- 检查 `-L` 右侧的远端目标和 `<REMOTE_MODEL_PORT>`，不要把本地端口误当远端端口。
- 在远端确认模型进程未退出、端口已监听、checkpoint 仍在加载或已经就绪。
- 若日志每次在约 `WS_OPEN_TIMEOUT` 后重试，通常是握手/TCP 不通；若本地 SSH 进程已退出，先重建隧道。
- 调试时把 `WS_SERVER_WAIT_TIMEOUT` 设为有限值，避免错误配置无限等待。

### B. WebSocket 已接通，但 metadata 超时或协议错误

- 确认转发目标真的是 OpenPI WebSocket policy server，而不是普通 HTTP 服务、health 端口或另一个模型协议。
- 第一条 WebSocket frame 必须是可解码为字典的 binary msgpack metadata；文本、非法 msgpack 或非字典值不会重试。
- metadata 超时受 `WS_RESPONSE_TIMEOUT` 及有限的 server/reconnect 总预算共同约束。
- 检查客户端/服务端 `msgpack_numpy` 合同和依赖版本；不要把 JSON server 接到该端口。
- 完整 `ws://.../path` URI 可以使用，但 embedded port 必须与 `REMOTE_PORT` 相同。

### C. 运行中出现超时、文本错误或显式启用后的 reconnect/replaying

- 只有 `WS_INFERENCE_RETRIES>0` 时才会出现 `reconnecting and replaying request`；它表示客户端正在重新握手、读 metadata 并重发同一 payload。
- 首次 JAX 编译超过 300 秒时可适当提高 `WS_RESPONSE_TIMEOUT`；不要直接用 0 掩盖未知挂死。
- 默认不重发已经发送的请求；只有显式令 `WS_INFERENCE_RETRIES>0` 时才采用 at-least-once 语义。
- 文本响应是服务端 traceback，客户端故意不重试。先阅读完整 traceback、修复模型/输入，再重启本地进程。
- 服务端重启或更换 checkpoint 后，即使显式启用的重连成功，也应主动重启客户端以重新执行 RTC warmup。

### D. 网络正常，但机器人、RTC 或 rollout 异常

- 持续显示 `hardware initialization is still waiting`：按 warning 中的
  `missing=` 判断是 head 相机、手臂 DDS 还是夹爪 DDS 未就绪。程序会继续等待，
  不会超时自动放行；确认不是正常慢启动后按 Ctrl-C，检查 SDK/DDS/相机再重启。
- shape/NaN 错误：核对 EE 的 `[joint16, ee14]` / EE14，或 joint 的 `[L7,Lhand,R7,Rhand]`，以及三路相机 key。
- RTC 显示 queue starved：系统在保持最后命令，不代表已经回到安全姿态；检查推理延迟、horizon、`rtc_refill_at` 和网络日志。
- EE/joint RTC 显示 `Rejecting ... RTC response beyond committed prefix`：推理耗时已经超过本次锁定前缀，结果为避免错接时间线而被丢弃；若持续出现，应检查推理/网络延迟，并联合调整 `rtc_prefix_len`、`rtc_refill_at` 与 horizon。
- control overrun：降低控制/rollout 压力，检查 `trajectory_time`、相机读取和 CPU；不要通过放宽动作限幅来解决性能问题。若同时出现 action-log 耗时 warning，下次运行先用 `ACTION_LOGGING=false` 对照验证。
- rollout 缺依赖：安装 OpenCV/PyArrow，或用 `--no-rollout-enabled`；检查输出盘空间、视频 codec 和 dropped/capture error 日志。
- 数据集 state/action 维度不匹配时不要继续追加到已有目录；EE/joint 或不同 processing mode 不能混写同一个 rollout root。

## 10. 真机执行前安全检查

- [ ] 急停可触达且已验证；有一名现场操作者持续观察，不进行无人值守真机运行。
- [ ] 机器人运动范围清空，人员、线缆、桌面物体和相机不会被双臂/夹爪碰撞。
- [ ] 只有一个控制客户端；没有遗留 Python、RTC worker、旧 SSH 隧道或其他上位机同时下发。
- [ ] 已确认连接的是正确 checkpoint 和模式：EE server 不接 joint 入口，joint server 不接 EE 入口。
- [ ] 已用 `--use-fake-obs` 完成网络、metadata、请求 shape、响应维度和两种 RTC warmup 检查。
- [ ] 三路图像方向和语义正确：head、left wrist、right wrist 未交换、冻结或严重延迟。
- [ ] 真实模式构造期间没有持续硬件就绪 warning；若出现，已根据 `missing=` 排查对应相机或 DDS，而不是无条件继续等待。
- [ ] joint16 顺序明确为 `[L7,Lhand,R7,Rhand]`，夹爪索引为 7/15；没有外部旧重排。
- [ ] EE14 顺序、轴角单位、坐标系、相机外参与 retargeter 标定均与 checkpoint/机器人一致。
- [ ] 从较小的 `MAX_TIMESTEPS`、`EXECUTION_HORIZON` 和保守的 joint/位置/旋转 delta 开始。
- [ ] `ACTION_HORIZON`、`RTC_PREFIX_LEN`、`RTC_REFILL_AT` 满足 `1 <= prefix <= refill < horizon`，RTC 的 `CHUNK_SKIP=0`。
- [ ] 保持默认 `WS_INFERENCE_RETRIES=0`；仅在确认接受请求可能被重复计算后才调大。
- [ ] 真机使用有限的 response 等待；若显式启用重发，再限制 reconnect 等待。出现 queue starvation、隧道断开或异常动作时立即停止。
- [ ] 已逐字阅读终端打印的 prompt；只有场景、左右臂、物体和目标位置都正确时才输入 `y/yes`。
- [ ] 已知确认之后 `RobotEnv.reset()` 就可能立即运动，确认前让机器人周围人员撤离。
- [ ] rollout 模式已确认目录、磁盘空间、fps 和依赖；录制故障不能阻塞或替代真机安全观察。
- [ ] 已观察当前磁盘下的 action-log 耗时；若程序建议关闭，下一次时序敏感运行使用 `ACTION_LOGGING=false`，同时保留 `console.log` 和 RTC overrun 观察。
- [ ] 已规划关闭顺序：Ctrl-C -> 结束 episode/录制 -> WebSocket -> RobotEnv -> SSH 隧道 -> 远端 server。
