# Astra + H45 双阶段机器人系统

Astra 使用图像、状态和历史示范，输出无人机或 Go2 的结构化原子动作；到达合适位置后，流程拟交给 Pi0.5 和 Piper 完成抓取。模型输出与实机执行结果应分别记录。仓库包含当前研究原型，不能把离线测试视为实机成功抓取。

## 文件导航

- `outputs/wam_astra/hybrid_live.py`：双阶段入口。`plan` 只生成规划；`execute` 需要已配置的真实设备；`replay` 读取记录并调用模型。
- `outputs/wam_astra/src/wam_astra/atomic_planner.py`：Astra 会话及结构化规划。`atomic_motion.py`：动作到 H45 轨迹的转换；`atomic_feedback.py`：执行反馈；`atomic_handoff.py`：Pi 交接依据。
- `outputs/wam_astra/src/wam_astra/` 还包含示范库构建、检索、任务记忆与动作审查相关代码。独立的任务记忆尚未作为当前双阶段规划的跨会话记忆接入。
- `outputs/wam_astra/h45_takeover/staged/client/`：H45 客户端和动作流；`staged/dog/`：Go2/Piper 侧接口及 Go2 SDK 桥源码；`staged/p450/`：P450 侧接口源码。
- `outputs/H45_live_repro_20261107/real_robot/`：原 H45 客户端；`openpi_client/`：Pi 通信依赖，保留 Apache 2.0 许可证。
- `outputs/wam_astra/two_stage_reference_20261111/input.json`：当前 Astra 参考上下文；引用的六张历史示范图位于 `navigation_memory_20261111/` 与 `h45_takeover/changing_atomic_navigation_20261111/`。历史示范不是当前观测，也不代表抓取成功。
- `outputs/wam_astra/tests/`：测试；`work/benchmark_*`：时延测试脚本；`UPSTREAM_LOCK.json`：上游仓库和版本来源。
- `client.atomic.example.json`：客户端配置模板，仅含占位符。真实主机地址、密钥、token 以及现场设备配置由使用者本地配置。

## 环境准备

Windows：运行 `./setup-windows.ps1`。WSL Ubuntu 中运行 `bash setup-wsl.sh`，从 GPT-Policy 官方仓库获取固定版本并安装必要依赖。模型调用还需要已登录且具备相应访问权限的 Codex 环境。将 `outputs/wam_astra/h45_takeover/atomic_deployment_20261111/client.atomic.example.json` 复制为同目录的 `client.atomic.json`，填入本地配置。

Pi0.5 checkpoint、Pi 模型服务、设备侧 SDK/ROS 环境、CAN 与网络、设备私钥和 token 均不包含在仓库中，需要现场单独部署。示范库全部 32 段结构化轨迹暂未上传；本仓库仅含上述六张参考图和参考上下文。执行实机模式前必须按设备环境配置，仓库克隆本身不保证实际机器人或模型服务已就绪。

## 验证

在仓库根目录使用已准备好的 Python 环境运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q outputs/wam_astra/tests/test_atomic_motion.py outputs/wam_astra/tests/test_atomic_handoff.py outputs/wam_astra/tests/test_atomic_feedback.py outputs/wam_astra/tests/test_astra_runtime.py outputs/wam_astra/tests/test_hybrid_native_loop.py outputs/wam_astra/tests/test_two_stage_entry.py outputs/wam_astra/tests/test_atomic_planner_input.py
```

完整现场链路仍需要可用的飞机、狗、Piper 和 Pi 服务。
