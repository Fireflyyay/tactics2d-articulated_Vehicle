![Tactics2D LOGO](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/Tactics_LOGO_long.jpg)

# Tactics2D: A Reinforcement Learning Environment Library with Generative Scenarios for Driving Decision-making

[![Codacy](https://app.codacy.com/project/badge/Grade/2bb48186b56d4e3ab963121a5923d6b5)](https://app.codacy.com/gh/WoodOxen/tactics2d/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codecov](https://codecov.io/gh/WoodOxen/tactics2d/graph/badge.svg?token=X81Z6AOIMV)](https://codecov.io/gh/WoodOxen/tactics2d)
![Test Modules](https://github.com/WoodOxen/tactics2d/actions/workflows/test_modules.yml/badge.svg?)
[![Read the Docs](https://img.shields.io/readthedocs/tactics2d)](https://tactics2d.readthedocs.io/en/latest/)

[![Downloads](https://img.shields.io/pypi/dm/tactics2d)](https://pypi.org/project/tactics2d/)
[![Discord](https://img.shields.io/discord/1209363816912126003)](https://discordapp.com/widget?id=1209363816912126003&theme=system)

![python-version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Github license](https://img.shields.io/github/license/WoodOxen/tactics2d)](https://github.com/WoodOxen/tactics2d/blob/dev/LICENSE)

[EN](README.md) | CN

## 关于

> [!note]
> 这个仓库为上海交通大学研究生课程AU7043提供了支持。
>
> **请各位同学切换到AU7043分支。使用`git clone`指令安装Tactics2D。在上课期间，仓库会实时更新！**

`tactics2d` 是一个开源的 Python 库，专为自动驾驶中的强化学习决策模型开发与评估提供多样且具有挑战性的交通场景。tactics2d 具备以下核心特性：

- **兼容性**
  - 📦 轨迹数据集：支持无缝导入多种真实世界的轨迹数据集，包括 Argoverse、Dragon Lake Parking (DLP)、INTERACTION、LevelX 系列（HighD、InD、RounD、ExiD）、NuPlan 以及 Waymo Open Motion Dataset (WOMD)，涵盖轨迹解析和地图信息。*欢迎大家通过Issue提出对其他数据集的解析需求*。
  - 📄 地图格式：支持解析和转换常用的开放地图格式，如 OpenDRIVE、Lanelet2 风格的 OpenStreetMap (OSM)，以及 SUMO roadnet。
- **可定制性**
  - 🚗 交通参与者：支持创建新的交通参与者类别，可自定义物理属性、动力学/运动学模型及行为模型。
  - 🚧 道路元素：支持定义新的道路元素，重点支持各类交通规则相关设置。
- **多样性**
  - 🛣️ 交通场景：内置大量遵循 `gym` 架构的交通场景仿真环境，包括高速公路、并线、无信号/有信号路口、环形交叉口、停车场以及赛车道等。
  - 🚲 交通参与者：提供多种内置交通参与者，具备真实的物理参数，详细说明可参考[此处](https://tactics2d.readthedocs.io/en/latest/api/participant/#templates-for-traffic-participants)。
  - 📷 传感器：提供鸟瞰图（BEV）语义分割 RGB 图像和单线激光雷达点云作为模型输入。
- **可视化**：提供用户友好的可视化工具，可实时渲染交通场景及参与者，并支持录制与回放交通过程。
- **可靠性**：超过 85% 的代码已被单元测试与集成测试覆盖，保障系统稳定性与可用性。

如需了解 `tactics2d` 的更多信息，请参考[文档](https://tactics2d.readthedocs.io/en/latest/)。

## 社区与支持

- [Discord 频道](https://discord.gg/bJ5yHT3bcd)
- [Github Issues](https://github.com/WoodOxen/tactics2d/issues)
- QQ群：929488317

## 安装

### 0. 系统要求

我们在以下系统版本和Python版本上进行了测试：

| System | 3.8 | 3.9 | 3.10 | 3.11 |
| --- | --- | --- | --- | --- |
| Ubuntu 18.04 | :white_check_mark: | - | - | - |
| Ubuntu 20.04 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Ubuntu 22.04 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Windows | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| MacOS | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

### 1. 安装

强烈推荐大家使用环境管理工具 `conda` 或 `virtualenv` 来创建独立的 Python 环境，以避免依赖冲突。如果你还没有安装 `conda`，请参考[官方文档](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)进行安装。

```bash
# 创建一个新的conda环境
conda create -n tactics2d python=3.9
conda activate tactics2d
```

#### 1.1 通过 PyPI 安装

如果你只是想使用稳定版本，可以通过 `pip` 安装：

```bash
pip install tactics2d
```

#### 1.2 通过源码安装

如果你想要尝试最新的功能，可以通过源码安装。自从 v0.1.7之后，你需要先安装GCC才能编译：

```bash
# 路径中不包含数据集，请根据需要自行下载并建立软链接
git clone --recurse-submodules git@github.com:WoodOxen/tactics2d.git
cd tactics2d
pip install -v .
```

### 2. 准备数据集

根据开源协议，`tactics2d`不会分发任何数据集。你可以通过以下方式获取数据集：

- [Argoverse 2](https://www.argoverse.org/av2.html)
- [Dragon Lake Parking (DLP)](https://sites.google.com/berkeley.edu/dlp-dataset)
- [HighD](https://www.highd-dataset.com/)
- [InD](https://www.ind-dataset.com/)
- [RounD](https://www.round-dataset.com/)
- [ExiD](https://www.exid-dataset.com/)
- [INTERACTION](http://interaction-dataset.com/)
- [NuPlan](https://www.nuscenes.org/nuplan)
- [Waymo Open Motion Dataset v1.2 (WOMD)](https://waymo.com/open/about/)

对于HighD, InD, RounD, ExiD, INTERACTION，如果申请数据集所需时间过长，可以考虑加入QQ群互帮互助。

你可以将数据集放在任意位置，然后通过软链接的方式将数据集链接到`tactics2d`的数据目录下，或者修改数据集解析函数的路径。

### 3. 运行示例

安装好`tactics2d`后，你可以运行[样例代码](docs/tutorials)。

其中，[train_parking_demo.ipynb](docs/tutorial/train_parking_demo.ipynb)是[HOPE](https://github.com/jiamiya/HOPE)的简化版本，为了成功运行这一示例，你需要安装`torch`和`torchvision`，并拉取子模块`rllib`。

```bash
git submodule update --init --recursive
```

#### 3.1 Wheel Loader 场景的 PPO primitive 在线规划

当前仓库已经支持在`tactics2d`内部直接加载`PPO_articulated_vehicle`的策略权重和 primitive 库，用于 wheel loader 场景的在线重规划。运行时链路为：PPO 策略先选择 primitive，再将 primitive rollout 转成短时参考轨迹，最后交给`tactics2d`现有的跟踪与渲染链路执行。

运行前请确认工作区中存在以下内容：

- `PPO_articulated_vehicle`
- `BestCheckPoint/PPO_best.pt`
- `BestCheckPoint/adaptive_primitives/active_version.json`

#### 3.2 交互式 pygame 可视化

如果你想查看单个随机场景中的规划结果、参考轨迹和车辆执行过程，推荐使用`pygame_scene_player.py`：

```bash
cd tactics2d-articulated_Vehicle
PYGAME_HIDE_SUPPORT_PROMPT=1 python examples/pygame_scene_player.py \
  --scene wheel_loader \
  --backend ppo \
  --scene-type navigation \
  --map-level Normal \
  --wheel-loader-planner ppo \
  --ppo-checkpoint /Users/firefly/Desktop/Research/Codes/BestCheckPoint/PPO_best.pt \
  --ppo-root /Users/firefly/Desktop/Research/Codes/PPO_articulated_vehicle
```

如果你希望显式指定环境，也可以这样运行：

```bash
cd tactics2d-articulated_Vehicle
PYGAME_HIDE_SUPPORT_PROMPT=1 conda run -n ppomp python examples/pygame_scene_player.py \
  --scene wheel_loader \
  --backend ppo \
  --scene-type navigation \
  --map-level Normal \
  --wheel-loader-planner ppo \
  --ppo-checkpoint /Users/firefly/Desktop/Research/Codes/BestCheckPoint/PPO_best.pt \
  --ppo-root /Users/firefly/Desktop/Research/Codes/PPO_articulated_vehicle
```

常用参数说明：

- `--scene-type {navigation,bay,parallel}`：切换生成的 wheel loader 任务类型
- `--map-level`：选择 PPO 场景生成器的难度等级
- `--wheel-loader-planner ppo`：启用 PPO primitive 在线规划；默认值`default`会保留原有参考轨迹
- `--ppo-checkpoint`：指定 PPO 权重文件路径
- `--ppo-root`：当`PPO_articulated_vehicle`不在默认相对位置时，可手动指定 PPO 仓库根目录
- `--ppo-stochastic`：默认使用 deterministic 推理；传入该参数后改为随机采样动作

说明：

- 示例脚本会优先导入当前仓库内的本地`tactics2d`源码，因此建议从仓库目录运行
- 当前 PPO 接入主要面向`wheel_loader + backend=ppo`组合
- 如果 PPO 规划无法生成有效的短时参考轨迹，运行时会自动回退到基础 guidance/reference path，不会直接中断示例

如果你只想做无窗口快速验证，可以运行对应测试：

```bash
python -m pytest tests/test_pygame_runtime_ppo_bridge.py -q
```

#### 3.3 Checkpoint stress test

如果你想批量评估 checkpoint 在不同随机场景和不同难度等级下的表现，可以使用`wheel_loader_stress_test.py`：

```bash
cd tactics2d-articulated_Vehicle
python examples/wheel_loader_stress_test.py \
  --checkpoint /Users/firefly/Desktop/Research/Codes/BestCheckPoint/PPO_best.pt \
  --ppo-root /Users/firefly/Desktop/Research/Codes/PPO_articulated_vehicle \
  --levels Normal Complex Extrem \
  --episodes-per-level 1 \
  --mode visual \
  --scene-type navigation \
  --replan-every-steps 5
```

参数说明：

- `--mode visual`：显示 pygame 窗口和规划参考轨迹
- `--mode background`：隐藏窗口，但保留同样的 rollout 逻辑，适合批量跑实验
- `--levels`：指定难度等级列表，默认会跑`Normal`、`Complex`、`Extrem`
- `--episodes-per-level`：每个难度等级生成并评估多少个随机场景
- `--output`：将完整结果写入 JSON 文件，便于后处理

stress test 汇总结果包含：

- 各难度等级的规划成功率
- 最终状态计数，如`goal_reached`、`collision`、`out_of_bounds`、`max_steps`
- 平均 rollout 步数、平均规划次数和单回合耗时
- 每次规划调用的 checkpoint 推理延迟统计，包括`mean`、`p50`、`p95`、`max`

该测试复用了现有运行时链路：[tactics2d/map/generator/generate_wheel_loader_scenario.py](tactics2d/map/generator/generate_wheel_loader_scenario.py)、[tactics2d/renderer/ppo_primitive_bridge.py](tactics2d/renderer/ppo_primitive_bridge.py)、[tactics2d/renderer/pygame_runtime.py](tactics2d/renderer/pygame_runtime.py)和[tactics2d/renderer/wheel_loader_stress.py](tactics2d/renderer/wheel_loader_stress.py)，而不是额外维护一套独立仿真栈。

### 4. 更多示例

我们为`tactics2d`搭建了一套完整的集成测试流程，其中的测试代码可以作为函数接口用法的参考。你可以在[这里](tests)找到这些测试代码。运行测试代码的方法如下：

```bash
pip install pytest
pytest tests/[test_file_name]::[test_function_name]
```

## 可视化展示

### 高速场景

<table>
  <tr>
    <th>HighD (Location 3)</th>
    <th>ExiD (Location 6)</th>
  </tr>
  <tr>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/highD_loc_3.gif" align="left" style="width: 100%" />
    </td>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/exiD_loc_6.gif" align="left" style="width: 100%" />
    </td>
  </tr>
</table>

### 路口场景

<table>
  <tr>
    <th>InD (Location 4)</th>
    <th>Argoverse</th>
  </tr>
  <tr>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/inD_loc_4.gif" align="left" style="width: 95%" />
    </td>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/argoverse_sample.gif" align="left" style="width: 100%" />
    </td>
  </tr>
</table>

<table>
  <tr>
    <th>INTERACTION</th>
    <th>WOMD</th>
  </tr>
  <tr>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/DR_USA_Intersection_GL.gif" align="left" style="width: 100%" />
    </td>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/womd_sample.gif" align="left" style="width: 70%" />
    </td>
  </tr>
</table>

### 环岛场景

<table>
  <tr>
    <th>RounD (Location 0)</th>
    <th>INTERACTION</th>
  </tr>
  <tr>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/rounD_loc_0.gif" align="left" style="width: 100%" />
    </td>
    <td valign="top" width="50%">
    <img src="docs/assets/replay_dataset/DR_DEU_Roundabout_OF.gif" align="left" style="width: 100%" />
    </td>
  </tr>
</table>

### 泊车场景

<table>
  <tr>
    <th>DLP</th>
    <th>Self-generated</th>
  </tr>
  <tr>
    <td valign="top" width="70%">
    <img src="docs/assets/replay_dataset/DLP_sample.gif" align="left" style="width: 100%" />
    </td>
    <td valign="top" width="20%">
    <img src="" align="left" style="width: 100%" />
    </td>
  </tr>
</table>

### 赛车场景

## 引用

如果`tactics2d`对你的研究有所帮助，请在你的论文中引用我们。

```bibtex
@article{li2024tactics2d,
  title={Tactics2D: A Highly Modular and Extensible Simulator for Driving Decision-Making},
  author={Li, Yueyuan and Zhang, Songan and Jiang, Mingyang and Chen, Xingyuan and Yang, Jing and Qian, Yeqiang and Wang, Chunxiang and Yang, Ming},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE}
}
```

## 基于`tactics2d`的工作

欢迎大家提交 Pull Request，更新基于`tactics2d`的工作。

Jiang, Mingyang\*, Li, Yueyuan\*, Zhang, Songan, et al. "[HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios](https://arxiv.org/abs/2405.20579)." *IEEE Transactions on Intelligent Transportation Systems* (2025). (\*Co-first author) | [Code](https://github.com/jiamiya/HOPE) | [Demo](https://www.youtube.com/watch?v=62w9qhjIuRI)
