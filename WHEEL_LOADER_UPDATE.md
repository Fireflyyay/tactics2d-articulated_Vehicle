# 铰接转向车辆（轮式装载机）功能更新说明

## 概述

本次更新为 Tactics2D 仿真器增加了铰接转向车辆（轮式装载机）的运动学仿真功能，包括车辆模型、控制器、场景生成和可视化支持。

## 新增文件

### 1. 物理模型
- **文件**: `tactics2d/physics/articulated_vehicle_kinematics.py`
- **类**: `ArticulatedVehicleKinematics`
- **功能**: 实现铰接转向车辆的运动学模型
  - 前后桥距离：3米（可配置）
  - 支持铰接角控制
  - 支持前进和后退运动
  - 状态验证功能

### 2. 车辆类
- **文件**: `tactics2d/participant/element/wheel_loader.py`
- **类**: `WheelLoader`
- **功能**: 铰接转向车辆参与者类
  - 继承自 `ParticipantBase`
  - 前后两部分的几何形状定义
  - 支持铰接角状态管理
  - 提供前后桥位置查询接口

### 3. 控制器
- **文件**: `tactics2d/controller/articulated_pure_pursuit_controller.py`
- **类**: `ArticulatedPurePursuitController`
- **功能**: 铰接转向车辆的纯跟踪控制器
  - 前进模式：以前桥几何中心为参考点跟踪路径
  - 后退模式：以后桥几何中心为参考点跟踪路径
  - 输出铰接角和加速度命令

### 4. 场景生成器
- **文件**: `tactics2d/map/generator/generate_wheel_loader_scenario.py`
- **类**: `WheelLoaderScenarioGenerator`
- **功能**: 生成测试场景
  - 50米 × 50米方形边界
  - 随机分布的圆形障碍物（默认8个）
  - 可配置障碍物数量和大小

### 5. 演示代码
- **文件**: `examples/wheel_loader_demo.py`
- **功能**: 完整的演示程序
  - 场景创建和加载
  - 车辆初始化和配置
  - 路径跟踪控制
  - 仿真运行和结果输出

- **文件**: `examples/articulated_steering_test.py`
- **功能**: 铰接转向运动学可视化测试
  - 验证静止状态下的折腰转向几何
  - 可视化前后车体绕铰接点的相对转动

- **文件**: `examples/wheel_loader_logging.py`
- **功能**: 数据记录仿真
  - 运行仿真并将车辆状态数据保存为 CSV 文件

## 修改文件

### 1. `tactics2d/physics/__init__.py`
- 添加 `ArticulatedVehicleKinematics` 的导出

### 2. `tactics2d/participant/element/__init__.py`
- 添加 `WheelLoader` 的导出

### 3. `tactics2d/controller/__init__.py`
- 添加 `ArticulatedPurePursuitController` 的导出

## 主要特性

### 1. 运动学模型参数
- **前后桥距离 (L)**: 3.0米（默认值，可配置）
- **铲斗长度**: 2.0米（可配置）
- **最大铰接角**: ±60度（π/3弧度，可配置）
- **最大速度**: 5.0 m/s（可配置）
- **最大加速度**: 2.0 m/s²（可配置）
- **最大减速度**: 5.0 m/s²（可配置）

### 2. 控制器特性
- **预览距离**: 根据速度自适应调整，最小2.0米
- **目标速度**: 2.0 m/s（可配置）
- **方向切换**: 自动根据行驶方向选择参考点
  - 前进：前桥中心
  - 后退：后桥中心

### 3. 场景特性
- **边界尺寸**: 50米 × 50米
- **障碍物**: 圆形障碍物，半径范围1.0-3.0米
- **障碍物间距**: 最小5.0米
- **障碍物数量**: 默认8个（可配置）

## 使用方法

### 基本使用示例

```python
from tactics2d.participant.element.wheel_loader import WheelLoader
from tactics2d.physics import ArticulatedVehicleKinematics
from tactics2d.controller import ArticulatedPurePursuitController
from tactics2d.map.element.map import Map
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator

# 1. 创建场景
map_ = Map(name="WheelLoaderScenario", scenario_type="wheel_loader")
generator = WheelLoaderScenarioGenerator(width=50.0, height=50.0, num_obstacles=8)
generator.generate(map_, seed=42)

# 2. 创建车辆
wheel_loader = WheelLoader(
    id_=0,
    length=6.0,
    width=2.5,
    axle_distance=3.0,
    bucket_length=2.0,
    max_articulation=np.pi/3,
    verify=True,
)

# 3. 创建物理模型
wheel_loader.physics_model = ArticulatedVehicleKinematics(
    L=wheel_loader.axle_distance,
    articulation_range=wheel_loader.articulation_range,
    speed_range=wheel_loader.speed_range,
    accel_range=wheel_loader.accel_range,
)

# 4. 创建控制器
controller = ArticulatedPurePursuitController(
    min_pre_aiming_distance=3.0,
    target_speed=2.0,
)

# 5. 运行仿真
# ... (参考 examples/wheel_loader_demo.py)
```

### 运行演示

#### 1. 完整仿真演示
运行包含路径跟踪和障碍物环境的完整仿真：
```bash
python examples/wheel_loader_demo.py
```

#### 2. 运动学可视化测试
验证静止状态下的折腰转向机制（观察前后车体如何绕铰接点转动）：
```bash
python examples/articulated_steering_test.py
```

#### 3. 数据记录仿真
运行仿真并将状态数据（位置、速度、航向、铰接角等）记录到 CSV 文件：
```bash
python examples/wheel_loader_logging.py
```

## 技术细节

### 运动学方程（更新版）

采用了符合真实物理特性的铰接车辆运动学模型，支持静止转向（Scrubbing）效应。

**模型假设：**
1. 车辆由前后两个刚体组成，通过铰接点连接。
2. 控制输入为**折腰角速度** $\omega$（模拟铰链力矩效果）和后桥纵向速度 $v$。
3. 考虑了转向过程中前后车体的几何约束耦合。

**参数定义：**
- $l_1$: 前轴到铰接点的距离
- $l_2$: 后轴到铰接点的距离
- $\phi$: 折腰角（前车体相对于后车体的偏角）
- $\theta_2$: 后车体航向角

**核心方程：**

1. **后车体航向角变化率**：
   $$ \dot{\theta}_2 = \frac{v \sin\phi - l_2 \omega \cos\phi}{l_1 \cos\phi + l_2} $$
   
   该公式表明，即使在静止状态下（$v=0$），只要有折腰动作（$\omega \neq 0$），后车体也会发生转动。

2. **前车体航向角变化率**：
   $$ \dot{\theta}_1 = \dot{\theta}_2 + \omega $$

3. **位置更新**：
   后轴中心的位置变化不仅取决于纵向速度，还包含由折腰转向引起的侧向位移分量。

### 控制器算法

纯跟踪控制器基于以下原理：
1. 在路径上选择预览点（preview point）
2. 计算参考点到预览点的角度误差
3. 将角度误差转换为铰接角命令
4. 使用加速度控制器调节速度

### 可视化

车辆由两部分组成：
- **后部**: 以后桥中心为参考的矩形
- **前部**: 以前桥中心为参考的矩形（包括铲斗）

两部分通过铰接角连接，形成完整的车辆形状。

## 注意事项

1. **状态管理**: 铰接角不包含在 `State` 类中，需要单独管理（`current_articulation` 属性）
2. **物理模型**: 当前实现为运动学模型，不考虑动力学效应
3. **控制器**: 纯跟踪控制器适用于低速场景，高速时可能需要更复杂的控制策略
4. **场景生成**: 障碍物生成可能因空间限制而少于请求数量

## 未来改进方向

1. 添加动力学模型支持
2. 实现更高级的路径跟踪控制器（如MPC）
3. 添加碰撞检测功能
4. 实现更复杂的场景生成（如动态障碍物）
5. 添加可视化渲染支持

## 版本信息

- **版本**: 1.0.0
- **更新日期**: 2025年
- **兼容性**: 与 Tactics2D 现有框架兼容

