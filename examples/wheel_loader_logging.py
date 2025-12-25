##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: wheel_loader_logging.py
# @Description: Simulation of wheel loader with data logging.
# @Author: Tactics2D Team
# @Version: 1.0.0

import csv
import os
import sys
import numpy as np
from shapely.geometry import LineString

# Add the parent directory to sys.path to allow importing tactics2d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tactics2d.controller import ArticulatedPurePursuitController
from tactics2d.map.element.map import Map
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator
from tactics2d.participant.element.wheel_loader import WheelLoader
from tactics2d.participant.trajectory import State
from tactics2d.physics import ArticulatedVehicleKinematics


def create_path() -> LineString:
    """Create a simple path for the wheel loader to follow.

    Returns:
        LineString: The path waypoints.
    """
    # Create a simple S-shaped path
    waypoints = [
        (5.0, 5.0),
        (15.0, 8.0),
        (25.0, 12.0),
        (35.0, 15.0),
        (40.0, 25.0),
        (35.0, 35.0),
        (25.0, 40.0),
        (15.0, 42.0),
        (5.0, 45.0),
    ]
    return LineString(waypoints)


def main():
    """Main simulation and logging function."""
    print("=== 工程铰链车（轮式装载机）仿真与数据记录 ===")
    
    # 1. 创建场景
    print("\n1. 初始化场景...")
    map_ = Map(name="WheelLoaderScenario", scenario_type="wheel_loader")
    scenario_generator = WheelLoaderScenarioGenerator(
        width=50.0,
        height=50.0,
        num_obstacles=8,
        obstacle_radius_range=(1.0, 3.0),
        min_obstacle_spacing=5.0,
    )
    scenario_generator.generate(map_, seed=42)
    
    # 2. 创建轮式装载机
    print("2. 初始化车辆...")
    wheel_loader = WheelLoader(
        id_=0,
        type_="wheel_loader",
        length=6.0,
        width=2.5,
        height=3.0,
        axle_distance=3.0,
        bucket_length=2.0,
        max_articulation=np.pi / 3,
        max_speed=5.0,
        max_accel=2.0,
        max_decel=5.0,
        verify=True,
    )
    
    # 创建物理模型
    wheel_loader.physics_model = ArticulatedVehicleKinematics(
        L=wheel_loader.axle_distance,
        articulation_range=wheel_loader.articulation_range,
        speed_range=wheel_loader.speed_range,
        accel_range=wheel_loader.accel_range,
        interval=100,  # 100ms
    )
    
    # 3. 创建路径
    print("3. 生成路径...")
    path = create_path()
    
    # 4. 创建控制器
    print("4. 初始化控制器...")
    controller = ArticulatedPurePursuitController(
        min_pre_aiming_distance=3.0,
        target_speed=2.0,
    )
    
    # 5. 初始化状态
    print("5. 设置初始状态...")
    initial_x, initial_y = path.coords[0]
    initial_heading = np.arctan2(
        path.coords[1][1] - path.coords[0][1],
        path.coords[1][0] - path.coords[0][0],
    )
    
    initial_state = State(
        frame=0,
        x=initial_x,
        y=initial_y,
        heading=initial_heading,
        speed=0.0,
        accel=0.0,
    )
    wheel_loader.trajectory.add_state(initial_state)
    wheel_loader.current_articulation = 0.0
    
    # 6. 准备日志文件
    log_filename = "wheel_loader_log.csv"
    print(f"\n6. 开始仿真并记录数据到 {log_filename} ...")
    
    with open(log_filename, mode='w', newline='') as csv_file:
        fieldnames = ['frame', 'time_sec', 'x', 'y', 'speed', 'heading_rad', 'heading_deg', 'articulation_rad', 'articulation_deg']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # 记录初始状态
        writer.writerow({
            'frame': initial_state.frame,
            'time_sec': initial_state.frame / 1000.0,
            'x': f"{initial_state.x:.4f}",
            'y': f"{initial_state.y:.4f}",
            'speed': f"{initial_state.speed:.4f}",
            'heading_rad': f"{initial_state.heading:.4f}",
            'heading_deg': f"{np.degrees(initial_state.heading):.4f}",
            'articulation_rad': f"{wheel_loader.current_articulation:.4f}",
            'articulation_deg': f"{np.degrees(wheel_loader.current_articulation):.4f}",
        })

        # 7. 运行仿真
        max_steps = 500
        dt_ms = 100  # 100ms per step
        is_forward = True
        
        for step in range(max_steps):
            current_state = wheel_loader.current_state
            
            # 获取前桥位置
            front_axle_pos = wheel_loader.get_front_axle_position(
                articulation_angle=wheel_loader.current_articulation
            )
            
            # 计算控制命令
            articulation_angle, accel = controller.step(
                rear_axle_state=current_state,
                front_axle_state=front_axle_pos,
                waypoints=path,
                axle_distance=wheel_loader.axle_distance,
                is_forward=is_forward,
            )
            
            # 更新状态
            next_state, applied_accel, applied_articulation = wheel_loader.physics_model.step(
                state=current_state,
                accel=accel,
                articulation_angle=articulation_angle,
                interval=dt_ms,
            )
            
            # 添加状态到轨迹
            wheel_loader.add_state(next_state, articulation_angle=applied_articulation)
            
            # 记录数据
            writer.writerow({
                'frame': next_state.frame,
                'time_sec': next_state.frame / 1000.0,
                'x': f"{next_state.x:.4f}",
                'y': f"{next_state.y:.4f}",
                'speed': f"{next_state.speed:.4f}",
                'heading_rad': f"{next_state.heading:.4f}",
                'heading_deg': f"{np.degrees(next_state.heading):.4f}",
                'articulation_rad': f"{applied_articulation:.4f}",
                'articulation_deg': f"{np.degrees(applied_articulation):.4f}",
            })

            # 检查是否到达终点
            rear_pos = current_state.location
            distance_to_end = np.sqrt(
                (rear_pos[0] - path.coords[-1][0])**2 + 
                (rear_pos[1] - path.coords[-1][1])**2
            )
            
            if distance_to_end < 2.0:
                print(f"   到达终点！总步数: {step + 1}")
                break
            
            # 每50步打印一次状态
            if (step + 1) % 50 == 0:
                print(
                    f"   步数 {step + 1}: "
                    f"位置=({current_state.x:.2f}, {current_state.y:.2f}), "
                    f"速度={current_state.speed:.2f}m/s, "
                    f"铰接角={np.degrees(wheel_loader.current_articulation):.1f}度"
                )
    
    print(f"\n仿真结束。数据已保存至 {os.path.abspath(log_filename)}")


if __name__ == "__main__":
    main()
