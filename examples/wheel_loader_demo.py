##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: wheel_loader_demo.py
# @Description: Demonstration of wheel loader with pure pursuit controller.
# @Author: Tactics2D Team
# @Version: 1.0.0

import numpy as np
import pygame
import sys
from shapely.geometry import LineString, Point, Polygon

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


class Visualizer:
    def __init__(self, width=800, height=800, map_size=50.0):
        pygame.init()
        self.width = width
        self.height = height
        self.map_size = map_size
        self.scale = min(width, height) / map_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wheel Loader Demo")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 100, 100)
        self.BLUE = (100, 100, 255)
        self.GREEN = (100, 255, 100)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)

    def transform_point(self, x, y):
        """Transform world coordinates to screen coordinates."""
        screen_x = int(x * self.scale)
        screen_y = int(self.height - y * self.scale)
        return screen_x, screen_y

    def draw_polygon(self, polygon, color, width=0):
        if isinstance(polygon, Polygon):
            points = [self.transform_point(x, y) for x, y in polygon.exterior.coords]
            pygame.draw.polygon(self.screen, color, points, width)

    def draw_arrow(self, start_pos, angle, length=2.0, color=(0, 0, 0)):
        start_screen = self.transform_point(*start_pos)
        end_x = start_pos[0] + length * np.cos(angle)
        end_y = start_pos[1] + length * np.sin(angle)
        end_screen = self.transform_point(end_x, end_y)
        
        pygame.draw.line(self.screen, color, start_screen, end_screen, 2)
        
        # Draw arrow head
        arrow_angle = np.pi / 6
        head_len = length * 0.3
        
        left_x = end_x - head_len * np.cos(angle - arrow_angle)
        left_y = end_y - head_len * np.sin(angle - arrow_angle)
        left_screen = self.transform_point(left_x, left_y)
        
        right_x = end_x - head_len * np.cos(angle + arrow_angle)
        right_y = end_y - head_len * np.sin(angle + arrow_angle)
        right_screen = self.transform_point(right_x, right_y)
        
        pygame.draw.polygon(self.screen, color, [end_screen, left_screen, right_screen])

    def render(self, map_, wheel_loader, path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.WHITE)

        # Draw obstacles
        for area in map_.areas.values():
            self.draw_polygon(area.geometry, self.GRAY)

        # Draw path
        if path:
            points = [self.transform_point(x, y) for x, y in path.coords]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.GREEN, False, points, 2)

        # Draw trajectory
        traj_points = []
        for frame in wheel_loader.trajectory.frames:
            state = wheel_loader.trajectory.get_state(frame)
            traj_points.append(self.transform_point(state.x, state.y))
        
        if len(traj_points) > 1:
            pygame.draw.lines(self.screen, self.BLUE, False, traj_points, 1)

        # Draw wheel loader
        rear_poly, front_poly = wheel_loader.get_pose()
        self.draw_polygon(rear_poly, self.RED)
        self.draw_polygon(front_poly, self.RED)
        
        # Draw rear axle center
        current_state = wheel_loader.current_state
        center_screen = self.transform_point(current_state.x, current_state.y)
        pygame.draw.circle(self.screen, self.BLACK, center_screen, 5)
        
        # Draw heading arrow
        self.draw_arrow((current_state.x, current_state.y), current_state.heading)

        pygame.display.flip()
        # self.clock.tick(60)


def main():
    """Main demonstration function."""
    print("=== 铰接转向车辆（轮式装载机）演示 ===")
    
    # 1. 创建场景
    print("\n1. 创建场景...")
    map_ = Map(name="WheelLoaderScenario", scenario_type="wheel_loader")
    scenario_generator = WheelLoaderScenarioGenerator(
        width=50.0,
        height=50.0,
        num_obstacles=8,
        obstacle_radius_range=(1.0, 3.0),
        min_obstacle_spacing=5.0,
    )
    scenario_generator.generate(map_, seed=42)
    print(f"   场景大小: {scenario_generator.width}m × {scenario_generator.height}m")
    print(f"   障碍物数量: {len(map_.areas) - 1}")  # -1 for boundary
    
    # 2. 创建轮式装载机
    print("\n2. 创建轮式装载机...")
    wheel_loader = WheelLoader(
        id_=0,
        type_="wheel_loader",
        # length=6.0,  # Removed total length to specify section lengths
        width=2.5,  # 宽度约2.5米
        height=3.0,  # 高度约3米
        axle_distance=3.0,  # 前后桥距离3米
        bucket_length=1.0,  # 铲斗长度1米 (Adjusted to fit in 2m front section)
        rear_section_length=2.0, # 后车长度 5.0m
        front_section_length=2.0, # 前车长度 5.0m
        max_articulation=np.pi / 3,  # 最大铰接角60度
        max_speed=2.0,  # 最大速度2 m/s
        max_accel=2.0,  # 最大加速度2 m/s^2
        max_decel=5.0,  # 最大减速度5 m/s^2
        verify=True,
    )
    
    # 创建物理模型
    wheel_loader.physics_model = ArticulatedVehicleKinematics(
        L=wheel_loader.axle_distance,
        l1=wheel_loader.axle_distance / 2,
        l2=wheel_loader.axle_distance / 2,
        articulation_range=wheel_loader.articulation_range,
        speed_range=wheel_loader.speed_range,
        accel_range=wheel_loader.accel_range,
        interval=100,  # 100ms
    )
    
    print(f"   车辆长度: {wheel_loader.length}m")
    print(f"   车辆宽度: {wheel_loader.width}m")
    print(f"   前后桥距离: {wheel_loader.axle_distance}m")
    print(f"   铲斗长度: {wheel_loader.bucket_length}m")
    print(f"   最大铰接角: {np.degrees(wheel_loader.max_articulation):.1f}度")
    
    # 3. 创建路径
    print("\n3. 创建路径...")
    path = create_path()
    print(f"   路径长度: {path.length:.2f}m")
    print(f"   路径点数: {len(path.coords)}")
    
    # 4. 创建控制器
    print("\n4. 创建纯跟踪控制器...")
    controller = ArticulatedPurePursuitController(
        min_pre_aiming_distance=3.0,
        target_speed=0.5,
    )
    print(f"   预览距离: {controller.min_pre_aiming_distance}m")
    print(f"   目标速度: {controller.target_speed}m/s")
    
    # 5. 初始化状态
    print("\n5. 初始化车辆状态...")
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
    
    print(f"   初始位置: ({initial_x:.2f}, {initial_y:.2f})")
    print(f"   初始航向: {np.degrees(initial_heading):.1f}度")
    
    # 初始化可视化
    visualizer = Visualizer(map_size=50.0)

    # 6. 运行仿真
    print("\n6. 运行仿真...")
    max_steps = 5000
    dt_ms = 100  # 100ms per step
    is_forward = True  # 前进模式
    
    print(f"   最大步数: {max_steps}")
    print(f"   时间步长: {dt_ms}ms")
    print(f"   行驶方向: {'前进' if is_forward else '后退'}")
    
    for step in range(max_steps):
        current_state = wheel_loader.current_state
        current_frame = current_state.frame
        
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
            current_articulation=wheel_loader.current_articulation,
        )
        
        # 添加状态到轨迹
        wheel_loader.add_state(next_state, articulation_angle=applied_articulation)
        
        # 更新可视化
        visualizer.render(map_, wheel_loader, path)
        pygame.time.wait(20)  # 稍微延时以便观察

        # 检查是否到达终点
        rear_pos = current_state.location
        distance_to_end = np.sqrt(
            (rear_pos[0] - path.coords[-1][0])**2 + 
            (rear_pos[1] - path.coords[-1][1])**2
        )
        
        if distance_to_end < 2.0:
            print(f"\n   到达终点！步数: {step + 1}")
            break
        
        # 每50步打印一次状态
        if (step + 1) % 50 == 0:
            print(
                f"   步数 {step + 1}: "
                f"位置=({current_state.x:.2f}, {current_state.y:.2f}), "
                f"速度={current_state.speed:.2f}m/s, "
                f"铰接角={np.degrees(wheel_loader.current_articulation):.1f}度"
            )
    
    # 7. 输出结果
    print("\n7. 仿真结果:")
    final_state = wheel_loader.current_state
    print(f"   最终位置: ({final_state.x:.2f}, {final_state.y:.2f})")
    print(f"   最终速度: {final_state.speed:.2f}m/s")
    print(f"   最终航向: {np.degrees(final_state.heading):.1f}度")
    print(f"   轨迹点数: {len(wheel_loader.trajectory.frames)}")
    
    # 计算路径跟踪误差
    rear_positions = [
        (wheel_loader.trajectory.get_state(frame).x, 
         wheel_loader.trajectory.get_state(frame).y)
        for frame in wheel_loader.trajectory.frames
    ]
    
    # 计算到路径的平均距离
    distances = []
    for pos in rear_positions:
        point = Point(pos)
        closest_point = path.interpolate(path.project(point))
        distance = point.distance(closest_point)
        distances.append(distance)
    
    avg_error = np.mean(distances)
    max_error = np.max(distances)
    print(f"   平均跟踪误差: {avg_error:.3f}m")
    print(f"   最大跟踪误差: {max_error:.3f}m")
    
    print("\n=== 演示完成 ===")
    
    # 保持窗口打开直到关闭
    print("按任意键退出...")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                return wheel_loader, map_, path
        pygame.time.wait(100)

    return wheel_loader, map_, path


if __name__ == "__main__":
    wheel_loader, map_, path = main()

