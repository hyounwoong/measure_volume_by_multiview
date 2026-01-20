"""
시각화 함수 모듈
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from .constants import MIN_POINTS_PER_SLICE, INNER_RADIUS_RATIO_FOR_GRID
from .geometry import get_slice, detect_circle_at_axis
from .metrics import calculate_angle_coverage, calculate_grid_coverage


def visualize_volume_3d(volume_data, center_axis, z_min=None, z_max=None):
    """3D visualization of volume calculation result"""
    if not volume_data:
        print("No volume data to visualize")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sort by height
    sorted_data = sorted(volume_data, key=lambda x: x['z'])
    
    # Draw circles at each height
    for data in sorted_data:
        z = data['z']
        radius = data['radius']
        
        theta = np.linspace(0, 2*np.pi, 50)
        x = center_axis[0] + radius * np.cos(theta)
        y = center_axis[1] + radius * np.sin(theta)
        z_circle = np.full_like(theta, z)
        
        ax.plot(x, y, z_circle, 'b-', linewidth=1.5, alpha=0.7)
    
    # Draw center axis
    if sorted_data:
        z_min_actual = sorted_data[0]['z']
        z_max_actual = sorted_data[-1]['z']
        
        z_axis_min = z_min if z_min is not None else z_min_actual
        z_axis_max = z_max if z_max is not None else z_max_actual
        
        ax.plot([center_axis[0], center_axis[0]], 
                [center_axis[1], center_axis[1]], 
                [z_axis_min, z_axis_max], 'g-', linewidth=3, label='Center Axis')
    
    # Calculate ranges for equal aspect ratio
    max_radius = max([d['radius'] for d in sorted_data])
    z_min_actual = sorted_data[0]['z']
    z_max_actual = sorted_data[-1]['z']
    
    x_range = max_radius * 2
    y_range = max_radius * 2
    z_range = z_max_actual - z_min_actual
    
    max_range = max(x_range, y_range, z_range)
    
    # Set equal aspect ratio
    ax.set_xlim(center_axis[0] - max_range/2, center_axis[0] + max_range/2)
    ax.set_ylim(center_axis[1] - max_range/2, center_axis[1] + max_range/2)
    ax.set_zlim((z_min_actual + z_max_actual)/2 - max_range/2, 
                (z_min_actual + z_max_actual)/2 + max_range/2)
    
    # Set equal aspect (matplotlib 3.3.0+)
    try:
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    except AttributeError:
        pass  # Older matplotlib versions
    
    # Calculate total volume
    total_volume = sum((sorted_data[i]['area'] + sorted_data[i+1]['area']) / 2 * 
                      (sorted_data[i+1]['z'] - sorted_data[i]['z']) 
                      for i in range(len(sorted_data) - 1))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Volume Visualization\n'
                f'Total Volume: {total_volume * 1000:.2f} mL ({total_volume * 1e6:.2f} cm³)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_detect_circle_at_axis(points, z, center_axis, threshold=0.0003):
    """Visualize detect_circle_at_axis result for a given height"""
    # Extract slice
    slice_points = get_slice(points, z)
    if len(slice_points) < MIN_POINTS_PER_SLICE:
        print(f"Insufficient points at height {z:.4f}m ({len(slice_points)} points)")
        return
    
    slice_2d = slice_points[:, :2]
    
    # Detect circle
    radius = detect_circle_at_axis(slice_2d, center_axis, threshold=threshold)
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Calculate distances from center axis
    distances = np.linalg.norm(slice_2d - center_axis, axis=1)
    
    if radius:
        # Separate inliers/outliers
        inliers = np.abs(distances - radius) < threshold
        
        # Inliers (red)
        ax.scatter(slice_2d[inliers, 0], slice_2d[inliers, 1], 
                  c='red', s=20, alpha=0.7, label=f'Inliers ({np.sum(inliers)})')
        
        # Outliers (gray)
        ax.scatter(slice_2d[~inliers, 0], slice_2d[~inliers, 1], 
                  c='lightgray', s=5, alpha=0.3, label='Outliers')
        
        # Draw detected circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = center_axis[0] + radius * np.cos(theta)
        circle_y = center_axis[1] + radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'b-', linewidth=2, label=f'Circle (r={radius:.4f}m)')
        
        # Calculate point density
        circumference = 2 * np.pi * radius
        ppm = np.sum(inliers) / circumference if circumference > 0 else 0
        
        ax.set_title(f'Height {z:.4f}m - Circle Detected\nRadius: {radius:.4f}m, Density: {ppm:.1f} points/m')
    else:
        # Circle detection failed
        ax.scatter(slice_2d[:, 0], slice_2d[:, 1], c='gray', s=10, alpha=0.5)
        ax.set_title(f'Height {z:.4f}m - Circle Detection Failed')
    
    # Mark center axis
    ax.plot(center_axis[0], center_axis[1], 'go', markersize=10, 
           label=f'Center Axis ({center_axis[0]:.4f}, {center_axis[1]:.4f})')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def visualize_radius_vs_height(volume_data, figsize=(10, 8)):
    """높이와 반지름의 관계를 2D 그래프로 시각화"""
    if not volume_data:
        print("No volume data to visualize")
        return
    
    from matplotlib.ticker import MultipleLocator
    
    # 높이 순으로 정렬
    sorted_data = sorted(volume_data, key=lambda x: x['z'])
    
    # 데이터 추출
    heights = [data['z'] for data in sorted_data]
    radii = [data['radius'] for data in sorted_data]
    areas = [data['area'] for data in sorted_data]
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 1. 반지름 vs 높이
    ax1 = axes[0]
    ax1.plot(radii, heights, 'b-o', linewidth=1, markersize=4, label='Radius')
    ax1.fill_betweenx(heights, 0, radii, alpha=0.3, color='blue', label='Cross-section')
    ax1.set_xlabel('Radius (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('Radius vs Height', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.invert_yaxis()  # 높이가 아래에서 위로 증가하도록 (일반적인 좌표계)
    
    # x축과 y축 스케일 맞추기
    ax1.set_aspect('equal', adjustable='box')
    
    # 5mm (0.005m) 단위로 tick 설정 (minor ticks로 설정하고 일부만 표시)
    tick_interval = 0.005  # 5mm
    ax1.xaxis.set_minor_locator(MultipleLocator(tick_interval))
    ax1.yaxis.set_minor_locator(MultipleLocator(tick_interval))
    
    # Major ticks는 10mm (0.01m) 간격으로 설정하여 라벨이 너무 촘촘하지 않게
    ax1.xaxis.set_major_locator(MultipleLocator(0.01))
    ax1.yaxis.set_major_locator(MultipleLocator(0.01))
    
    # 통계 정보 추가 (왼쪽 밖으로 많이 이동)
    if heights:
        height_range = heights[-1] - heights[0]
        avg_radius = np.mean(radii)
        max_radius = np.max(radii)
        min_radius = np.min(radii)
        
        info_text = (f'Height range: {height_range:.4f} m\n'
                    f'Avg radius: {avg_radius:.4f} m\n'
                    f'Min radius: {min_radius:.4f} m\n'
                    f'Max radius: {max_radius:.4f} m')
        # 왼쪽 밖으로 많이 이동 (x=-0.3으로 설정)
        ax1.text(-5, 0.98, info_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), 
                fontsize=9)
    
    # 2. 단면적 vs 높이
    ax2 = axes[1]
    ax2.plot(areas, heights, 'r-o', linewidth=2, markersize=4, label='Cross-sectional Area')
    ax2.fill_betweenx(heights, 0, areas, alpha=0.3, color='red')
    ax2.set_xlabel('Cross-sectional Area (m²)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Cross-sectional Area vs Height', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.invert_yaxis()
    
    # 부피 계산 및 표시
    if len(sorted_data) > 1:
        total_volume = sum((sorted_data[i]['area'] + sorted_data[i+1]['area']) / 2 * 
                          (sorted_data[i+1]['z'] - sorted_data[i]['z']) 
                          for i in range(len(sorted_data) - 1))
        
        fig.suptitle(f'Volume Analysis\n'
                    f'Total Volume: {total_volume * 1000:.2f} mL ({total_volume * 1e6:.2f} cm³) | '
                    f'Data points: {len(sorted_data)}', 
                    fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()


def visualize_circle_at_axis(slice_2d, center_axis, radius, threshold=0.0025, title=None, 
                            z=None, points_3d=None, z_thickness=0.0003):
    """
    detect_circle_at_axis로 검출한 원 단면을 시각화 (2D + 3D)
    
    Parameters:
    - slice_2d: 2D 단면 포인트 배열 (N, 2)
    - center_axis: 중심축 좌표 (x, y)
    - radius: 검출된 원의 반지름
    - threshold: 인라이너 판별 임계값
    - title: 그래프 제목 (선택)
    - z: 단면의 z 좌표 (3D 시각화용, 필수)
    - points_3d: 전체 포인트클라우드 (M, 3) - 3D 시각화용
    - z_thickness: 단면 두께 (기본 0.0003m)
    """
    if radius is None:
        print("원이 검출되지 않았습니다.")
        return
    
    # 2D 시각화
    fig_2d, ax_2d = plt.subplots(1, 1, figsize=(10, 10))
    
    # 거리 계산
    distances = np.linalg.norm(slice_2d - center_axis, axis=1)
    
    # 인라이너/아웃라이너 구분
    inlier_mask = np.abs(distances - radius) < threshold
    inlier_points_2d = slice_2d[inlier_mask]
    outlier_points_2d = slice_2d[~inlier_mask]
    
    # 포인트 플롯
    if len(outlier_points_2d) > 0:
        ax_2d.scatter(outlier_points_2d[:, 0], outlier_points_2d[:, 1], 
                     s=1, alpha=1, c='lightgray', label='Outliers')
    
    if len(inlier_points_2d) > 0:
        ax_2d.scatter(inlier_points_2d[:, 0], inlier_points_2d[:, 1], 
                     s=2, alpha=0.6, c='blue', label='Inliers')
    
    # 검출된 원 그리기
    circle_patch = patches.Circle(center_axis, radius, 
                                 fill=False, edgecolor='red', 
                                 linewidth=2, label='Detected Circle')
    ax_2d.add_patch(circle_patch)
    
    # 중심축 표시
    ax_2d.plot(center_axis[0], center_axis[1], 'ro', markersize=10, label='Center Axis')
    
    # 각도 표시를 위한 선 그리기
    if len(inlier_points_2d) > 0:
        relative_points = inlier_points_2d - center_axis
        angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        
        for angle in np.linspace(min_angle, max_angle, 20):
            x_end = center_axis[0] + radius * 1.1 * np.cos(angle)
            y_end = center_axis[1] + radius * 1.1 * np.sin(angle)
            ax_2d.plot([center_axis[0], x_end], [center_axis[1], y_end], 
                      'g-', alpha=0.2, linewidth=0.5)
    
    # 지표 계산
    if len(inlier_points_2d) >= MIN_POINTS_PER_SLICE:
        angle_coverage = calculate_angle_coverage(inlier_points_2d, center_axis)
        grid_coverage = calculate_grid_coverage(slice_2d, center_axis, radius)
        circumference = 2 * np.pi * radius
        ppm = len(inlier_points_2d) / circumference if circumference > 0 else 0
        
        # z 좌표 정보 추가
        z_info = f"Z: {z:.4f}m\n" if z is not None else ""
        info_text = (
            f"{z_info}"
            f"Radius: {radius:.4f}m\n"
            f"Inlier Count: {len(inlier_points_2d)}\n"
            f"Outlier Count: {len(outlier_points_2d)}\n"
            f"Angle Coverage: {angle_coverage:.3f} ({angle_coverage*100:.1f}%)\n"
            f"Grid Coverage: {grid_coverage:.3f} ({grid_coverage*100:.1f}%)\n"
            f"PPM: {ppm:.1f} points/m\n"
            f"Threshold: {threshold:.4f}m"
        )
    else:
        z_info = f"Z: {z:.4f}m\n" if z is not None else ""
        info_text = (
            f"{z_info}"
            f"Radius: {radius:.4f}m\n"
            f"Inlier Count: {len(inlier_points_2d)}\n"
            f"Outlier Count: {len(outlier_points_2d)}\n"
            f"Insufficient points for metrics"
        )
    
    ax_2d.text(0.02, 0.98, info_text, 
              transform=ax_2d.transAxes,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
              fontsize=10)
    
    # 축 설정: 단면 전체를 보여주도록
    x_range = slice_2d[:, 0].max() - slice_2d[:, 0].min()
    y_range = slice_2d[:, 1].max() - slice_2d[:, 1].min()
    x_margin = x_range * 0.1 if x_range > 0 else 0.01
    y_margin = y_range * 0.1 if y_range > 0 else 0.01
    
    ax_2d.set_xlim(slice_2d[:, 0].min() - x_margin, slice_2d[:, 0].max() + x_margin)
    ax_2d.set_ylim(slice_2d[:, 1].min() - y_margin, slice_2d[:, 1].max() + y_margin)
    ax_2d.set_aspect('equal')
    ax_2d.grid(True, alpha=0.3)
    ax_2d.set_xlabel('X (m)')
    ax_2d.set_ylabel('Y (m)')
    
    if title:
        ax_2d.set_title(title)
    else:
        z_str = f"z={z:.4f}m, " if z is not None else ""
        ax_2d.set_title(f'Circle Detection at Axis (2D)\n{z_str}Center: ({center_axis[0]:.4f}, {center_axis[1]:.4f}), Radius: {radius:.4f}m')
    
    ax_2d.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()
    
    # 3D 시각화 (z 좌표와 전체 포인트클라우드가 제공된 경우)
    if z is not None and points_3d is not None:
        # 참고 코드 방식으로 단면 추출: plane_normal 방식 사용
        plane_origin = np.array([center_axis[0], center_axis[1], z])  # 중심축 + z 좌표
        plane_normal = np.array([0, 0, 1])  # Z+ 방향 (중력 방향)
        
        # 단면 추출: 참고 코드와 동일한 방식
        distances_to_plane = np.abs(np.dot(points_3d - plane_origin, plane_normal))
        slice_mask = distances_to_plane < z_thickness
        slice_points_3d = points_3d[slice_mask]  # 단면의 3D 포인트
        other_points_3d = points_3d[~slice_mask]  # 단면이 아닌 포인트
        
        fig_3d = plt.figure(figsize=(14, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # 전체 포인트클라우드 표시 (단면이 아닌 포인트들)
        if len(other_points_3d) > 0:
            # 샘플링하여 표시 (너무 많으면 느려질 수 있음)
            if len(other_points_3d) > 10000:
                sample_idx = np.random.choice(len(other_points_3d), 10000, replace=False)
                other_points_3d_sample = other_points_3d[sample_idx]
            else:
                other_points_3d_sample = other_points_3d
            
            ax_3d.scatter(other_points_3d_sample[:, 0], other_points_3d_sample[:, 1], other_points_3d_sample[:, 2],
                         s=0.5, alpha=0.1, c='gray', label='Other Points')
        
        # 단면 포인트에서 인라이너/아웃라이너 계산 (3D 포인트 사용)
        if len(slice_points_3d) > 0:
            slice_distances = np.linalg.norm(slice_points_3d[:, :2] - center_axis, axis=1)
            slice_inlier_mask = np.abs(slice_distances - radius) < threshold
            slice_inlier_points = slice_points_3d[slice_inlier_mask]
            slice_outlier_points = slice_points_3d[~slice_inlier_mask]
            
            # 단면 아웃라이너 표시
            if len(slice_outlier_points) > 0:
                ax_3d.scatter(slice_outlier_points[:, 0], slice_outlier_points[:, 1], slice_outlier_points[:, 2],
                             s=2, alpha=0.5, c='orange', label='Slice Outliers')
            
            # 단면 인라이너 표시
            if len(slice_inlier_points) > 0:
                ax_3d.scatter(slice_inlier_points[:, 0], slice_inlier_points[:, 1], slice_inlier_points[:, 2],
                             s=3, alpha=0.8, c='blue', label='Inliers')
        
        # 검출된 원을 3D로 표시 (원형)
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = center_axis[0] + radius * np.cos(theta)
        circle_y = center_axis[1] + radius * np.sin(theta)
        circle_z = np.full_like(theta, z)
        
        ax_3d.plot(circle_x, circle_y, circle_z, 'r-', linewidth=3, label='Detected Circle')
        
        # 중심축 표시 (z 방향으로)
        z_range_3d = points_3d[:, 2].max() - points_3d[:, 2].min()
        z_min_3d = points_3d[:, 2].min()
        z_max_3d = points_3d[:, 2].max()
        
        # 중심축 선 (z 방향)
        ax_3d.plot([center_axis[0], center_axis[0]], 
                  [center_axis[1], center_axis[1]], 
                  [z_min_3d, z_max_3d], 
                  'r--', linewidth=2, alpha=0.7, label='Center Axis')
        
        # 중심축 점 (단면 위치)
        ax_3d.scatter([center_axis[0]], [center_axis[1]], [z], 
                     s=200, c='red', marker='o', edgecolors='darkred', 
                     linewidths=2, label='Center at Slice')
        
        # 단면 평면 표시 (반투명)
        if len(slice_points_3d) > 0:
            x_range_3d = slice_points_3d[:, 0].max() - slice_points_3d[:, 0].min()
            y_range_3d = slice_points_3d[:, 1].max() - slice_points_3d[:, 1].min()
            x_center = slice_points_3d[:, 0].mean()
            y_center = slice_points_3d[:, 1].mean()
        else:
            x_range_3d = radius * 2
            y_range_3d = radius * 2
            x_center = center_axis[0]
            y_center = center_axis[1]
        
        plane_size = max(x_range_3d, y_range_3d) * 1.5
        xx = np.array([x_center - plane_size/2, x_center + plane_size/2])
        yy = np.array([y_center - plane_size/2, y_center + plane_size/2])
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.full_like(XX, z)
        
        ax_3d.plot_surface(XX, YY, ZZ, alpha=0.15, color='yellow', label='Slice Plane')
        
        # 축 범위 설정
        x_margin_3d = x_range_3d * 0.1 if x_range_3d > 0 else 0.01
        y_margin_3d = y_range_3d * 0.1 if y_range_3d > 0 else 0.01
        
        # X, Y 범위는 단면 포인트 기준으로
        if len(slice_points_3d) > 0:
            ax_3d.set_xlim(slice_points_3d[:, 0].min() - x_margin_3d, slice_points_3d[:, 0].max() + x_margin_3d)
            ax_3d.set_ylim(slice_points_3d[:, 1].min() - y_margin_3d, slice_points_3d[:, 1].max() + y_margin_3d)
        else:
            ax_3d.set_xlim(center_axis[0] - radius - x_margin_3d, center_axis[0] + radius + x_margin_3d)
            ax_3d.set_ylim(center_axis[1] - radius - y_margin_3d, center_axis[1] + radius + y_margin_3d)
        
        ax_3d.set_zlim(z_min_3d - z_range_3d * 0.1, z_max_3d + z_range_3d * 0.1)
        
        # 참고 코드처럼 box_aspect 설정
        ax_3d.set_box_aspect([1, 1, 1])
        
        ax_3d.set_xlabel('X (m)', fontsize=10)
        ax_3d.set_ylabel('Y (m)', fontsize=10)
        ax_3d.set_zlabel('Z (중력 방향)', fontsize=10)
        
        if title:
            ax_3d.set_title(f'{title} (3D View)', fontsize=12)
        else:
            ax_3d.set_title(f'Circle Detection at Axis (3D)\nz={z:.4f}m, Radius={radius:.4f}m', fontsize=12)
        
        ax_3d.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    elif z is not None:
        print("3D 시각화를 위해서는 points_3d 파라미터가 필요합니다.")


def visualize_top_circles(circle_data, sorted_points, sorted_heights, top_n=5, THICKNESS=0.0003):
    """
    중심축 선정 시 점수가 높은 상위 N개 단면을 시각화
    
    Parameters:
    - circle_data: 점수 계산이 완료된 원 데이터 리스트
    - sorted_points: 정렬된 포인트 배열
    - sorted_heights: 정렬된 높이 배열
    - top_n: 시각화할 상위 개수 (기본 5개)
    - THICKNESS: 단면 두께
    """
    # final_score 기준으로 정렬
    sorted_circles = sorted(circle_data, key=lambda x: x['final_score'], reverse=True)
    top_circles = sorted_circles[:top_n]
    
    # 서브플롯 생성
    n_cols = min(3, top_n)
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if top_n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, circle in enumerate(top_circles):
        ax = axes[idx]
        z = circle['z']
        center = circle['center']
        radius = circle['radius']
        
        # 해당 높이의 슬라이스 포인트 추출
        start = np.searchsorted(sorted_heights, z - THICKNESS)
        end = np.searchsorted(sorted_heights, z + THICKNESS)
        slice_points = sorted_points[start:end, :2]
        
        # 포인트 플롯
        ax.scatter(slice_points[:, 0], slice_points[:, 1], 
                  s=1, alpha=0.5, c='gray', label='Points')
        
        # 검출된 원 그리기
        circle_patch = patches.Circle(center, radius, 
                                     fill=False, edgecolor='red', 
                                     linewidth=2, label='Detected Circle')
        ax.add_patch(circle_patch)
        
        # 원의 중심 표시
        ax.plot(center[0], center[1], 'ro', markersize=8, label='Center')
        
        # 내부 영역 표시 (그리드 커버리지 계산용)
        inner_radius = radius * INNER_RADIUS_RATIO_FOR_GRID
        inner_circle = patches.Circle(center, inner_radius,
                                      fill=False, edgecolor='blue',
                                      linewidth=1, linestyle='--', 
                                      label='Inner Region (60%)')
        ax.add_patch(inner_circle)
        
        # 지표 정보 텍스트
        info_text = (
            f"Rank: #{idx+1}\n"
            f"Height: {z:.4f}m\n"
            f"Final Score: {circle['final_score']:.3f}\n"
            f"PPM: {circle['ppm']:.1f}\n"
            f"Angle Coverage: {circle['angle_coverage']:.3f}\n"
            f"Grid Coverage: {circle['grid_coverage']:.3f}\n"
            f"Radius: {radius:.4f}m"
        )
        
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        # 축 설정
        ax.set_xlim(center[0] - radius * 1.5, center[0] + radius * 1.5)
        ax.set_ylim(center[1] - radius * 1.5, center[1] + radius * 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Top {idx+1}: z={z:.4f}m, Score={circle["final_score"]:.3f}')
        ax.legend(loc='upper right', fontsize=7)
    
    # 사용하지 않는 서브플롯 숨기기
    for idx in range(len(top_circles), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 상위 5개 원의 통계 정보 출력
    print(f"\n=== 상위 {top_n}개 원 통계 ===")
    print(f"{'Rank':<6} {'Height(m)':<12} {'Final Score':<12} {'PPM':<10} {'Angle Cov':<12} {'Grid Cov':<12} {'Radius(m)':<12}")
    print("-" * 90)
    for idx, circle in enumerate(top_circles):
        print(f"{idx+1:<6} {circle['z']:<12.4f} {circle['final_score']:<12.3f} "
              f"{circle['ppm']:<10.1f} {circle['angle_coverage']:<12.3f} "
              f"{circle['grid_coverage']:<12.3f} {circle['radius']:<12.4f}")
