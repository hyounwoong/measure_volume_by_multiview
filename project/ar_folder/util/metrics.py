"""
평가 지표 계산 모듈
"""
import numpy as np
from .constants import ANGLE_SEGMENTS, INNER_RADIUS_RATIO_FOR_GRID, GRID_SIZE


def calculate_ppm(inlier_points, radius):
    """둘레당 인라이너 포인트 수 계산 (Points Per Meter)"""
    if radius <= 0:
        return 0.0
    circumference = 2 * np.pi * radius
    n_inliers = len(inlier_points)
    ppm = n_inliers / circumference if circumference > 0 else 0
    return ppm


def calculate_angle_coverage(inlier_points, center):
    """인라이너가 원 둘레를 커버하는 정도 계산 (0.0 ~ 1.0)"""
    if len(inlier_points) == 0:
        return 0.0
    
    relative_points = inlier_points - center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    segment_size = 2 * np.pi / ANGLE_SEGMENTS
    segment_indices = (angles / segment_size).astype(int)
    segment_indices = np.clip(segment_indices, 0, ANGLE_SEGMENTS - 1)
    
    covered_segments = len(np.unique(segment_indices))
    coverage = covered_segments / ANGLE_SEGMENTS
    
    return coverage


def calculate_grid_coverage(points_2d, center, radius, inner_radius_ratio=INNER_RADIUS_RATIO_FOR_GRID, grid_size=GRID_SIZE):
    """그리드 기반 면적 커버리지 계산 (0.0 ~ 1.0, 낮을수록 좋음)"""
    inner_radius = radius * inner_radius_ratio
    
    distances = np.linalg.norm(points_2d - center, axis=1)
    inner_points = points_2d[distances < inner_radius]
    
    if len(inner_points) == 0:
        return 0.0
    
    x_min, x_max = center[0] - inner_radius, center[0] + inner_radius
    y_min, y_max = center[1] - inner_radius, center[1] + inner_radius
    
    cell_size_x = (x_max - x_min) / grid_size
    cell_size_y = (y_max - y_min) / grid_size
    
    if cell_size_x <= 0 or cell_size_y <= 0:
        return 0.0
    
    grid_x = ((inner_points[:, 0] - x_min) / cell_size_x).astype(int)
    grid_y = ((inner_points[:, 1] - y_min) / cell_size_y).astype(int)
    
    grid_x = np.clip(grid_x, 0, grid_size - 1)
    grid_y = np.clip(grid_y, 0, grid_size - 1)
    
    occupied_cells = set(zip(grid_x, grid_y))
    coverage = len(occupied_cells) / (grid_size * grid_size)
    
    return coverage
