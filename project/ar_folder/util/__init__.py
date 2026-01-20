"""
Util 패키지 - 부피 측정 관련 유틸리티 함수들

이 파일은 backward compatibility를 위해 모든 함수를 export합니다.
기존 코드: from util import function_name
새 코드: from util.module_name import function_name (권장)
"""

# 상수
from .constants import *

# 평가 지표 계산
from .metrics import (
    calculate_ppm,
    calculate_angle_coverage,
    calculate_grid_coverage,
)

# 기하학 계산
from .geometry import (
    get_slice,
    refine_circle_ls,
    fit_circle_ransac,
    detect_circle_at_axis,
    detect_plane_below_point,
    compute_ray_intersection,
)

# 파일 입출력
from .io import (
    load_and_align_scene,
    load_jsonl,
    get_image_center_ray,
)

# 부피 계산
from .volume import (
    add_volume_data,
    search_heights,
)

# 시각화
from .visualization import (
    visualize_volume_3d,
    visualize_detect_circle_at_axis,
    visualize_radius_vs_height,
    visualize_circle_at_axis,
    visualize_top_circles,
)

# ARCore 좌표 변환
from .transforms import (
    transform_point_to_anchor_relative,
    transform_point_to_arcore,
    transform_point_from_arcore,
)

__all__ = [
    # 상수
    'THICKNESS', 'MIN_POINTS_PER_SLICE', 'MAX_CONSECUTIVE_FAILURES',
    'MIN_RADIUS', 'MAX_RADIUS',
    'WEIGHT_PPM_AXIS', 'WEIGHT_ANGLE_COVERAGE_AXIS', 'WEIGHT_GRID_COVERAGE_AXIS',
    'WEIGHT_INLIER_RATIO_AXIS', 'WEIGHT_RAY_DISTANCE_AXIS',
    'WEIGHT_PPM_DETECTION', 'WEIGHT_ANGLE_COVERAGE_DETECTION',
    'WEIGHT_GRID_COVERAGE_DETECTION', 'WEIGHT_INLIER_RATIO_DETECTION',
    'GRID_SIZE', 'INNER_RADIUS_RATIO_FOR_GRID', 'ANGLE_SEGMENTS',
    'PLANE_SEARCH_BELOW', 'PLANE_THICKNESS', 'PLANE_MIN_POINTS',
    'AXIS_SEARCH_RANGE', 'AXIS_SEARCH_STEP',
    
    # 평가 지표
    'calculate_ppm', 'calculate_angle_coverage', 'calculate_grid_coverage',
    
    # 기하학
    'get_slice', 'refine_circle_ls', 'fit_circle_ransac',
    'detect_circle_at_axis', 'detect_plane_below_point',
    'compute_ray_intersection',
    
    # 파일 IO
    'load_and_align_scene', 'load_jsonl', 'get_image_center_ray',
    
    # 부피
    'add_volume_data', 'search_heights',
    
    # 시각화
    'visualize_volume_3d', 'visualize_detect_circle_at_axis',
    'visualize_radius_vs_height', 'visualize_circle_at_axis',
    'visualize_top_circles',
    
    # 좌표 변환
    'transform_point_to_anchor_relative', 'transform_point_to_arcore',
    'transform_point_from_arcore',
]
