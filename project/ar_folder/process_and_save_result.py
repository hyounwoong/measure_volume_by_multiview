"""
GPU 서버에서 실행할 스크립트
inference_with_pose.py를 실행하고 결과를 JSON으로 저장
부피 계산 및 회전축 좌표 계산 포함
"""
import sys
import os
import json
import glob
import numpy as np
import time
from pathlib import Path

# inference_with_pose.py의 로직을 재사용
sys.path.insert(0, str(Path(__file__).parent))

from inference_with_pose import (
    load_jsonl,
    extract_frame_number,
    get_frame_number_from_timestamp,
    SAMPLING_INTERVAL_NS
)
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from depth_anything_3.api import DepthAnything3
import trimesh

# ============================================
# 상수 정의 (부피 계산용)
# ============================================
THICKNESS = 0.0003  # 단면의 두께
MIN_POINTS_PER_SLICE = 10  # 원 검출 시 최소 포인트 수
MAX_CONSECUTIVE_FAILURES = 10  # 원 검출 시 최대 연속 실패 횟수
MIN_RADIUS, MAX_RADIUS = 0.02, 0.1  # 원 검출 시 최소 및 최대 반지름
MIN_POINTS_PER_METER = 50.0  # 중심축 기반 원검출시 최소 포인트 수
MIN_ANGLE_COVERAGE_THRESHOLD = 0.2  # 원 검출 시 최소 각도 커버리지

# 바닥 검출 파라미터
BOTTOM_SEARCH_RADIUS = 0.02  # 중심축으로부터 2cm 범위
BOTTOM_GRID_SIZE = 20  # 그리드 20x20 = 400개 셀
MIN_BOTTOM_GRID_COVERAGE = 0.2  # 바닥 검출을 위한 최소 그리드 커버리지
BOTTOM_SEARCH_RATIO = 0.3  # 하위 30%만 바닥 검출

# 중심축 선정을 위한 지표 가중치
WEIGHT_PPM = 0.3
WEIGHT_ANGLE_COVERAGE = 0.4
WEIGHT_GRID_COVERAGE = 0
WEIGHT_INLIER_RATIO = 0.3

# 그리드 기반 커버리지 계산 파라미터
GRID_SIZE = 40
INNER_RADIUS_RATIO_FOR_GRID = 0.8  # 원 반지름의 80% 이내

# 각도 커버리지 계산 파라미터
ANGLE_SEGMENTS = 180

# ============================================
# GLB 로드 및 정렬 함수
# ============================================
def load_and_align_scene(filepath):
    """GLB 파일을 로드하고 중력 방향으로 정렬"""
    scene = trimesh.load(filepath)
    A = np.array(scene.metadata.get('hf_alignment', np.eye(4))).reshape(4, 4)
    gravity = (A[:3, :3] @ np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ np.array([0,1,0]))
    gravity = gravity / np.linalg.norm(gravity)
    
    R = Rotation.align_vectors([[0, 0, 1]], [gravity])[0]
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    for geom in scene.geometry.values():
        geom.apply_transform(T)
    
    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.PointCloud):
            return geometry.vertices, scene.metadata
    return None, None

# ============================================
# 원 검출 관련 함수들
# ============================================
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

def refine_circle_ls(inlier_points):
    """인라이너 포인트들을 받아서 원의 중심과 반지름을 정확하게 추정"""
    if len(inlier_points) < 3:
        return None
    center_init = inlier_points.mean(axis=0)
    radius_init = np.mean(np.linalg.norm(inlier_points - center_init, axis=1))
    
    def residuals(params):
        cx, cy, r = params
        return np.linalg.norm(inlier_points - [cx, cy], axis=1) - r
    
    try:
        result = least_squares(residuals, [center_init[0], center_init[1], radius_init],
                              bounds=([-0.5, -0.5, 0.001], [0.5, 0.5, 0.2]))
        cx, cy, r = result.x
        if 0.01 < r < 0.15:
            return {'center': np.array([cx, cy]), 'radius': r}
    except:
        pass
    return None

def fit_circle_ransac(points_2d, n_iter=200, threshold=0.005, min_inliers=10):
    """랜덤 샘플링을 통해 원을 추정하는 함수"""
    if len(points_2d) < 3:
        return None
    
    sum_without_ppm = WEIGHT_ANGLE_COVERAGE + WEIGHT_GRID_COVERAGE + WEIGHT_INLIER_RATIO
    weight_angle_norm = WEIGHT_ANGLE_COVERAGE / sum_without_ppm
    weight_grid_norm = WEIGHT_GRID_COVERAGE / sum_without_ppm
    weight_inlier_norm = WEIGHT_INLIER_RATIO / sum_without_ppm
    
    best_circle, best_score = None, -1.0
    for _ in range(n_iter):
        p1, p2, p3 = points_2d[np.random.choice(len(points_2d), 3, replace=False)]
        try:
            A = np.array([[2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])],
                         [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1])]])
            b = np.array([p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2,
                         p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2])
            center = np.linalg.solve(A, b)
            radius = np.linalg.norm(p1 - center)
            
            if not (0.01 < radius < 0.15):
                continue
            
            distances = np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)
            inliers = distances < threshold
            n_inliers = np.sum(inliers)
            
            if n_inliers < min_inliers:
                continue
            
            refined = refine_circle_ls(points_2d[inliers])
            if refined:
                inlier_points = points_2d[inliers]
                inlier_ratio = n_inliers / len(points_2d)
                
                angle_coverage = calculate_angle_coverage(inlier_points, refined['center'])
                grid_coverage = calculate_grid_coverage(points_2d, refined['center'], refined['radius'])
                grid_coverage_inverted = 1.0 - grid_coverage
                
                score = ((inlier_ratio + 1e-8) ** weight_inlier_norm * 
                        (angle_coverage + 1e-8) ** weight_angle_norm * 
                        (grid_coverage_inverted + 1e-8) ** weight_grid_norm)
                
                if score > best_score:
                    best_score = score
                    ppm = calculate_ppm(inlier_points, refined['radius'])
                    best_circle = {
                        **refined, 
                        'score': inlier_ratio,
                        'n_inliers': n_inliers,
                        'ppm': ppm,
                        'angle_coverage': angle_coverage,
                        'grid_coverage': grid_coverage
                    }
        except np.linalg.LinAlgError:
            continue
    return best_circle

def detect_circle_at_axis(slice_2d, center_axis, n_iter=100, threshold=0.0025, prefer_inner=True, max_radius=None):
    """중심축 기반으로 원 검출 함수"""
    if len(slice_2d) < MIN_POINTS_PER_SLICE:
        return None
    
    distances = np.linalg.norm(slice_2d - center_axis, axis=1)
    
    if max_radius is not None:
        filtered_mask = distances <= max_radius
        filtered_slice_2d = slice_2d[filtered_mask]
        filtered_distances = distances[filtered_mask]
        
        if len(filtered_slice_2d) < MIN_POINTS_PER_SLICE:
            return None
    else:
        filtered_slice_2d = slice_2d
        filtered_distances = distances
    
    best_radius, best_score = None, -1.0
    
    for _ in range(n_iter):
        r_candidate = filtered_distances[np.random.randint(len(filtered_slice_2d))]
        if not (MIN_RADIUS <= r_candidate <= MAX_RADIUS):
            continue
        
        inlier_count = np.sum(np.abs(filtered_distances - r_candidate) < threshold)
        circumference = 2 * np.pi * r_candidate
        ppm = inlier_count / circumference if circumference > 0 else 0
        
        if prefer_inner:
            radius_weight = 1.0 + (MAX_RADIUS - r_candidate) / MAX_RADIUS
            score = ppm * radius_weight
        else:
            score = ppm
        
        if score > best_score:
            best_score, best_radius = score, r_candidate
    
    if best_radius:
        inlier_mask = np.abs(filtered_distances - best_radius) < threshold
        inlier_points = filtered_slice_2d[inlier_mask]
        
        if len(inlier_points) >= MIN_POINTS_PER_SLICE:
            angle_coverage = calculate_angle_coverage(inlier_points, center_axis)
            
            if angle_coverage >= MIN_ANGLE_COVERAGE_THRESHOLD:
                return best_radius
    
    return None

def check_has_bottom(slice_2d, center_axis, radius, threshold=0.0003):
    """바닥이 있는지 확인하는 함수 (그리드 기반 커버리지 방식)"""
    if len(slice_2d) < MIN_POINTS_PER_SLICE:
        return False
    
    distances = np.linalg.norm(slice_2d - center_axis, axis=1)
    bottom_mask = distances <= BOTTOM_SEARCH_RADIUS
    bottom_points = slice_2d[bottom_mask]
    
    if len(bottom_points) < MIN_POINTS_PER_SLICE:
        return False
    
    x_min = center_axis[0] - BOTTOM_SEARCH_RADIUS
    x_max = center_axis[0] + BOTTOM_SEARCH_RADIUS
    y_min = center_axis[1] - BOTTOM_SEARCH_RADIUS
    y_max = center_axis[1] + BOTTOM_SEARCH_RADIUS
    
    cell_size_x = (x_max - x_min) / BOTTOM_GRID_SIZE
    cell_size_y = (y_max - y_min) / BOTTOM_GRID_SIZE
    
    if cell_size_x <= 0 or cell_size_y <= 0:
        return False
    
    grid_x = ((bottom_points[:, 0] - x_min) / cell_size_x).astype(int)
    grid_y = ((bottom_points[:, 1] - y_min) / cell_size_y).astype(int)
    
    grid_x = np.clip(grid_x, 0, BOTTOM_GRID_SIZE - 1)
    grid_y = np.clip(grid_y, 0, BOTTOM_GRID_SIZE - 1)
    
    occupied_cells = set(zip(grid_x, grid_y))
    total_cells = BOTTOM_GRID_SIZE * BOTTOM_GRID_SIZE
    grid_coverage = len(occupied_cells) / total_cells
    
    has_bottom = grid_coverage >= MIN_BOTTOM_GRID_COVERAGE
    
    return has_bottom

def get_slice(points, z, thickness=THICKNESS):
    """높이 z에서 슬라이스 추출"""
    mask = (points[:, 2] >= z - thickness) & (points[:, 2] <= z + thickness)
    return points[mask]

def add_volume_data(volume_data, z, radius):
    """volume_data에 단면 정보 추가"""
    volume_data.append({'z': z, 'radius': radius, 'area': np.pi * radius**2})

def search_heights(points, z_range, center_axis, volume_data, max_failures=MAX_CONSECUTIVE_FAILURES, max_radius=None):
    """높이 범위를 탐색하며 원 검출"""
    failures = 0
    for z in z_range:
        slice_points = get_slice(points, z)
        if len(slice_points) < MIN_POINTS_PER_SLICE:
            failures += 1
            if failures >= max_failures:
                break
            continue
        
        radius = detect_circle_at_axis(slice_points[:, :2], center_axis, max_radius=max_radius)
        if radius is not None:
            failures = 0
            add_volume_data(volume_data, z, radius)
        else:
            failures += 1
            if failures >= max_failures:
                break

# ============================================
# ARCore 좌표 변환 함수
# ============================================
def transform_point_to_anchor_relative(point_scene, first_frame_params, scene_metadata=None):
    """
    중력 정렬 씬 좌표계의 점을 앵커 기준 상대 좌표로 변환
    
    변환 경로:
    1. 중력 정렬 씬 → hf_alignment 적용 씬 (T_inv)
    2. hf_alignment 적용 씬 → 첫 프레임 카메라 기준 좌표계 (T_camera_origin)
    3. 첫 프레임 카메라 기준 좌표계 → 앵커 기준 상대 좌표
    
    Args:
        point_scene: 중력 정렬 씬 좌표계의 점 [x, y, z]
        first_frame_params: 첫 프레임의 ARCore pose 정보 (quat, pos, anchor_pos, anchor_quat 포함)
        scene_metadata: 씬 메타데이터 (hf_alignment 포함)
    
    Returns:
        앵커 기준 상대 좌표계의 점 [x, y, z] (OpenGL 좌표계)
    """
    point_scene = np.array(point_scene, dtype=np.float32)
    
    if scene_metadata is None:
        raise ValueError("scene_metadata is required")
    
    A = np.array(scene_metadata.get('hf_alignment', np.eye(4))).reshape(4, 4)
    
    # 중력 벡터 계산 (hf_alignment 적용 씬 좌표계에서)
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    arcore_y = np.array([0, 1, 0])
    gravity = (A[:3, :3] @ gl_to_cv @ arcore_y)
    gravity = gravity / np.linalg.norm(gravity)
    
    # 중력 정렬 회전 행렬 계산
    R = Rotation.align_vectors([[0, 0, 1]], [gravity])[0]
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T_inv = np.linalg.inv(T)
    
    # 1단계: 중력 정렬 역변환 (중력 정렬 씬 → hf_alignment 적용 씬)
    point_after_inv = T_inv[:3, :3] @ point_scene + T_inv[:3, 3]
    
    # 첫 프레임 카메라의 c2w 계산
    quat = np.array(first_frame_params['quat'])
    pos_arcore = np.array(first_frame_params['pos'])
    
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R_c2w_gl = Rotation.from_quat(quat).as_matrix()
    R_c2w_cv = gl_to_cv @ R_c2w_gl @ gl_to_cv.T
    pos_cv = gl_to_cv @ pos_arcore
    
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w_cv
    c2w[:3, 3] = pos_cv
    
    w2c0 = np.linalg.inv(c2w)
    
    # M 행렬 (CV → glTF 좌표계 변환)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0
    M[2, 2] = -1.0
    M_inv = np.linalg.inv(M)
    
    A_no_center = M @ w2c0.astype(np.float64)
    A_no_center_inv = np.linalg.inv(A_no_center)
    
    # c2w0 계산 (원본 씬 좌표계에서의 첫 프레임 카메라 → 월드)
    c2w0 = A_no_center_inv @ M
    
    # 첫 프레임 카메라 위치 (hf_alignment 적용 씬 좌표계에서)
    first_camera_pos_original = c2w0[:3, 3]
    first_camera_pos_hf_aligned = A[:3, :3] @ first_camera_pos_original + A[:3, 3]
    
    # 2단계: 첫 프레임 카메라 기준 좌표계로 변환 (glTF 좌표계)
    point_camera_frame_gltf = point_after_inv - first_camera_pos_hf_aligned
    
    # glTF → OpenCV 좌표계 변환 (c2w가 OpenCV 기준이므로)
    point_camera_frame_cv = M_inv[:3, :3] @ point_camera_frame_gltf
    
    # 3단계: ARCore 월드 좌표계로 변환 (OpenCV 좌표계 기준)
    point_arcore_world_cv = c2w[:3, :3] @ point_camera_frame_cv + c2w[:3, 3]
    
    # OpenCV → OpenGL 좌표계 변환 (ARCore 월드 좌표계)
    cv_to_gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    point_arcore_world_gl = cv_to_gl @ point_arcore_world_cv
    
    # 4단계: 앵커 기준 상대 좌표로 변환
    anchor_pos = np.array(first_frame_params.get('anchor_pos', first_frame_params.get('pos', [0.0, 0.0, 0.0])), dtype=np.float32)
    anchor_quat = np.array(first_frame_params.get('anchor_quat', first_frame_params.get('quat', [0.0, 0.0, 0.0, 1.0])), dtype=np.float32)
    anchor_rotation = Rotation.from_quat(anchor_quat).as_matrix()
    
    # 월드 좌표계에서 앵커 기준 상대 좌표로 변환
    point_relative = point_arcore_world_gl - anchor_pos
    point_anchor_relative = anchor_rotation.T @ point_relative
    
    return point_anchor_relative

def transform_point_to_arcore(point_scene, first_frame_params, scene_metadata=None):
    """
    중력 정렬 씬 좌표계의 점을 ARCore 월드 좌표계로 변환 (기존 함수, 호환성 유지)
    
    변환 경로:
    1. 중력 정렬 씬 → hf_alignment 적용 씬 (T_inv)
    2. hf_alignment 적용 씬 → 첫 프레임 카메라 기준 좌표계 (T_camera_origin)
    3. 첫 프레임 카메라 기준 좌표계 → ARCore 월드 좌표계 (c2w)
    
    Args:
        point_scene: 중력 정렬 씬 좌표계의 점 [x, y, z]
        first_frame_params: 첫 프레임의 ARCore pose 정보 (quat, pos 포함)
        scene_metadata: 씬 메타데이터 (hf_alignment 포함)
    
    Returns:
        ARCore 월드 좌표계의 점 [x, y, z] (OpenGL 좌표계)
    """
    point_scene = np.array(point_scene, dtype=np.float32)
    
    if scene_metadata is None:
        raise ValueError("scene_metadata is required")
    
    A = np.array(scene_metadata.get('hf_alignment', np.eye(4))).reshape(4, 4)
    
    # 중력 벡터 계산 (hf_alignment 적용 씬 좌표계에서)
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    arcore_y = np.array([0, 1, 0])
    gravity = (A[:3, :3] @ gl_to_cv @ arcore_y)
    gravity = gravity / np.linalg.norm(gravity)
    
    # 중력 정렬 회전 행렬 계산
    R = Rotation.align_vectors([[0, 0, 1]], [gravity])[0]
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T_inv = np.linalg.inv(T)
    
    # 1단계: 중력 정렬 역변환 (중력 정렬 씬 → hf_alignment 적용 씬)
    point_after_inv = T_inv[:3, :3] @ point_scene + T_inv[:3, 3]
    
    # 첫 프레임 카메라의 c2w 계산
    quat = np.array(first_frame_params['quat'])
    pos_arcore = np.array(first_frame_params['pos'])
    
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R_c2w_gl = Rotation.from_quat(quat).as_matrix()
    R_c2w_cv = gl_to_cv @ R_c2w_gl @ gl_to_cv.T
    pos_cv = gl_to_cv @ pos_arcore
    
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w_cv
    c2w[:3, 3] = pos_cv
    
    w2c0 = np.linalg.inv(c2w)
    
    # M 행렬 (CV → glTF 좌표계 변환)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0
    M[2, 2] = -1.0
    M_inv = np.linalg.inv(M)
    
    A_no_center = M @ w2c0.astype(np.float64)
    A_no_center_inv = np.linalg.inv(A_no_center)
    
    # c2w0 계산 (원본 씬 좌표계에서의 첫 프레임 카메라 → 월드)
    c2w0 = A_no_center_inv @ M
    
    # 첫 프레임 카메라 위치 (hf_alignment 적용 씬 좌표계에서)
    first_camera_pos_original = c2w0[:3, 3]
    first_camera_pos_hf_aligned = A[:3, :3] @ first_camera_pos_original + A[:3, 3]
    
    # 2단계: 첫 프레임 카메라 기준 좌표계로 변환 (glTF 좌표계)
    point_camera_frame_gltf = point_after_inv - first_camera_pos_hf_aligned
    
    # glTF → OpenCV 좌표계 변환 (c2w가 OpenCV 기준이므로)
    point_camera_frame_cv = M_inv[:3, :3] @ point_camera_frame_gltf
    
    # 3단계: ARCore 월드 좌표계로 변환 (OpenCV 좌표계 기준)
    point_arcore_world_cv = c2w[:3, :3] @ point_camera_frame_cv + c2w[:3, 3]
    
    # OpenCV → OpenGL 좌표계 변환 (ARCore 월드 좌표계)
    cv_to_gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    point_arcore_world_gl = cv_to_gl @ point_arcore_world_cv
    
    return point_arcore_world_gl

def transform_point_from_arcore(point_arcore, first_frame_params, scene_metadata=None):
    """
    ARCore 월드 좌표계의 점을 중력 정렬 씬 좌표계로 변환 (transform_point_to_arcore의 역변환)
    
    변환 경로 (역순):
    1. ARCore 월드 좌표계 → 첫 프레임 카메라 기준 좌표계 (w2c)
    2. 첫 프레임 카메라 기준 좌표계 → hf_alignment 적용 씬 (T_camera_origin 역변환)
    3. hf_alignment 적용 씬 → 중력 정렬 씬 (T)
    
    Args:
        point_arcore: ARCore 월드 좌표계의 점 [x, y, z] (OpenGL 좌표계)
        first_frame_params: 첫 프레임의 ARCore pose 정보 (quat, pos 포함)
        scene_metadata: 씬 메타데이터 (hf_alignment 포함)
    
    Returns:
        중력 정렬 씬 좌표계의 점 [x, y, z]
    """
    point_arcore = np.array(point_arcore, dtype=np.float32)
    
    if scene_metadata is None:
        raise ValueError("scene_metadata is required")
    
    A = np.array(scene_metadata.get('hf_alignment', np.eye(4))).reshape(4, 4)
    
    # 중력 벡터 계산 (hf_alignment 적용 씬 좌표계에서)
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    arcore_y = np.array([0, 1, 0])
    gravity = (A[:3, :3] @ gl_to_cv @ arcore_y)
    gravity = gravity / np.linalg.norm(gravity)
    
    # 중력 정렬 회전 행렬 계산
    R = Rotation.align_vectors([[0, 0, 1]], [gravity])[0]
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T_inv = np.linalg.inv(T)
    
    # 첫 프레임 카메라의 c2w 계산 (원본 함수와 동일)
    quat = np.array(first_frame_params['quat'])
    pos_arcore_input = np.array(first_frame_params['pos'])
    
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R_c2w_gl = Rotation.from_quat(quat).as_matrix()
    R_c2w_cv = gl_to_cv @ R_c2w_gl @ gl_to_cv.T
    pos_cv = gl_to_cv @ pos_arcore_input
    
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w_cv
    c2w[:3, 3] = pos_cv
    
    w2c0 = np.linalg.inv(c2w)
    
    # M 행렬 (CV → glTF 좌표계 변환)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0
    M[2, 2] = -1.0
    M_inv = np.linalg.inv(M)
    
    A_no_center = M @ w2c0.astype(np.float64)
    A_no_center_inv = np.linalg.inv(A_no_center)
    
    # c2w0 계산 (원본 씬 좌표계에서의 첫 프레임 카메라 → 월드)
    c2w0 = A_no_center_inv @ M
    
    # 첫 프레임 카메라 위치 (hf_alignment 적용 씬 좌표계에서)
    first_camera_pos_original = c2w0[:3, 3]
    first_camera_pos_hf_aligned = A[:3, :3] @ first_camera_pos_original + A[:3, 3]
    
    # 3단계 역변환: ARCore 월드 좌표계 → 첫 프레임 카메라 기준 좌표계
    # OpenGL → OpenCV 좌표계 변환 (cv_to_gl의 역변환)
    cv_to_gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    # cv_to_gl은 대각 행렬이므로 전치가 역행렬
    point_arcore_world_cv = cv_to_gl.T @ point_arcore
    
    # ARCore 월드 (OpenCV) → 카메라 기준 (OpenCV 좌표계)
    # transform_point_to_arcore에서: point_arcore_world_cv = c2w[:3, :3] @ point_camera_frame_cv + c2w[:3, 3]
    # 역변환: point_camera_frame_cv = w2c[:3, :3] @ (point_arcore_world_cv - c2w[:3, 3])
    w2c = np.linalg.inv(c2w)
    point_camera_frame_cv = w2c[:3, :3] @ (point_arcore_world_cv - c2w[:3, 3])
    
    # OpenCV → glTF 좌표계 변환
    # transform_point_to_arcore에서: point_camera_frame_cv = M_inv[:3, :3] @ point_camera_frame_gltf
    # 역변환: point_camera_frame_gltf = M[:3, :3] @ point_camera_frame_cv
    point_camera_frame_gltf = M[:3, :3] @ point_camera_frame_cv
    
    # 2단계 역변환: 첫 프레임 카메라 기준 좌표계 → hf_alignment 적용 씬
    # transform_point_to_arcore에서: point_camera_frame_gltf = point_after_inv - first_camera_pos_hf_aligned
    # 역변환: point_after_inv = point_camera_frame_gltf + first_camera_pos_hf_aligned
    point_after_inv = point_camera_frame_gltf + first_camera_pos_hf_aligned
    
    # 1단계 역변환: hf_alignment 적용 씬 → 중력 정렬 씬
    # transform_point_to_arcore에서: point_after_inv = T_inv[:3, :3] @ point_scene + T_inv[:3, 3]
    # 역변환: point_after_inv - T_inv[:3, 3] = T_inv[:3, :3] @ point_scene
    #         point_scene = T[:3, :3] @ (point_after_inv - T_inv[:3, 3])
    point_scene = T[:3, :3] @ (point_after_inv - T_inv[:3, 3])
    
    return point_scene

# ============================================
# 컵 좌표 및 회전축 추출 함수
# ============================================
def calculate_rotation_axis(glb_path, jsonl_data, scene_metadata):
    """
    GLB 파일에서 중심축을 계산하고 바닥/상단 점을 ARCore 좌표계로 변환
    
    Returns:
        dict: {
            'bottom_point': [x, y, z],  # ARCore 좌표
            'top_point': [x, y, z],     # ARCore 좌표
            'center_axis': [x, y],      # 씬 좌표계 중심축
            'height_range': [z_min, z_max]  # 씬 좌표계 높이 범위
        }
    """
    try:
        points, metadata = load_and_align_scene(glb_path)
        if points is None or len(points) == 0:
            return None
        
        # 원점 근처 필터링
        mask = np.linalg.norm(points[:, :2], axis=1) < 0.5
        filtered = points[mask]
        
        if len(filtered) == 0:
            return None
        
        # 정렬
        sorted_idx = np.argsort(filtered[:, 2])
        sorted_points, sorted_heights = filtered[sorted_idx], filtered[sorted_idx, 2]
        
        # 1단계: 중심축 계산
        print(f"[GPU] [회전축 계산] 1단계: 중심축 계산 시작", flush=True)
        print(f"[GPU] [회전축 계산] 필터링된 점 수: {len(filtered)}, z 범위: [{sorted_heights.min():.6f}, {sorted_heights.max():.6f}]", flush=True)
        
        # z 범위를 균등하게 나누어 탐색
        z_range = np.linspace(sorted_heights.min(), sorted_heights.max(), 50)
        print(f"[GPU] [회전축 계산] z 범위 탐색: {len(z_range)}개 구간 (범위: [{z_range[0]:.6f}, {z_range[-1]:.6f}])", flush=True)
        
        circle_data = []
        
        print(f"[GPU] [회전축 계산] z 범위 탐색 시작: {len(z_range)}개 구간", flush=True)
        for z in z_range:
            start = np.searchsorted(sorted_heights, z - THICKNESS)
            end = np.searchsorted(sorted_heights, z + THICKNESS)
            if end - start < MIN_POINTS_PER_SLICE:
                continue
            
            slice_2d = sorted_points[start:end, :2]
            circle = fit_circle_ransac(slice_2d, n_iter=30, threshold=0.005, min_inliers=10)
            
            if circle and 'ppm' in circle and 'angle_coverage' in circle and 'grid_coverage' in circle:
                circle_data.append({
                    'z': z,
                    'center': circle['center'],
                    'radius': circle['radius'],
                    'score': circle['score'],
                    'ppm': circle['ppm'],
                    'angle_coverage': circle['angle_coverage'],
                    'grid_coverage': circle['grid_coverage']
                })
        
        print(f"[GPU] [회전축 계산] 원 검출 완료: {len(circle_data)}개 원 발견", flush=True)
        
        if not circle_data:
            print(f"[GPU] [회전축 계산] 오류: 원을 검출하지 못했습니다", flush=True)
            return None
        
        # 점수 계산 및 정렬
        all_ppms = [c['ppm'] for c in circle_data]
        ppm_min, ppm_max = min(all_ppms), max(all_ppms)
        
        print(f"[GPU] [회전축 계산] PPM 범위: [{ppm_min:.2f}, {ppm_max:.2f}]", flush=True)
        
        for circle in circle_data:
            ppm_norm = (circle['ppm'] - ppm_min) / (ppm_max - ppm_min + 1e-8)
            inlier_ratio = circle['score']
            angle_coverage = circle['angle_coverage']
            grid_coverage_inverted = 1.0 - circle['grid_coverage']
            
            final_score = ((ppm_norm + 1e-8) ** WEIGHT_PPM * 
                          (inlier_ratio + 1e-8) ** WEIGHT_INLIER_RATIO * 
                          (angle_coverage + 1e-8) ** WEIGHT_ANGLE_COVERAGE * 
                          (grid_coverage_inverted + 1e-8) ** WEIGHT_GRID_COVERAGE)
            
            circle['final_score'] = final_score
        
        circle_data.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 상위 5개 원 후보 정보 출력 (중심축 좌표 포함)
        print(f"[GPU] [회전축 계산] 상위 5개 원 후보:", flush=True)
        for i, circle in enumerate(circle_data[:5]):
            center = circle['center']
            print(f"[GPU]   [{i+1}] z={circle['z']:.6f}, 중심축 좌표=({center[0]:.6f}, {center[1]:.6f}), "
                  f"반지름={circle['radius']:.6f}, 최종점수={circle['final_score']:.6f}, "
                  f"ppm={circle['ppm']:.2f}, angle_cov={circle['angle_coverage']:.3f}, grid_cov={circle['grid_coverage']:.3f}", flush=True)
        
        best_circle = circle_data[0]
        center_axis = best_circle['center']
        reference_z = best_circle['z']
        reference_radius = best_circle['radius']
        max_radius_limit = reference_radius * 1.15
        
        print(f"[GPU] [회전축 계산] 최종 선택된 중심축:", flush=True)
        print(f"[GPU]   중심축 좌표 (씬 좌표계): ({center_axis[0]:.6f}, {center_axis[1]:.6f})", flush=True)
        print(f"[GPU]   기준 높이 (z): {reference_z:.6f}", flush=True)
        print(f"[GPU]   기준 반지름: {reference_radius:.6f}, 최대 반지름 제한: {max_radius_limit:.6f}", flush=True)
        print(f"[GPU]   최종 점수: {best_circle['final_score']:.6f}", flush=True)
        
        # 2단계: 부피 계산을 위한 원 검출
        print(f"[GPU] [회전축 계산] 2단계: 부피 계산을 위한 원 검출 시작", flush=True)
        volume_data = []
        z_min, z_max = filtered[:, 2].min(), filtered[:, 2].max()
        STEP = 0.002
        
        print(f"[GPU] [회전축 계산] 전체 z 범위: [{z_min:.6f}, {z_max:.6f}], 탐색 간격: {STEP:.6f}", flush=True)
        print(f"[GPU] [회전축 계산] 중심축 ({center_axis[0]:.6f}, {center_axis[1]:.6f}) 기준으로 원 검출", flush=True)
        
        ref_slice = get_slice(filtered, reference_z)
        if len(ref_slice) >= MIN_POINTS_PER_SLICE:
            ref_radius = detect_circle_at_axis(ref_slice[:, :2], center_axis, max_radius=max_radius_limit)
            if ref_radius:
                add_volume_data(volume_data, reference_z, ref_radius)
                print(f"[GPU] [회전축 계산] 기준 높이({reference_z:.6f})에서 원 검출: 반지름={ref_radius:.6f}", flush=True)
            else:
                print(f"[GPU] [회전축 계산] 기준 높이({reference_z:.6f})에서 원 검출 실패", flush=True)
        else:
            print(f"[GPU] [회전축 계산] 기준 높이({reference_z:.6f})에서 포인트 부족: {len(ref_slice)}개", flush=True)
        
        z_above = np.arange(reference_z + STEP, z_max + STEP, STEP)
        print(f"[GPU] [회전축 계산] 위쪽 탐색: {len(z_above)}개 구간", flush=True)
        search_heights(filtered, z_above, center_axis, volume_data, max_radius=max_radius_limit)
        
        z_below = np.arange(reference_z - STEP, z_min - STEP, -STEP)
        print(f"[GPU] [회전축 계산] 아래쪽 탐색: {len(z_below)}개 구간", flush=True)
        search_heights(filtered, z_below, center_axis, volume_data, max_radius=max_radius_limit)
        
        print(f"[GPU] [회전축 계산] 원 검출 완료: {len(volume_data)}개 단면 발견", flush=True)
        
        # 3단계: 바닥 검출 및 필터링
        print(f"[GPU] [회전축 계산] 3단계: 바닥 검출 및 필터링 시작", flush=True)
        if volume_data:
            volume_data.sort(key=lambda x: x['z'])
            
            height_range = volume_data[-1]['z'] - volume_data[0]['z']
            bottom_20_percent_range = height_range * BOTTOM_SEARCH_RATIO
            bottom_20_end_z = volume_data[0]['z'] + bottom_20_percent_range
            
            print(f"[GPU] [회전축 계산] 전체 높이 범위: {height_range:.6f}, 바닥 검색 범위: [{volume_data[0]['z']:.6f}, {bottom_20_end_z:.6f}]", flush=True)
            
            bottom_detected_heights = []
            for data in volume_data:
                z = data['z']
                if z <= bottom_20_end_z:
                    radius = data['radius']
                    slice_points = get_slice(filtered, z)
                    
                    if len(slice_points) >= MIN_POINTS_PER_SLICE:
                        has_bottom = check_has_bottom(slice_points[:, :2], center_axis, radius)
                        if has_bottom:
                            bottom_detected_heights.append(z)
            
            if bottom_detected_heights:
                bottom_end_z = max(bottom_detected_heights)
                print(f"[GPU] [회전축 계산] 바닥 검출: {len(bottom_detected_heights)}개 높이에서 바닥 발견, 최대 높이: {bottom_end_z:.6f}", flush=True)
                volume_data_filtered = [data for data in volume_data if data['z'] > bottom_end_z]
                
                if len(volume_data_filtered) > 1:
                    print(f"[GPU] [회전축 계산] 바닥 필터링: {len(volume_data)}개 → {len(volume_data_filtered)}개", flush=True)
                    volume_data = volume_data_filtered
                else:
                    print(f"[GPU] [회전축 계산] 바닥 필터링 후 데이터 부족, 필터링 취소", flush=True)
            else:
                print(f"[GPU] [회전축 계산] 바닥 검출 실패, 필터링 없이 진행", flush=True)
        
        if not volume_data:
            return None
        
        volume_data.sort(key=lambda x: x['z'])
        z_min_filtered = volume_data[0]['z']
        z_max_filtered = volume_data[-1]['z']
        
        print(f"[GPU] [회전축 계산] 필터링 후 z 범위: [{z_min_filtered:.6f}, {z_max_filtered:.6f}], 높이: {z_max_filtered - z_min_filtered:.6f}", flush=True)
        
        # 4단계: 두 점 계산 (씬 좌표계)
        print(f"[GPU] [회전축 계산] 4단계: 회전축 끝점 계산 (씬 좌표계)", flush=True)
        bottom_point_scene = np.array([center_axis[0], center_axis[1], z_min_filtered])
        top_point_scene = np.array([center_axis[0], center_axis[1], z_max_filtered])
        
        print(f"[GPU] [회전축 계산] 씬 좌표계 회전축 끝점:", flush=True)
        print(f"[GPU]   중심축 좌표 (x, y): ({center_axis[0]:.6f}, {center_axis[1]:.6f})", flush=True)
        print(f"[GPU]   하단점: ({bottom_point_scene[0]:.6f}, {bottom_point_scene[1]:.6f}, {bottom_point_scene[2]:.6f})", flush=True)
        print(f"[GPU]   상단점: ({top_point_scene[0]:.6f}, {top_point_scene[1]:.6f}, {top_point_scene[2]:.6f})", flush=True)
        
        # 5단계: ARCore 월드 좌표계로 변환 (첫 프레임 앵커 기준 월드 W*)
        # GPU에서 프레임 간 드리프트 보정이 이미 적용되어 있으므로,
        # 결과는 "첫 프레임 앵커 기준 월드 좌표"로 반환
        print(f"[GPU] [회전축 계산] 5단계: ARCore 월드 좌표계로 변환", flush=True)
        jsonl_data_sorted = sorted(jsonl_data, key=lambda x: x.get('t_ns', 0))
        first_frame_params = jsonl_data_sorted[0]
        
        bottom_point_world = transform_point_to_arcore(
            bottom_point_scene, first_frame_params, scene_metadata
        )
        top_point_world = transform_point_to_arcore(
            top_point_scene, first_frame_params, scene_metadata
        )
        
        print(f"[GPU] [회전축 계산] ARCore 좌표계 회전축 끝점:", flush=True)
        print(f"[GPU]   하단점: ({bottom_point_world[0]:.6f}, {bottom_point_world[1]:.6f}, {bottom_point_world[2]:.6f})", flush=True)
        print(f"[GPU]   상단점: ({top_point_world[0]:.6f}, {top_point_world[1]:.6f}, {top_point_world[2]:.6f})", flush=True)
        print(f"[GPU] [회전축 계산] 회전축 계산 완료", flush=True)
        
        return {
            'bottom_point': [float(bottom_point_world[0]), float(bottom_point_world[1]), float(bottom_point_world[2])],
            'top_point': [float(top_point_world[0]), float(top_point_world[1]), float(top_point_world[2])],
            'center_axis': [float(center_axis[0]), float(center_axis[1])],
            'height_range': [float(z_min_filtered), float(z_max_filtered)]
        }
    except Exception as e:
        print(f"회전축 계산 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_cup_coordinates(glb_path):
    """GLB 파일에서 컵의 중심 좌표를 추출 (기존 호환성 유지)"""
    try:
        points, _ = load_and_align_scene(glb_path)
        if points is None or len(points) == 0:
            return [0.0, 0.0, 0.0]
        
        mask = np.linalg.norm(points[:, :2], axis=1) < 0.5
        filtered = points[mask]
        
        if len(filtered) == 0:
            return [0.0, 0.0, 0.0]
        
        center_x = float(np.mean(filtered[:, 0]))
        center_y = float(np.mean(filtered[:, 1]))
        center_z = float(np.mean(filtered[:, 2]))
        
        return [center_x, center_y, center_z]
    except Exception as e:
        print(f"컵 좌표 추출 중 오류: {e}")
        return [0.0, 0.0, 0.0]

if __name__ == "__main__":
    start_time = time.time()
    
    # base_folder 설정 (GPU 서버의 작업 디렉토리)
    base_folder = sys.argv[1] if len(sys.argv) > 1 else "/data/ephemeral/home/measure_volume_by_multiview/project/ar_folder"
    print(f"[GPU] 작업 디렉토리: {base_folder}", flush=True)
    
    # 가장 최근 세션 폴더 찾기
    print(f"[GPU] 세션 폴더 검색 중...", flush=True)
    session_folders = sorted(glob.glob(os.path.join(base_folder, "session_*")))
    if not session_folders:
        print(json.dumps({"success": False, "error": "세션 폴더를 찾을 수 없습니다"}))
        sys.exit(1)
    
    session_folder = session_folders[-1]
    print(f"[GPU] 세션 폴더: {os.path.basename(session_folder)}", flush=True)
    
    jsonl_files = glob.glob(os.path.join(session_folder, "*.jsonl"))
    if not jsonl_files:
        print(json.dumps({"success": False, "error": "JSONL 파일을 찾을 수 없습니다"}))
        sys.exit(1)
    
    jsonl_file = jsonl_files[0]
    image_files = sorted(glob.glob(os.path.join(session_folder, "frame_*.jpg")))
    
    if not image_files:
        print(json.dumps({"success": False, "error": "이미지 파일을 찾을 수 없습니다"}))
        sys.exit(1)
    
    print(f"[GPU] 이미지 파일: {len(image_files)}개", flush=True)
    
    # JSONL 데이터 로드
    print(f"[GPU] JSONL 데이터 로드 중...", flush=True)
    jsonl_data = load_jsonl(jsonl_file)
    jsonl_data.sort(key=lambda x: x.get('t_ns', 0))
    print(f"[GPU] JSONL 레코드: {len(jsonl_data)}개", flush=True)
    
    # 이미지와 JSONL 데이터 매칭
    image_dict = {}
    for img_path in image_files:
        frame_num = extract_frame_number(os.path.basename(img_path))
        if frame_num is not None:
            image_dict[frame_num] = img_path
    
    matched_data = []
    for jsonl_item in jsonl_data:
        t_ns = jsonl_item.get('t_ns')
        if t_ns is None:
            continue
        
        frame_num = get_frame_number_from_timestamp(t_ns)
        if frame_num in image_dict:
            matched_data.append({
                'image': image_dict[frame_num],
                'params': jsonl_item,
                't_ns': t_ns
            })
    
    matched_data.sort(key=lambda x: x['t_ns'])
    
    if len(matched_data) == 0:
        print(json.dumps({"success": False, "error": "매칭된 데이터가 없습니다"}))
        sys.exit(1)
    
    print(f"[GPU] 매칭된 데이터: {len(matched_data)}개", flush=True)
    
    # 첫 프레임 앵커 정보 추출 (A_1)
    first_frame_params = matched_data[0]['params']
    first_anchor_pos = np.array(first_frame_params.get('anchor_pos', first_frame_params.get('pos', [0.0, 0.0, 0.0])), dtype=np.float32)
    first_anchor_quat = np.array(first_frame_params.get('anchor_quat', first_frame_params.get('quat', [0.0, 0.0, 0.0, 1.0])), dtype=np.float32)
    first_anchor_rotation = Rotation.from_quat(first_anchor_quat).as_matrix()
    print(f"[GPU] 첫 프레임 앵커: pos=({first_anchor_pos[0]:.3f}, {first_anchor_pos[1]:.3f}, {first_anchor_pos[2]:.3f})", flush=True)
    
    # 첫 프레임 앵커 변환 행렬 (A_1): OpenGL → OpenCV 좌표계
    gl_to_cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)
    
    first_anchor_rotation_cv = gl_to_cv @ first_anchor_rotation @ gl_to_cv.T
    first_anchor_pos_cv = gl_to_cv @ first_anchor_pos
    
    # A_1 행렬 (OpenCV 좌표계)
    A1 = np.eye(4, dtype=np.float32)
    A1[:3, :3] = first_anchor_rotation_cv
    A1[:3, 3] = first_anchor_pos_cv
    
    # 데이터 변환
    print(f"[GPU] 데이터 변환 중 (앵커 드리프트 보정 적용)...", flush=True)
    N = len(matched_data)
    extrinsics = np.zeros((N, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((N, 3, 3), dtype=np.float32)
    matched_image_files = []
    
    for i, item in enumerate(matched_data):
        params = item['params']
        img_path = item['image']
        
        jsonl_w = params.get('w')
        jsonl_h = params.get('h')
        if jsonl_w is None or jsonl_h is None:
            continue
        
        # 카메라 pose (P_i)
        quat = params.get('quat')
        pos = params.get('pos')
        if quat is None or pos is None:
            continue
        
        quat = np.array(quat, dtype=np.float32)
        pos = np.array(pos, dtype=np.float32)
        
        # 현재 프레임 앵커 pose (A_i)
        current_anchor_pos = np.array(params.get('anchor_pos', params.get('pos', [0.0, 0.0, 0.0])), dtype=np.float32)
        current_anchor_quat = np.array(params.get('anchor_quat', params.get('quat', [0.0, 0.0, 0.0, 1.0])), dtype=np.float32)
        current_anchor_rotation = Rotation.from_quat(current_anchor_quat).as_matrix()
        
        # A_i 행렬 (OpenCV 좌표계)
        current_anchor_rotation_cv = gl_to_cv @ current_anchor_rotation @ gl_to_cv.T
        current_anchor_pos_cv = gl_to_cv @ current_anchor_pos
        
        Ai = np.eye(4, dtype=np.float32)
        Ai[:3, :3] = current_anchor_rotation_cv
        Ai[:3, 3] = current_anchor_pos_cv
        
        # 보정 변환: C_i = A_1 * A_i^(-1)
        # 이 변환은 "현재 프레임의 월드 좌표계"를 "첫 프레임의 월드 좌표계"로 되돌림
        Ai_inv = np.linalg.inv(Ai)
        Ci = A1 @ Ai_inv
        
        # 카메라 pose를 OpenCV 좌표계로 변환
        R_c2w_gl = Rotation.from_quat(quat).as_matrix()
        R_c2w_cv = gl_to_cv @ R_c2w_gl @ gl_to_cv.T
        pos_cv = gl_to_cv @ pos
        
        # 원본 카메라 pose 행렬 (P_i): c2w (카메라 → 월드)
        Pi = np.eye(4, dtype=np.float32)
        Pi[:3, :3] = R_c2w_cv
        Pi[:3, 3] = pos_cv
        
        # 보정된 카메라 pose: P_i' = C_i * P_i
        # C_i는 월드 좌표계를 변환하므로, c2w에 적용
        Pi_corrected = Ci @ Pi
        
        # 보정된 pose에서 w2c 추출 (모델 입력용)
        w2c_corrected = np.linalg.inv(Pi_corrected)
        
        extrinsics[i] = w2c_corrected
        matched_image_files.append(img_path)
        
        # 디버깅: 첫 프레임과 마지막 프레임만 로그
        if i == 0 or i == len(matched_data) - 1:
            print(f"[GPU] Frame {i}: original pos=({pos_cv[0]:.3f}, {pos_cv[1]:.3f}, {pos_cv[2]:.3f}), "
                  f"corrected pos=({Pi_corrected[0,3]:.3f}, {Pi_corrected[1,3]:.3f}, {Pi_corrected[2,3]:.3f})", flush=True)
        
        fx = params.get('fx')
        fy = params.get('fy')
        cx = params.get('cx')
        cy = params.get('cy')
        
        if fx is None or fy is None or cx is None or cy is None:
            continue
        
        cx_colmap = cx - 0.5
        cy_colmap = cy - 0.5
        
        intrinsics[i] = np.array([
            [fx, 0.0, cx_colmap],
            [0.0, fy, cy_colmap],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    # DA3 추론
    print(f"[GPU] Depth-Anything-3 모델 로딩 중...", flush=True)
    model_load_start = time.time()
    project_root = os.path.dirname(base_folder)
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    model = DepthAnything3.from_pretrained("depth-anything/da3-base").to("cuda")
    model_load_time = time.time() - model_load_start
    print(f"[GPU] 모델 로딩 완료 ({model_load_time:.2f}초)", flush=True)
    
    print(f"[GPU] DA3 추론 시작 (이미지 {len(matched_image_files)}개)...", flush=True)
    inference_start = time.time()
    prediction = model.inference(
        image=matched_image_files,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        use_ray_pose=True,
        export_dir=output_dir,
        export_format="glb-mini_npz",
        process_res=756
    )
    print(f"[GPU] DA3 추론 완료 ({(time.time() - inference_start):.2f}초)", flush=True)
    
    # GLB 파일에서 컵 좌표 및 회전축 추출
    print(f"[GPU] GLB 파일 처리 중...", flush=True)
    glb_files = glob.glob(os.path.join(output_dir, "*.glb"))
    if glb_files:
        glb_path = glb_files[0]
        
        # GLB 메타데이터 로드
        scene = trimesh.load(glb_path)
        scene_metadata = scene.metadata
        
        # 기존 컵 좌표 (호환성 유지)
        print(f"[GPU] 컵 좌표 추출 중...", flush=True)
        cup_coordinates = extract_cup_coordinates(glb_path)
        
        # 회전축 계산
        print(f"[GPU] 회전축 계산 중...", flush=True)
        rotation_axis_data = calculate_rotation_axis(glb_path, jsonl_data, scene_metadata)
        
        if rotation_axis_data:
            result = {
                "success": True,
                "cup_coordinates": cup_coordinates,  # 기존 호환성
                "rotation_axis": {
                    "bottom_point": rotation_axis_data['bottom_point'],
                    "top_point": rotation_axis_data['top_point']
                },
                "message": "처리 완료"
            }
        else:
            # 회전축 계산 실패 시 기존 방식 사용
            result = {
                "success": True,
                "cup_coordinates": cup_coordinates,
                "rotation_axis": None,
                "message": "처리 완료 (회전축 계산 실패)"
            }
    else:
        # GLB 파일이 없으면 첫 번째 카메라 위치를 반환
        first_pos = matched_data[0]['params'].get('pos', [0.0, 0.0, 0.0])
        cup_coordinates = first_pos
        result = {
            "success": True,
            "cup_coordinates": cup_coordinates,
            "rotation_axis": None,
            "message": "GLB 파일을 찾을 수 없습니다"
        }
    
    # 결과를 파일로도 저장
    result_file = os.path.join(base_folder, f"{os.path.basename(session_folder)}_result.json")
    print(f"[GPU] 결과 저장 중: {result_file}", flush=True)
    with open(result_file, 'w') as f:
        json.dump(result, f)
    
    total_time = time.time() - start_time
    print(f"[GPU] 전체 처리 완료 (총 {total_time:.2f}초)", flush=True)
    
    # stdout으로도 출력 (로컬 PC에서 받을 수 있도록)
    print(json.dumps(result), flush=True)
