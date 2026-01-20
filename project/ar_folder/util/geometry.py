"""
기하학 계산 함수 모듈
"""
import numpy as np
from scipy.optimize import least_squares
from .constants import (
    THICKNESS, MIN_POINTS_PER_SLICE, MIN_RADIUS, MAX_RADIUS,
    WEIGHT_ANGLE_COVERAGE_AXIS, WEIGHT_GRID_COVERAGE_AXIS,
    WEIGHT_INLIER_RATIO_AXIS, WEIGHT_RAY_DISTANCE_AXIS,
    WEIGHT_PPM_DETECTION, WEIGHT_ANGLE_COVERAGE_DETECTION,
    WEIGHT_GRID_COVERAGE_DETECTION, WEIGHT_INLIER_RATIO_DETECTION,
    PLANE_SEARCH_BELOW, PLANE_THICKNESS, PLANE_MIN_POINTS
)
from .metrics import calculate_ppm, calculate_angle_coverage, calculate_grid_coverage


def get_slice(points, z, thickness=THICKNESS):
    """높이 z에서 슬라이스 추출"""
    mask = (points[:, 2] >= z - thickness) & (points[:, 2] <= z + thickness)
    return points[mask]


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


def fit_circle_ransac(points_2d, n_iter=200, threshold=0.005, min_inliers=10, ray_intersection_2d=None):
    """랜덤 샘플링을 통해 원을 추정하는 함수 (레이 교차점 거리 포함)"""
    if len(points_2d) < 3:
        return None
    
    # 중심축 선정용 가중치 정규화
    sum_weights = WEIGHT_ANGLE_COVERAGE_AXIS + WEIGHT_GRID_COVERAGE_AXIS + WEIGHT_INLIER_RATIO_AXIS + WEIGHT_RAY_DISTANCE_AXIS
    if sum_weights > 0:
        weight_angle_norm = WEIGHT_ANGLE_COVERAGE_AXIS / sum_weights
        weight_grid_norm = WEIGHT_GRID_COVERAGE_AXIS / sum_weights
        weight_inlier_norm = WEIGHT_INLIER_RATIO_AXIS / sum_weights
        weight_ray_norm = WEIGHT_RAY_DISTANCE_AXIS / sum_weights
    else:
        weight_angle_norm = weight_grid_norm = weight_inlier_norm = weight_ray_norm = 0.0
    
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
                
                # 레이 교차점과 원 중심 거리 계산 (정규화: 0~1, 가까울수록 높은 값)
                ray_distance_score = 1.0
                if ray_intersection_2d is not None:
                    dist_to_ray = np.linalg.norm(refined['center'] - ray_intersection_2d)
                    # 최대 거리 0.1m로 정규화 (0.1m 이상이면 0)
                    ray_distance_score = max(0.0, 1.0 - dist_to_ray / 0.1)
                
                score = ((inlier_ratio + 1e-8) ** weight_inlier_norm * 
                        (angle_coverage + 1e-8) ** weight_angle_norm * 
                        (grid_coverage_inverted + 1e-8) ** weight_grid_norm *
                        (ray_distance_score + 1e-8) ** weight_ray_norm)
                
                if score > best_score:
                    best_score = score
                    ppm = calculate_ppm(inlier_points, refined['radius'])
                    best_circle = {
                        **refined, 
                        'score': inlier_ratio,
                        'n_inliers': n_inliers,
                        'ppm': ppm,
                        'angle_coverage': angle_coverage,
                        'grid_coverage': grid_coverage,
                        'ray_distance': dist_to_ray if ray_intersection_2d is not None else None
                    }
        except np.linalg.LinAlgError:
            continue
    return best_circle


def detect_circle_at_axis(slice_2d, center_axis, n_iter=30, threshold=0.0025, prefer_inner=True, max_radius=None, min_score=0):
    """
    중심축 기반으로 원 검출 함수 (점수 기반)
    
    Args:
        slice_2d: 2D 단면 포인트 배열
        center_axis: 중심축 좌표 (x, y)
        n_iter: 반복 횟수
        threshold: 인라이어 임계값
        prefer_inner: 작은 반지름 선호 여부
        max_radius: (사용 안 함, 하위 호환성 유지용)
        min_score: 최소 점수 임계값 (이 값 이상이어야 원으로 인정)
    
    Returns:
        반지름 (float) 또는 None
    """
    if len(slice_2d) < MIN_POINTS_PER_SLICE:
        return None
    
    distances = np.linalg.norm(slice_2d - center_axis, axis=1)
    
    # 필터링 없이 모든 포인트 사용
    filtered_slice_2d = slice_2d
    filtered_distances = distances
    
    # 가중치 정규화
    sum_weights = (WEIGHT_PPM_DETECTION + WEIGHT_ANGLE_COVERAGE_DETECTION + 
                   WEIGHT_GRID_COVERAGE_DETECTION + WEIGHT_INLIER_RATIO_DETECTION)
    
    if sum_weights > 0:
        weight_ppm_norm = WEIGHT_PPM_DETECTION / sum_weights
        weight_angle_norm = WEIGHT_ANGLE_COVERAGE_DETECTION / sum_weights
        weight_grid_norm = WEIGHT_GRID_COVERAGE_DETECTION / sum_weights
        weight_inlier_norm = WEIGHT_INLIER_RATIO_DETECTION / sum_weights
    else:
        # 가중치가 모두 0이면 기본값 사용 (PPM만)
        weight_ppm_norm = 1.0
        weight_angle_norm = 0.0
        weight_grid_norm = 0.0
        weight_inlier_norm = 0.0
    
    # PPM 정규화를 위한 범위 계산 (모든 후보에 대해)
    all_ppm_candidates = []
    for _ in range(min(n_iter * 2, len(filtered_slice_2d))):
        r_candidate = filtered_distances[np.random.randint(len(filtered_slice_2d))]
        if MIN_RADIUS <= r_candidate <= MAX_RADIUS:
            inlier_count = np.sum(np.abs(filtered_distances - r_candidate) < threshold)
            circumference = 2 * np.pi * r_candidate
            ppm = inlier_count / circumference if circumference > 0 else 0
            all_ppm_candidates.append(ppm)
    
    ppm_min = min(all_ppm_candidates) if all_ppm_candidates else 0.0
    ppm_max = max(all_ppm_candidates) if all_ppm_candidates else 1.0
    
    best_radius, best_final_score = None, -1.0
    
    for _ in range(n_iter):
        r_candidate = filtered_distances[np.random.randint(len(filtered_slice_2d))]
        if not (MIN_RADIUS <= r_candidate <= MAX_RADIUS):
            continue
        
        # 인라이어 포인트 추출
        inlier_mask = np.abs(filtered_distances - r_candidate) < threshold
        inlier_points = filtered_slice_2d[inlier_mask]
        
        if len(inlier_points) < MIN_POINTS_PER_SLICE:
            continue
        
        # 각 지표 계산
        inlier_ratio = len(inlier_points) / len(filtered_slice_2d)
        
        circumference = 2 * np.pi * r_candidate
        ppm = len(inlier_points) / circumference if circumference > 0 else 0
        ppm_norm = (ppm - ppm_min) / (ppm_max - ppm_min + 1e-8) if ppm_max > ppm_min else 0.0
        
        angle_coverage = calculate_angle_coverage(inlier_points, center_axis)
        grid_coverage = calculate_grid_coverage(filtered_slice_2d, center_axis, r_candidate)
        grid_coverage_inverted = 1.0 - grid_coverage
        
        # 최종 점수 계산 (가중치 적용)
        final_score = ((ppm_norm + 1e-8) ** weight_ppm_norm * 
                      (inlier_ratio + 1e-8) ** weight_inlier_norm * 
                      (angle_coverage + 1e-8) ** weight_angle_norm * 
                      (grid_coverage_inverted + 1e-8) ** weight_grid_norm)
        
        # prefer_inner 옵션: 작은 반지름에 추가 가중치
        if prefer_inner:
            radius_weight = 1.0 + (MAX_RADIUS - r_candidate) / MAX_RADIUS
            final_score *= radius_weight
        
        if final_score > best_final_score:
            best_final_score = final_score
            best_radius = r_candidate
    
    # 최종 검증: 점수가 임계값 이상이어야 함
    if best_radius and best_final_score >= min_score:
        return best_radius
    
    return None


def detect_plane_below_point(points, point_z, search_below=PLANE_SEARCH_BELOW, thickness=PLANE_THICKNESS, min_points=PLANE_MIN_POINTS):
    """특정 점 아래에서 평면 검출 (RANSAC 기반)"""
    # 검색 범위
    z_min = point_z - search_below
    z_max = point_z
    
    # 범위 내 포인트 필터링
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    search_points = points[mask]
    
    if len(search_points) < min_points:
        return None, None
    
    # RANSAC으로 평면 검출
    best_plane_z = None
    best_inlier_count = 0
    n_iter = 100
    threshold = thickness / 2.0
    
    for _ in range(n_iter):
        # 3개 점 랜덤 선택
        sample_indices = np.random.choice(len(search_points), 3, replace=False)
        p1, p2, p3 = search_points[sample_indices]
        
        # 평면 방정식: ax + by + cz + d = 0
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        if np.linalg.norm(normal) < 1e-6:
            continue
        
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        
        # 모든 점에서 평면까지 거리 계산
        distances = np.abs(np.dot(search_points, normal) + d)
        inliers = distances < threshold
        n_inliers = np.sum(inliers)
        
        if n_inliers > best_inlier_count:
            best_inlier_count = n_inliers
            # 평면의 평균 Z값 계산
            inlier_points = search_points[inliers]
            best_plane_z = np.mean(inlier_points[:, 2])
    
    if best_inlier_count >= min_points:
        return best_plane_z, best_inlier_count
    
    return None, None


def compute_ray_intersection(ray_origins, ray_directions):
    """
    여러 레이들의 최근접 교차점을 least squares로 계산
    
    수학적 원리:
    - 각 레이: P(t) = origin + t * direction
    - 목표: 모든 레이에서 거리의 제곱합을 최소화하는 점 찾기
    - 최소화: Σ ||point - (origin_i + t_i * direction_i)||²
    
    해결 방법:
    1. 각 레이에 수직인 평면에 투영
    2. 투영 행렬: P = I - d*d^T (d는 방향 벡터)
    3. 최소 제곱법: A @ x = b 형태로 변환
    4. 해: x = (A^T @ A)^(-1) @ A^T @ b
    
    Args:
        ray_origins: 레이 원점들 (N x 3)
        ray_directions: 레이 방향 벡터들 (N x 3, 정규화됨)
    
    Returns:
        intersection_point: 교차점 (3,)
        avg_distance: 평균 거리 오차
        individual_distances: 각 레이에서의 거리 오차
    """
    n_rays = len(ray_origins)
    
    # 각 레이에 대해: point = origin + t * direction
    # 최소화할 목표: sum of squared distances from point to each ray
    
    # A @ x = b 형태로 변환
    # A: (3N x 3) 행렬
    # b: (3N,) 벡터
    
    A = np.zeros((3 * n_rays, 3))
    b = np.zeros(3 * n_rays)
    
    for i in range(n_rays):
        origin = ray_origins[i]
        direction = ray_directions[i]
        
        # I - d*d^T 형태의 투영 행렬
        proj = np.eye(3) - np.outer(direction, direction)
        
        A[3*i:3*(i+1)] = proj
        b[3*i:3*(i+1)] = proj @ origin
    
    # 최소 제곱법으로 해결
    intersection = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # 각 레이에서의 거리 오차 계산
    distances = []
    closest_points = []
    for i in range(n_rays):
        origin = ray_origins[i]
        direction = ray_directions[i]
        
        # 교차점에서 레이까지의 최단 거리
        vec = intersection - origin
        proj_length = np.dot(vec, direction)
        closest_point = origin + direction * proj_length
        dist = np.linalg.norm(intersection - closest_point)
        distances.append(dist)
        closest_points.append(closest_point)
    
    avg_distance = np.mean(distances)
    closest_points = np.array(closest_points)
    
    return intersection, avg_distance, distances, closest_points
