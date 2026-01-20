



# ============================================
# 상수 정의
# ============================================
THICKNESS = 0.0003  # 원검출에 사용할 단면의 두께
MIN_POINTS_PER_SLICE = 10  # 원 검출 시 최소 포인트 수
MAX_CONSECUTIVE_FAILURES = 10  # 원 검출 시 최대 연속 실패 횟수
MIN_RADIUS, MAX_RADIUS = 0.02, 0.1  # 원 검출 시 최소 및 최대 반지름

# 중심축 선정을 위한 지표 가중치
WEIGHT_PPM_AXIS = 0.5
WEIGHT_ANGLE_COVERAGE_AXIS = 0.0
WEIGHT_GRID_COVERAGE_AXIS = 0.0
WEIGHT_INLIER_RATIO_AXIS = 0.3
WEIGHT_RAY_DISTANCE_AXIS = 0.2  # 레이 교차점과 원 중심 거리 가중치

# 중심축 기반 원 검출을 위한 지표 가중치
WEIGHT_PPM_DETECTION = 0.3
WEIGHT_ANGLE_COVERAGE_DETECTION = 0.3
WEIGHT_GRID_COVERAGE_DETECTION = 0.0
WEIGHT_INLIER_RATIO_DETECTION = 0.4

# 그리드 기반 커버리지 계산 파라미터
GRID_SIZE = 40
INNER_RADIUS_RATIO_FOR_GRID = 0.8  # 원 반지름의 80% 이내

# 각도 커버리지 계산 파라미터
ANGLE_SEGMENTS = 180

# 평면 검출 파라미터 (레이 교차점 아래)
PLANE_SEARCH_BELOW = 0.30  # 레이 교차점 아래 30cm
PLANE_THICKNESS = 0.0005  # 0.5mm
PLANE_MIN_POINTS = 20  # 평면 검출 최소 포인트 수

# 중심축 탐색 파라미터
AXIS_SEARCH_RANGE = 0.30  # 레이 교차점 기준 위아래 30cm
AXIS_SEARCH_STEP = 0.005  # 5mm 간격



# ============================================
# 유틸리티 함수들
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

def get_slice(points, z, thickness=THICKNESS):
    """높이 z에서 슬라이스 추출"""
    mask = (points[:, 2] >= z - thickness) & (points[:, 2] <= z + thickness)
    return points[mask]

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

def detect_circle_at_axis(slice_2d, center_axis, n_iter=30, threshold=0.0025, prefer_inner=True, max_radius=None, min_score=0.0001):
    """
    중심축 기반으로 원 검출 함수 (점수 기반)
    
    Args:
        slice_2d: 2D 단면 포인트 배열
        center_axis: 중심축 좌표 (x, y)
        n_iter: 반복 횟수
        threshold: 인라이어 임계값
        prefer_inner: 작은 반지름 선호 여부
        max_radius: 최대 반지름 제한
        min_score: 최소 점수 임계값 (이 값 이상이어야 원으로 인정)
    
    Returns:
        반지름 (float) 또는 None
    """
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


# JSONL 파일 읽기 함수
def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        content = f.read()
    
    current_obj = ""
    brace_count = 0
    for char in content:
        current_obj += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    data.append(json.loads(current_obj.strip()))
                except json.JSONDecodeError:
                    pass
                current_obj = ""
    return data

# 이미지 중심을 통과하는 레이 계산 (원점과 방향 벡터)
def get_image_center_ray(pos, quat, fx, fy, cx, cy):
    """
    이미지 중심을 통과하는 레이 계산
    
    Args:
        pos: 카메라 위치 (world 좌표계)
        quat: 카메라 방향 (quaternion)
        fx, fy: 초점 거리
        cx, cy: 주점
    
    Returns:
        ray_origin: 레이 원점 (카메라 위치)
        ray_direction: 레이 방향 벡터 (정규화됨)
    """
    # 이미지 중심 좌표
    u = cx
    v = cy
    
    # 카메라 좌표계에서의 방향 벡터 (OpenGL 좌표계)
    # OpenGL에서 카메라는 -z 방향을 바라봄
    # 이미지 중심을 통과하는 레이는 카메라에서 앞으로 나가는 방향
    # OpenGL에서 카메라가 -z를 바라보므로, 앞쪽은 -z 방향 = [0, 0, -1]
    x_cam = (u - cx) / fx  # = 0 (이미지 중심이므로)
    y_cam = (v - cy) / fy  # = 0
    z_cam = -1.0  # OpenGL: 카메라가 -z를 바라보므로 앞쪽은 -z 방향
    
    # 정규화
    dir_cam = np.array([x_cam, y_cam, z_cam])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)
    
    # 카메라 좌표계에서 world 좌표계로 변환 (OpenGL 좌표계)
    # quat는 OpenGL 좌표계의 c2w 변환
    R_c2w_gl = Rotation.from_quat(quat).as_matrix()
    
    # OpenGL 좌표계에서 월드 좌표계로 방향 벡터 변환
    # pos와 같은 좌표계(OpenGL)로 유지
    dir_world = R_c2w_gl @ dir_cam
    
    return pos, dir_world

# 여러 레이들의 최근접 교차점 계산 (least squares)
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
    
    import matplotlib.pyplot as plt
    import numpy as np
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
        from mpl_toolkits.mplot3d import Axes3D
        
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