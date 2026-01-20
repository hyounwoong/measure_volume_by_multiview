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
