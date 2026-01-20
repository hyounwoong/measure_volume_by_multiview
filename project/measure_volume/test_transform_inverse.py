# 역변환 테스트 코드
# transform_point_to_arcore와 transform_point_from_arcore가 서로 역변환인지 테스트

import numpy as np

def test_transform_inverse(transform_point_to_arcore, transform_point_from_arcore, 
                          first_frame_params, scene_metadata, 
                          anchor_pos_scene=None, anchor_pos_arcore=None):
    """
    두 변환 함수가 서로 역변환인지 테스트
    
    Args:
        transform_point_to_arcore: 씬 좌표계 → ARCore 좌표계 변환 함수
        transform_point_from_arcore: ARCore 좌표계 → 씬 좌표계 변환 함수
        first_frame_params: 첫 프레임 파라미터
        scene_metadata: 씬 메타데이터
        anchor_pos_scene: (선택) 앵커 위치 (씬 좌표계)
        anchor_pos_arcore: (선택) 앵커 위치 (ARCore 좌표계)
    """
    
    # 테스트 1: 씬 좌표계 → ARCore → 씬 좌표계 (왕복 테스트)
    print("=" * 60)
    print("테스트 1: 씬 좌표계 → ARCore → 씬 좌표계 (왕복)")
    print("=" * 60)
    
    # 테스트할 점들 (씬 좌표계)
    test_points_scene = [
        np.array([0.0, 0.0, 0.0]),  # 원점
        np.array([0.1, 0.1, 0.1]),  # 임의 점
        np.array([-0.05, 0.2, -0.1]),  # 음수 포함
    ]
    
    if anchor_pos_scene is not None:
        test_points_scene.append(anchor_pos_scene)
    
    all_passed_1 = True
    for i, point_scene_original in enumerate(test_points_scene):
        # 씬 → ARCore
        point_arcore = transform_point_to_arcore(
            point_scene_original,
            first_frame_params,
            scene_metadata
        )
        
        # ARCore → 씬
        point_scene_recovered = transform_point_from_arcore(
            point_arcore,
            first_frame_params,
            scene_metadata
        )
        
        # 오차 계산
        error = np.linalg.norm(point_scene_original - point_scene_recovered)
        
        print(f"\n테스트 점 {i+1}:")
        print(f"  원본 (씬):      {point_scene_original}")
        print(f"  ARCore:         {point_arcore}")
        print(f"  복원 (씬):       {point_scene_recovered}")
        print(f"  오차:           {error:.8f}")
        
        if error < 1e-5:
            print(f"  ✓ 통과 (오차 < 1e-5)")
        else:
            print(f"  ✗ 실패 (오차 >= 1e-5)")
            all_passed_1 = False
    
    # 테스트 2: ARCore 좌표계 → 씬 → ARCore 좌표계 (역방향 왕복 테스트)
    print("\n" + "=" * 60)
    print("테스트 2: ARCore 좌표계 → 씬 → ARCore 좌표계 (역방향 왕복)")
    print("=" * 60)
    
    # 테스트할 점들 (ARCore 좌표계)
    test_points_arcore = [
        np.array([0.0, 0.0, 0.0]),  # 원점
    ]
    
    if anchor_pos_arcore is not None:
        test_points_arcore.append(anchor_pos_arcore)
    
    test_points_arcore.append(np.array([0.01, -0.5, -0.2]))  # 임의 점
    
    all_passed_2 = True
    for i, point_arcore_original in enumerate(test_points_arcore):
        # ARCore → 씬
        point_scene = transform_point_from_arcore(
            point_arcore_original,
            first_frame_params,
            scene_metadata
        )
        
        # 씬 → ARCore
        point_arcore_recovered = transform_point_to_arcore(
            point_scene,
            first_frame_params,
            scene_metadata
        )
        
        # 오차 계산
        error = np.linalg.norm(point_arcore_original - point_arcore_recovered)
        
        print(f"\n테스트 점 {i+1}:")
        print(f"  원본 (ARCore):  {point_arcore_original}")
        print(f"  씬:             {point_scene}")
        print(f"  복원 (ARCore):   {point_arcore_recovered}")
        print(f"  오차:           {error:.8f}")
        
        if error < 1e-5:
            print(f"  ✓ 통과 (오차 < 1e-5)")
        else:
            print(f"  ✗ 실패 (오차 >= 1e-5)")
            all_passed_2 = False
    
    # 테스트 3: 랜덤 점들로 대량 테스트
    print("\n" + "=" * 60)
    print("테스트 3: 랜덤 점들로 대량 테스트")
    print("=" * 60)
    
    np.random.seed(42)  # 재현성을 위한 시드 설정
    n_tests = 100
    max_error = 0.0
    errors = []
    
    for i in range(n_tests):
        # 랜덤 씬 좌표계 점 생성
        point_scene_original = np.random.uniform(-0.5, 0.5, 3)
        
        # 왕복 변환
        point_arcore = transform_point_to_arcore(
            point_scene_original,
            first_frame_params,
            scene_metadata
        )
        point_scene_recovered = transform_point_from_arcore(
            point_arcore,
            first_frame_params,
            scene_metadata
        )
        
        # 오차 계산
        error = np.linalg.norm(point_scene_original - point_scene_recovered)
        errors.append(error)
        max_error = max(max_error, error)
    
    print(f"테스트한 점 수: {n_tests}")
    print(f"최대 오차: {max_error:.8f}")
    print(f"평균 오차: {np.mean(errors):.8f}")
    print(f"중앙값 오차: {np.median(errors):.8f}")
    
    all_passed_3 = max_error < 1e-5
    if all_passed_3:
        print(f"✓ 모든 테스트 통과! (최대 오차 < 1e-5)")
    else:
        print(f"✗ 일부 테스트 실패 (최대 오차 >= 1e-5)")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
    
    return all_passed_1 and all_passed_2 and all_passed_3

# 노트북에서 사용할 수 있는 간단한 버전
test_code_for_notebook = """
#역변환 테스트
import numpy as np

# 테스트 1: 씬 좌표계 → ARCore → 씬 좌표계 (왕복 테스트)
print("=" * 60)
print("테스트 1: 씬 좌표계 → ARCore → 씬 좌표계 (왕복)")
print("=" * 60)

test_points_scene = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.1, 0.1, 0.1]),
    np.array([-0.05, 0.2, -0.1]),
]

try:
    test_points_scene.append(anchor_pos_scene)
except NameError:
    pass

for i, point_scene_original in enumerate(test_points_scene):
    point_arcore = transform_point_to_arcore(point_scene_original, first_frame_params, scene_metadata)
    point_scene_recovered = transform_point_from_arcore(point_arcore, first_frame_params, scene_metadata)
    error = np.linalg.norm(point_scene_original - point_scene_recovered)
    
    print(f"\\n테스트 점 {i+1}:")
    print(f"  원본: {point_scene_original}")
    print(f"  ARCore: {point_arcore}")
    print(f"  복원: {point_scene_recovered}")
    print(f"  오차: {error:.8f} {'✓' if error < 1e-5 else '✗'}")

# 테스트 2: ARCore → 씬 → ARCore (역방향)
print("\\n" + "=" * 60)
print("테스트 2: ARCore → 씬 → ARCore (역방향)")
print("=" * 60)

test_points_arcore = [np.array([0.0, 0.0, 0.0])]
try:
    test_points_arcore.append(anchor_pos_arcore)
except NameError:
    pass
test_points_arcore.append(np.array([0.01, -0.5, -0.2]))

for i, point_arcore_original in enumerate(test_points_arcore):
    point_scene = transform_point_from_arcore(point_arcore_original, first_frame_params, scene_metadata)
    point_arcore_recovered = transform_point_to_arcore(point_scene, first_frame_params, scene_metadata)
    error = np.linalg.norm(point_arcore_original - point_arcore_recovered)
    
    print(f"\\n테스트 점 {i+1}:")
    print(f"  원본: {point_arcore_original}")
    print(f"  씬: {point_scene}")
    print(f"  복원: {point_arcore_recovered}")
    print(f"  오차: {error:.8f} {'✓' if error < 1e-5 else '✗'}")

# 테스트 3: 랜덤 대량 테스트
print("\\n" + "=" * 60)
print("테스트 3: 랜덤 100개 점 테스트")
print("=" * 60)

np.random.seed(42)
errors = []
for _ in range(100):
    point_scene_original = np.random.uniform(-0.5, 0.5, 3)
    point_arcore = transform_point_to_arcore(point_scene_original, first_frame_params, scene_metadata)
    point_scene_recovered = transform_point_from_arcore(point_arcore, first_frame_params, scene_metadata)
    errors.append(np.linalg.norm(point_scene_original - point_scene_recovered))

print(f"최대 오차: {max(errors):.8f}")
print(f"평균 오차: {np.mean(errors):.8f}")
print(f"{'✓ 통과' if max(errors) < 1e-5 else '✗ 실패'}")
"""

print("테스트 코드가 test_transform_inverse.py에 저장되었습니다.")
print("\n노트북에서 사용할 코드:")
print(test_code_for_notebook)
