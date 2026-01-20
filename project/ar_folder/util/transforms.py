"""
ARCore 좌표 변환 함수 모듈
"""
import numpy as np
from scipy.spatial.transform import Rotation


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
