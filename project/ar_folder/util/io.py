"""
파일 입출력 및 데이터 로딩 모듈
"""
import numpy as np
import json
import trimesh
from scipy.spatial.transform import Rotation


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


def load_jsonl(jsonl_path):
    """JSONL 파일 읽기 함수"""
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
