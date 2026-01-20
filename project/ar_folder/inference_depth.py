import numpy as np
import json
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
import glob
import os
import re

# 상수 정의
SAMPLING_INTERVAL_NS = 200_000_000

# JSONL 파일 읽기
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

def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)\.jpg', filename)
    return int(match.group(1)) if match else None

def get_frame_number_from_timestamp(t_ns):
    return int(t_ns // SAMPLING_INTERVAL_NS)

if __name__ == "__main__":
    # 폴더 경로 설정
    base_folder = "/data/ephemeral/home/measure_volume_by_multiview/project/ar_folder"
    
    # 가장 최근 세션 폴더 찾기
    session_folders = [
        f for f in sorted(glob.glob(os.path.join(base_folder, "session_*")))
        if os.path.isdir(f)
    ]
    if not session_folders:
        raise FileNotFoundError(f"세션 폴더를 찾을 수 없습니다: {os.path.join(base_folder, 'session_*')}")
    
    session_folder = session_folders[-1]
    print(f"사용할 세션 폴더: {session_folder}")
    
    jsonl_files = glob.glob(os.path.join(session_folder, "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {os.path.join(session_folder, '*.jsonl')}")
    
    jsonl_file = jsonl_files[0]
    image_files = sorted(glob.glob(os.path.join(session_folder, "frame_*.jpg")))
    print(f"발견된 이미지 파일 수: {len(image_files)}")
    
    # JSONL 데이터 로드 및 매칭
    jsonl_data = load_jsonl(jsonl_file)
    jsonl_data.sort(key=lambda x: x.get('t_ns', 0))
    print(f"JSONL 데이터 수: {len(jsonl_data)}")
    
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
    print(f"매칭된 데이터 수: {len(matched_data)}")
    
    if len(matched_data) == 0:
        raise ValueError("매칭된 데이터가 없습니다.")
    
    # 첫 프레임 앵커 정보 추출 (A_1)
    first_frame_params = matched_data[0]['params']
    first_anchor_pos = np.array(first_frame_params.get('anchor_pos', first_frame_params.get('pos', [0.0, 0.0, 0.0])), dtype=np.float32)
    first_anchor_quat = np.array(first_frame_params.get('anchor_quat', first_frame_params.get('quat', [0.0, 0.0, 0.0, 1.0])), dtype=np.float32)
    first_anchor_rotation = Rotation.from_quat(first_anchor_quat).as_matrix()
    print(f"첫 프레임 앵커: pos=({first_anchor_pos[0]:.3f}, {first_anchor_pos[1]:.3f}, {first_anchor_pos[2]:.3f})")
    
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
    
    # 데이터 변환 (앵커 드리프트 보정 적용)
    print("데이터 변환 중 (앵커 드리프트 보정 적용)...")
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
            print(f"Frame {i}: original pos=({pos_cv[0]:.3f}, {pos_cv[1]:.3f}, {pos_cv[2]:.3f}), "
                  f"corrected pos=({Pi_corrected[0,3]:.3f}, {Pi_corrected[1,3]:.3f}, {Pi_corrected[2,3]:.3f})")
        
        # Intrinsics 구성 (COLMAP convention)
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
    
    # DA3 추론 (depth map만 추출)
    print("Depth-Anything-3 모델 로딩 중...")
    model = DepthAnything3.from_pretrained("depth-anything/da3-base").to("cuda")
    
    project_root = os.path.dirname(base_folder)
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"DA3 추론 시작 (이미지 {len(matched_image_files)}개)...")
    prediction = model.inference(
        image=matched_image_files,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        use_ray_pose=True,
        process_res=1008,
        export_dir=output_dir,
        export_format="mini_npz"  # depth map만 저장 (GLB 제외)
    )
    
    print(f"Depth map 추출 완료!")
    print(f"출력 경로: {os.path.join(output_dir, 'exports', 'mini_npz', 'results.npz')}")
    print(f"Depth shape: {prediction.depth.shape}")
    print(f"Depth range: {prediction.depth.min():.3f} ~ {prediction.depth.max():.3f} meters")
