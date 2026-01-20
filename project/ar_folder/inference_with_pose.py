import numpy as np
import json
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
from PIL import Image
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

# 스크립트로 직접 실행할 때만 실행되는 코드
if __name__ == "__main__":
    # 폴더 경로 설정 (직접 실행 시에만 사용됨, process_and_save_result.py에서는 인자로 전달됨)
    base_folder = "/data/ephemeral/home/measure_volume_by_multiview/project/ar_folder"
    # 디렉토리만 필터링
    session_folders = [
        f for f in sorted(glob.glob(os.path.join(base_folder, "session_*")))
        if os.path.isdir(f)
    ]
    if not session_folders:
        raise FileNotFoundError(f"세션 폴더를 찾을 수 없습니다: {os.path.join(base_folder, 'session_*')}")

    session_folder = session_folders[-1]
    jsonl_files = glob.glob(os.path.join(session_folder, "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {os.path.join(session_folder, '*.jsonl')}")

    jsonl_file = jsonl_files[0]
    image_files = sorted(glob.glob(os.path.join(session_folder, "frame_*.jpg")))

    jsonl_data = load_jsonl(jsonl_file)
    jsonl_data.sort(key=lambda x: x.get('t_ns', 0))

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
        raise ValueError("매칭된 데이터가 없습니다.")

    # 데이터 변환
    N = len(matched_data)
    extrinsics = np.zeros((N, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((N, 3, 3), dtype=np.float32)
    matched_image_files = []

    gl_to_cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)

    for i, item in enumerate(matched_data):
        params = item['params']
        img_path = item['image']
        
        jsonl_w = params.get('w')
        jsonl_h = params.get('h')
        if jsonl_w is None or jsonl_h is None:
            continue
        
        # ARCore pose (c2w) → Depth-Anything-3 extrinsics (w2c) 변환
        quat = params.get('quat')
        pos = params.get('pos')
        if quat is None or pos is None:
            continue
        
        quat = np.array(quat, dtype=np.float32)
        pos = np.array(pos, dtype=np.float32)
        
        R_c2w_gl = Rotation.from_quat(quat).as_matrix()
        R_c2w_cv = gl_to_cv @ R_c2w_gl @ gl_to_cv.T
        pos_cv = gl_to_cv @ pos
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w_cv
        c2w[:3, 3] = pos_cv
        w2c = np.linalg.inv(c2w)
        
        extrinsics[i] = w2c
        matched_image_files.append(img_path)
        
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

    # DA3 추론
    model = DepthAnything3.from_pretrained("depth-anything/da3-small").to("cuda")

    prediction = model.inference(
        image=matched_image_files,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        use_ray_pose=True,
        export_dir="./output",
        export_format="glb"
    )
