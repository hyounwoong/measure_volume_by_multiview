import numpy as np
import json
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3
from PIL import Image
import glob
import os
import re

# ============================================
# 폴더 경로 설정
# ============================================
base_folder = "/data/ephemeral/home/ar_folder"  # 기본 폴더

# 방법 1: 특정 세션 폴더 지정
# session_folder = os.path.join(base_folder, "session_1766721309500")

# 방법 2: 가장 최근 세션 폴더 자동 찾기
session_folders = sorted(glob.glob(os.path.join(base_folder, "session_*")))
if not session_folders:
    raise FileNotFoundError(f"세션 폴더를 찾을 수 없습니다: {os.path.join(base_folder, 'session_*')}")

session_folder = session_folders[-1]  # 가장 최근 세션 사용
print(f"사용할 세션 폴더: {session_folder}")

# JSONL 파일 찾기 (세션 폴더 안에서)
jsonl_files = glob.glob(os.path.join(session_folder, "*.jsonl"))
if not jsonl_files:
    raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {os.path.join(session_folder, '*.jsonl')}")
if len(jsonl_files) > 1:
    print(f"경고: 여러 개의 JSONL 파일이 발견되었습니다. 첫 번째 파일을 사용합니다: {jsonl_files[0]}")

jsonl_file = jsonl_files[0]
image_files = sorted(glob.glob(os.path.join(session_folder, "frame_*.jpg")))

# ============================================
# JSONL 파일 읽기 (여러 줄에 걸친 JSON 객체 처리)
# ============================================
def load_jsonl(jsonl_path):
    """JSONL 파일 읽기 - 여러 줄에 걸친 JSON 객체 처리"""
    data = []
    with open(jsonl_path, 'r') as f:
        content = f.read()
    
    # 중괄호 카운팅으로 각 JSON 객체 분리
    current_obj = ""
    brace_count = 0
    for char in content:
        current_obj += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:  # JSON 객체 완성
                try:
                    data.append(json.loads(current_obj.strip()))
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                current_obj = ""
    return data

jsonl_data = load_jsonl(jsonl_file)

# ⚠️ 중요: t_ns로 정렬 (백그라운드 스레드로 인한 순서 섞임 방지)
jsonl_data.sort(key=lambda x: x.get('t_ns', 0))

print(f"JSONL 파일: {jsonl_file}")
print(f"JSONL에서 {len(jsonl_data)}개 프레임 로드됨 (t_ns로 정렬됨)")

# 정렬 확인
if len(jsonl_data) > 1:
    t_ns_list = [item.get('t_ns', 0) for item in jsonl_data]
    is_sorted = all(t_ns_list[i] <= t_ns_list[i+1] for i in range(len(t_ns_list)-1))
    if not is_sorted:
        print("⚠️  경고: JSONL 데이터가 시간 순서대로 정렬되지 않았습니다!")
    else:
        print(f"✅ JSONL 데이터가 시간 순서대로 정렬되었습니다.")
        print(f"   첫 번째 t_ns: {t_ns_list[0]}")
        print(f"   마지막 t_ns: {t_ns_list[-1]}")
        print(f"   시간 범위: {(t_ns_list[-1] - t_ns_list[0]) / 1e9:.2f}초")

image_files = sorted(glob.glob(os.path.join(session_folder, "frame_*.jpg")))
print(f"이미지 파일 {len(image_files)}개 찾음")

# ============================================
# 프레임 번호로 이미지와 JSONL 데이터 매칭 (수정)
# ============================================
SAMPLING_INTERVAL_NS = 200_000_000  # 0.2초 (5fps)

def extract_frame_number(filename):
    """파일명에서 프레임 번호 추출 (예: frame_1501460.jpg -> 1501460)"""
    match = re.search(r'frame_(\d+)\.jpg', filename)
    return int(match.group(1)) if match else None

def get_frame_number_from_timestamp(t_ns):
    """JSONL의 t_ns로 프레임 번호 계산 (ARCore와 동일한 로직)"""
    return int(t_ns // SAMPLING_INTERVAL_NS)

# 이미지 파일을 프레임 번호로 매핑
image_dict = {}
for img_path in image_files:
    frame_num = extract_frame_number(os.path.basename(img_path))
    if frame_num is not None:
        image_dict[frame_num] = img_path

print(f"이미지 파일 {len(image_dict)}개 매핑됨")

# 실제 SAMPLING_INTERVAL_NS 역산 (이미지 파일명과 JSONL의 t_ns 비교)
# 여러 프레임을 비교해서 더 정확하게 역산
if len(jsonl_data) > 0 and len(image_dict) > 0:
    intervals = []
    for jsonl_item in jsonl_data[:min(5, len(jsonl_data))]:  # 최대 5개 프레임 비교
        t_ns = jsonl_item.get('t_ns')
        if t_ns is None:
            continue
        
        # 이 t_ns로 계산된 프레임 번호
        calculated_frame = int(t_ns // SAMPLING_INTERVAL_NS)
        
        # 실제 이미지 파일명의 프레임 번호와 매칭되는지 확인
        # 가장 가까운 프레임 번호 찾기
        closest_frame = None
        min_diff = float('inf')
        for img_frame in image_dict.keys():
            diff = abs(img_frame - calculated_frame)
            if diff < min_diff:
                min_diff = diff
                closest_frame = img_frame
        
        # 가장 가까운 프레임이 있으면 역산
        if closest_frame is not None and min_diff < 10:  # 차이가 10 이하일 때만
            estimated_interval = t_ns // closest_frame
            intervals.append(estimated_interval)
    
    if intervals:
        # 중앙값 사용 (이상치에 덜 민감)
        intervals.sort()
        estimated_interval = intervals[len(intervals) // 2]
        estimated_fps = 1e9 / estimated_interval
        print(f"역산된 SAMPLING_INTERVAL_NS: {estimated_interval} (예상 FPS: {estimated_fps:.1f})")
        
        # 역산된 값이 현재 설정과 다르면 조정
        if abs(estimated_interval - SAMPLING_INTERVAL_NS) > 50_000_000:  # 0.05초 이상 차이
            print(f"⚠️  설정된 SAMPLING_INTERVAL_NS ({SAMPLING_INTERVAL_NS}, {1e9/SAMPLING_INTERVAL_NS:.1f}fps)와 실제 값 ({estimated_interval}, {estimated_fps:.1f}fps)이 다릅니다.")
            print(f"   실제 데이터에 맞게 {estimated_interval}로 조정합니다.")
            SAMPLING_INTERVAL_NS = estimated_interval

# JSONL 데이터와 이미지를 t_ns 기반으로 정확히 매칭
matched_data = []
missing_frames = []
for jsonl_item in jsonl_data:
    t_ns = jsonl_item.get('t_ns')
    if t_ns is None:
        print(f"경고: t_ns가 없는 JSONL 항목을 건너뜁니다.")
        continue
    
    frame_num = get_frame_number_from_timestamp(t_ns)
    
    if frame_num in image_dict:
        matched_data.append({
            'image': image_dict[frame_num],
            'params': jsonl_item,
            'frame_num': frame_num,
            't_ns': t_ns
        })
    else:
        missing_frames.append((frame_num, t_ns))

if missing_frames:
    print(f"\n⚠️  경고: {len(missing_frames)}개 프레임에 해당하는 이미지를 찾을 수 없습니다:")
    for frame_num, t_ns in missing_frames[:5]:
        print(f"  - 프레임 {frame_num} (t_ns={t_ns})")
    if len(missing_frames) > 5:
        print(f"  ... 외 {len(missing_frames)-5}개")

# 시간 순서대로 정렬 (안전장치)
matched_data.sort(key=lambda x: x['t_ns'])

print(f"\n매칭된 프레임 수: {len(matched_data)}")
if len(matched_data) == 0:
    raise ValueError("매칭된 데이터가 없습니다. 이미지와 JSONL 파일을 확인하세요.")

# 이미지와 JSONL 데이터 개수 확인
if len(image_files) != len(jsonl_data):
    print(f"⚠️  경고: 이미지 파일({len(image_files)}개)과 JSONL 데이터({len(jsonl_data)}개)의 개수가 다릅니다!")

# ============================================
# 데이터 변환 (검증 및 에러 처리 추가)
# ============================================
N = len(matched_data)
print(f"처리할 프레임 수: {N}")

extrinsics = np.zeros((N, 4, 4), dtype=np.float32)
intrinsics = np.zeros((N, 3, 3), dtype=np.float32)
matched_image_files = []

validation_errors = []
validation_warnings = []

for i, item in enumerate(matched_data):
    params = item['params']
    img_path = item['image']
    
    # 이미지 크기 확인 (에러 처리 추가)
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        validation_errors.append(f"프레임 {item['frame_num']}: 이미지 로드 실패 - {e}")
        continue
    
    jsonl_w = params.get('w')
    jsonl_h = params.get('h')
    
    if jsonl_w is None or jsonl_h is None:
        validation_errors.append(f"프레임 {item['frame_num']}: JSONL에 w, h 정보 없음")
        continue
    
    if img_w != jsonl_w or img_h != jsonl_h:
        validation_warnings.append(
            f"프레임 {item['frame_num']}: 이미지 크기 불일치 - "
            f"이미지: {img_w}x{img_h}, JSONL: {jsonl_w}x{jsonl_h}"
        )
        # JSONL의 크기를 사용 (이미지가 리사이즈되었을 수 있음)
        actual_w, actual_h = jsonl_w, jsonl_h
    else:
        actual_w, actual_h = img_w, img_h
    
    # ARCore pose (c2w) → Depth-Anything-3 extrinsics (w2c) 변환
    quat = params.get('quat')
    pos = params.get('pos')
    
    if quat is None or pos is None:
        validation_errors.append(f"프레임 {item['frame_num']}: pose 정보 없음")
        continue
    
    quat = np.array(quat, dtype=np.float32)  # [x, y, z, w]
    pos = np.array(pos, dtype=np.float32)  # [x, y, z]
    
    # 쿼터니언 → 회전 행렬 (OpenGL 좌표계)
    try:
        R_c2w_gl = Rotation.from_quat(quat).as_matrix()
    except Exception as e:
        validation_errors.append(f"프레임 {item['frame_num']}: 쿼터니언 변환 실패 - {e}")
        continue
    
    # OpenGL -> OpenCV 좌표계 변환
    # ARCore는 OpenGL 좌표계 (Y-up, Z-backward)
    # Depth-Anything-3는 OpenCV 좌표계 (Y-down, Z-forward) 기대
    gl_to_cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)
    
    # 회전 행렬 변환: R_cv = gl_to_cv @ R_gl @ gl_to_cv.T
    R_c2w_cv = gl_to_cv @ R_c2w_gl @ gl_to_cv.T
    pos_cv = gl_to_cv @ pos
    
    # Camera-to-World (OpenCV 좌표계)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w_cv
    c2w[:3, 3] = pos_cv
    
    # World-to-Camera 변환 (역행렬)
    try:
        w2c = np.linalg.inv(c2w)
        # 회전 행렬의 determinant 확인 (약 1.0이어야 함)
        det = np.linalg.det(w2c[:3, :3])
        if abs(det - 1.0) > 0.01:
            validation_warnings.append(
                f"프레임 {item['frame_num']}: 회전 행렬 determinant 이상 - {det:.6f}"
            )
    except np.linalg.LinAlgError as e:
        validation_errors.append(f"프레임 {item['frame_num']}: 역행렬 계산 실패 - {e}")
        continue
    
    extrinsics[i] = w2c
    matched_image_files.append(img_path)
    
    # Intrinsics 구성 및 COLMAP convention 적용
    fx = params.get('fx')
    fy = params.get('fy')
    cx = params.get('cx')
    cy = params.get('cy')
    
    if fx is None or fy is None or cx is None or cy is None:
        validation_errors.append(f"프레임 {item['frame_num']}: Intrinsics 정보 불완전")
        continue
    
    # COLMAP convention: principal point에서 0.5를 빼기
    # ARCore는 픽셀 중심 기준, COLMAP은 픽셀 모서리 기준
    cx_colmap = cx - 0.5
    cy_colmap = cy - 0.5
    
    # Intrinsics가 이미지 크기 범위 내에 있는지 확인 (COLMAP 기준으로)
    if cx_colmap < -0.5 or cx_colmap >= actual_w - 0.5 or cy_colmap < -0.5 or cy_colmap >= actual_h - 0.5:
        validation_warnings.append(
            f"프레임 {item['frame_num']}: Intrinsics 범위 초과 - "
            f"cx={cx_colmap:.1f}, cy={cy_colmap:.1f}, 크기={actual_w}x{actual_h}"
        )
    
    intrinsics[i] = np.array([
        [fx, 0.0, cx_colmap],
        [0.0, fy, cy_colmap],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # 첫 번째 프레임 디버깅 정보
    if i == 0:
        print(f"\n첫 번째 프레임 검증:")
        print(f"  프레임 번호: {item['frame_num']}")
        print(f"  t_ns: {item['t_ns']}")
        print(f"  이미지 크기: {img_w}x{img_h}")
        print(f"  JSONL 크기: {jsonl_w}x{jsonl_h}")
        print(f"  원본 Intrinsics (ARCore): fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  COLMAP Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx_colmap:.1f}, cy={cy_colmap:.1f}")
        print(f"  원본 Pose pos (OpenGL): {pos}")
        print(f"  변환된 Pose pos (OpenCV): {pos_cv}")
        print(f"  Pose quat: {quat}")
        print(f"  w2c rotation det: {np.linalg.det(w2c[:3, :3]):.6f}")

# 검증 결과 출력
if validation_errors:
    print(f"\n❌ 심각한 오류 ({len(validation_errors)}개):")
    for error in validation_errors:
        print(f"  - {error}")
    raise ValueError(f"{len(validation_errors)}개 프레임에서 오류가 발생했습니다!")

if validation_warnings:
    print(f"\n⚠️  경고 ({len(validation_warnings)}개):")
    for warning in validation_warnings[:10]:
        print(f"  - {warning}")
    if len(validation_warnings) > 10:
        print(f"  ... 외 {len(validation_warnings)-10}개")
else:
    print("\n✅ 모든 데이터 검증 통과!")

print(f"\n최종 데이터:")
print(f"  Extrinsics shape: {extrinsics.shape}")
print(f"  Intrinsics shape: {intrinsics.shape}")
print(f"  이미지 파일 수: {len(matched_image_files)}")

# ============================================
# DA3 추론
# ============================================
model = DepthAnything3.from_pretrained("depth-anything/da3-giant").to("cuda")





prediction = model.inference(
    image=matched_image_files,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    align_to_input_ext_scale=True,
    use_ray_pose=True,  # 입력 스케일에 맞춤
    export_dir="./output",
    export_format="glb"
)

print("\n추론 완료!")
print(f"Depth shape: {prediction.depth.shape}")
print(f"Extrinsics shape: {prediction.extrinsics.shape}")
print(f"Intrinsics shape: {prediction.intrinsics.shape}")