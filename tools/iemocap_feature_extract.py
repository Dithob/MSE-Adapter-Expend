import os
import re
import cv2
import glob
import torch
import librosa
import pickle
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm
from itertools import groupby

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# =========================================================
# 配置区 (已更新为 AutoDL 绝对路径)
# =========================================================
# 直接指向你挂载的 IEMOCAP 数据集绝对路径
IEMOCAP_DIR = '/root/autodl-tmp/datasets/iemocap/IEMOCAP_full_release' 
# 将提取出的特征文件保存在同级目录下，避免占用系统盘
OUTPUT_DIR = '/root/autodl-tmp/datasets/IEMOCAP_Processed'

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# IEMOCAP_DIR = os.path.join(PROJECT_ROOT, "datasets", "IEMOCAP", "IEMOCAP_full_release") 
# OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "IEMOCAP_Processed")

TARGET_AUDIO_DIM = 64   # Mel-Spectrogram 通道数
TARGET_VIDEO_DIM = 768  # ViT-Base 输出维度

EMOTION_MAP_4 = {'ang': 'angry', 'hap': 'happy', 'exc': 'happy', 'sad': 'sad', 'neu': 'neutral'}
EMOTION_MAP_6 = {'ang': 'angry', 'hap': 'happy', 'exc': 'excited', 'sad': 'sad', 'neu': 'neutral', 'fru': 'frustrated'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# 模型加载
# =========================================================
print(f"🔥 检测到计算设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("1/2 初始化 MTCNN 人脸检测器...")
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=True, device=device)

print("2/2 初始化 Vision Transformer (ViT-Base)...")
video_extractor = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
video_extractor = video_extractor.to(device)
video_extractor.eval()

fallback_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# =========================================================
# 提取函数
# =========================================================
def extract_audio_features(full_y, sr, start_time, end_time):
    try:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = full_y[start_sample:end_sample]
        
        if len(y) < 400: return np.zeros((1, TARGET_AUDIO_DIM)), 0
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=TARGET_AUDIO_DIM, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        features = log_mel_spec.T 
        return features, features.shape[0]
    except Exception:
        return np.zeros((1, TARGET_AUDIO_DIM)), 0

def extract_video_features(cap, fps, start_time, end_time, max_seq_len=32):
    try:
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = max(1, end_frame - start_frame + 1)
        
        # 计算需要提取的帧索引
        if total_frames > max_seq_len:
            indices = sorted(list(set(np.linspace(start_frame, end_frame, max_seq_len, dtype=int))))
        else:
            indices = list(range(start_frame, end_frame + 1))
            
        target_indices = set(indices)  # 转为集合，O(1) 极速查找
        
        # 🚀 优化 1：只进行 1 次寻址，随后连续读取！
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        pil_images = []
        
        # 连续读取直到当前切片结束
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: 
                break
                
            # 只有当帧号在我们的目标列表中时，才保存
            if current_frame in target_indices:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_images.append(img_pil)
                
            current_frame += 1

        if not pil_images: 
            return np.zeros((1, TARGET_VIDEO_DIM)), 0, 0, 0

        # 🚀 优化 2：MTCNN 批处理火力全开！
        # MTCNN 原生支持传入 List[Image]，一次性在 GPU 上检测所有帧
        faces_batch = mtcnn(pil_images)
        
        frames_tensor_list = []
        faces_found = 0
        
        for i, face in enumerate(faces_batch):
            if face is not None:
                frames_tensor_list.append(face)
                faces_found += 1
            else:
                frames_tensor_list.append(fallback_transform(pil_images[i]))

        # 把这 32 张图推入 4090D 显存
        frames_tensor = torch.stack(frames_tensor_list).to(device)
        total_sampled = len(pil_images)
        
        features_list = []
        batch_size = 64 # ViT 也是满载运行
        
        # 🚀 优化 3：ViT 提取
        with torch.no_grad():
            for i in range(0, frames_tensor.size(0), batch_size):
                out = video_extractor(frames_tensor[i:i+batch_size])
                features_list.append(out.cpu().numpy())
                
        return np.concatenate(features_list, axis=0), total_sampled, faces_found, total_sampled
        
    except Exception as e:
        # print(f"提取报错: {e}") # 调试时可以解开这句
        return np.zeros((1, TARGET_VIDEO_DIM)), 0, 0, 0

# =========================================================
# 解析工具
# =========================================================
def parse_labels():
    info = {}
    for s in range(1, 6):
        d = os.path.join(IEMOCAP_DIR, f'Session{s}', 'dialog', 'EmoEvaluation')
        if not os.path.exists(d): continue
        for txt in glob.glob(os.path.join(d, '*.txt')):
            with open(txt, 'r') as f:
                for line in f:
                    if line.startswith('['):
                        t_match = re.match(r'\[(.*) - (.*)\]', line.split('\t')[0])
                        parts = line.strip().split('\t')
                        if t_match and len(parts) >= 3:
                            info[parts[1]] = {'emo': parts[2], 'start': float(t_match.group(1)), 'end': float(t_match.group(2))}
    return info

def parse_trans():
    text = {}
    for s in range(1, 6):
        d = os.path.join(IEMOCAP_DIR, f'Session{s}', 'dialog', 'transcriptions')
        if not os.path.exists(d): continue
        for txt in glob.glob(os.path.join(d, '*.txt')):
            with open(txt, 'r') as f:
                for line in f:
                    match = re.match(r'(Ses\w+) \[.*\]:\s*(.*)', line)
                    if match: text[match.group(1)] = match.group(2).strip()
    return text

# =========================================================
# 主流程
# =========================================================
def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c_map = EMOTION_MAP_4 if args.classes == 4 else EMOTION_MAP_6
        
    print(f"\n📁 检查数据集路径: {IEMOCAP_DIR}")
    info_dict = parse_labels()
    text_dict = parse_trans()

    avi_files = glob.glob(os.path.join(IEMOCAP_DIR, 'Session*', 'dialog', 'avi', 'DivX', '*.avi'))
    if not avi_files: avi_files = glob.glob(os.path.join(IEMOCAP_DIR, 'Session*', 'dialog', 'avi', '*.avi'))

    valid_tasks = []
    for avi in avi_files:
        s_id = os.path.basename(avi).replace('.avi', '')
        for u_id, u_info in {k: v for k, v in info_dict.items() if k.startswith(s_id)}.items():
            if u_info['emo'] in c_map:
                valid_tasks.append({'avi': avi, 's_id': s_id, 'u_id': u_id, 'info': u_info})

    print(f"📊 需要处理的有效对话总数: {len(valid_tasks)}")
    if len(valid_tasks) == 0:
        print("❌ 未找到匹配的标签，请检查数据集路径！")
        return

    all_data = []
    global_faces, global_sampled = 0, 0
    pbar = tqdm(total=len(valid_tasks), desc="🚀 特征提取中", unit="句", dynamic_ncols=True)

    valid_tasks.sort(key=lambda x: x['avi'])
    for avi, group in groupby(valid_tasks, key=lambda x: x['avi']):
        tasks = list(group)
        s_id = tasks[0]['s_id']
        
        avi_dir = os.path.dirname(avi)
        wav_dir = os.path.join(os.path.dirname(os.path.dirname(avi_dir)), 'wav') if os.path.basename(avi_dir) == 'DivX' else os.path.join(os.path.dirname(avi_dir), 'wav')
        wav_path = os.path.join(wav_dir, s_id + '.wav')
        
        try:
            full_y, sr = librosa.load(wav_path, sr=16000)
        except Exception:
            pbar.update(len(tasks)); continue

        cap = cv2.VideoCapture(avi)
        if not cap.isOpened(): pbar.update(len(tasks)); continue
        fps = cap.get(cv2.CAP_PROP_FPS)

        for t in tasks:
            u_info = t['info']
            v_feat, v_len, faces, sampled = extract_video_features(cap, fps, u_info['start'], u_info['end'])
            a_feat, a_len = extract_audio_features(full_y, sr, u_info['start'], u_info['end'])

            global_faces += faces; global_sampled += sampled
            all_data.append((t['u_id'], {
                'label': c_map[u_info['emo']],
                'features': {
                    'text': text_dict.get(t['u_id'], ""),
                    'audio': a_feat.astype(np.float32), 'video': v_feat.astype(np.float32),
                    'audio_len': a_len, 'video_len': v_len
                }
            }))
            pbar.set_postfix({'Session': s_id[-5:]}); pbar.update(1)
            
        cap.release()
    pbar.close()
    
    if global_sampled > 0:
        print(f"\n📈 统计面板: 共采样 {global_sampled} 帧, 成功抓取人脸 {global_faces} 帧 (成功率 {(global_faces/global_sampled)*100:.2f}%)")
    
    train_data = [samp for id, samp in all_data if 'Ses05' not in id]
    test_data  = [samp for id, samp in all_data if 'Ses05' in id]
    
    train_pkl, test_pkl = os.path.join(OUTPUT_DIR, f'iemocap_{args.classes}class_train.pkl'), os.path.join(OUTPUT_DIR, f'iemocap_{args.classes}class_test.pkl')
    with open(train_pkl, 'wb') as f: pickle.dump(train_data, f)
    with open(test_pkl, 'wb') as f: pickle.dump(test_data, f)
    print(f"🎉 提取完成！Train: {len(train_data)} 条, Test: {len(test_data)} 条. 数据已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, choices=[4, 6], default=4)
    main(parser.parse_args())