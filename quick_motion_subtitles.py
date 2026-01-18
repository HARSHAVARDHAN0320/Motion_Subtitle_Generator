# quick_motion_subtitles.py
import cv2
import math
import numpy as np
from pathlib import Path
from transformers import BlipForConditionalGeneration, BlipProcessor
import mediapipe as mp
from PIL import Image
import os
import hashlib
from difflib import SequenceMatcher

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

CACHE_DIR = Path(".caption_cache")
CACHE_DIR.mkdir(exist_ok=True)

# small verb list to check whether BLIP caption actually contains an action word
COMMON_VERBS = set([
    "walk", "walking", "run", "running", "jump", "jumping", "sit", "sitting", "stand", "standing",
    "raise", "raising", "wave", "waves", "turn", "turning", "bend", "bending", "fall", "falls",
    "pick", "pick up", "throw", "catch", "move", "moves", "drive", "drives", "stop", "stops",
    "open", "opens", "close", "closes", "push", "pull", "ride", "riding", "kick", "kicks"
])

def extract_frames(video_path, fps_out=2):
    vid = cv2.VideoCapture(str(video_path))
    in_fps = vid.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frame_count / in_fps if in_fps>0 else 0
    sample_every = max(1, int(round(in_fps / fps_out)))
    frames = []
    timestamps = []
    i = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if i % sample_every == 0:
            t = i / in_fps if in_fps>0 else 0
            frames.append(frame.copy())
            timestamps.append(t)
        i += 1
    vid.release()
    return frames, timestamps, dur

def pose_movement_tag(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    nose = lm[0]
    left_hip = lm[23]
    right_hip = lm[24]
    # visibility check
    if nose.visibility < 0.4:
        return None
    hip_vs_nose = (left_hip.y + right_hip.y)/2 - nose.y
    if hip_vs_nose > 0.18:
        return 'bending'
    left_shoulder = lm[11]
    left_hip_x_diff = abs(left_shoulder.x - left_hip.x)
    if left_hip_x_diff > 0.08:
        return 'turning'
    # very simple motion detection via approximate vertical velocity of nose over frames is possible,
    # but for now return 'standing' as default (we only use tag when caption is unreliable)
    return 'standing'

def caption_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    small = img.resize((224,224)).convert("RGB")
    h = hashlib.md5(small.tobytes()).hexdigest()
    cache_file = CACHE_DIR / f"{h}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding='utf-8')
    inputs = processor(images=img, return_tensors='pt')
    with np.errstate(all='ignore'):
        out = model.generate(**inputs, max_length=25)
    text = processor.decode(out[0], skip_special_tokens=True)
    # shorten long captions and take the first clause
    if ',' in text:
        text = text.split(',')[0].strip()
    # lowercase for easier verb checking later
    text = text.strip()
    cache_file.write_text(text, encoding='utf-8')
    return text

def contains_common_verb(caption):
    if not caption:
        return False
    low = caption.lower()
    for v in COMMON_VERBS:
        if v in low:
            return True
    return False

def fuse_tags_and_captions(tag, caption, min_verb_hit=True):
    caption = (caption or "").strip()
    # if caption is empty or doesn't contain a verb we trust, fall back to tag
    if not caption:
        return tag_text_from_tag(tag)
    if min_verb_hit and not contains_common_verb(caption):
        # BLIP likely described objects/scenery instead of an action — prefer pose tag
        if tag:
            return tag_text_from_tag(tag)
        return caption
    # otherwise combine: keep caption (short) and append short verb phrase
    verb_map = {
        'bending': 'bends',
        'turning': 'turns',
        'standing': 'stands'
    }
    verb = verb_map.get(tag, '')
    # avoid duplication: if caption already contains the verb, just return caption
    if verb and verb in caption.lower():
        return caption
    if verb:
        # return short sentence: "<caption>. <verb>."
        return f"{caption}. {verb}."
    return caption

def tag_text_from_tag(tag):
    if not tag:
        return None
    if tag == 'bending':
        return 'person bending'
    if tag == 'turning':
        return 'person turning'
    if tag == 'standing':
        return 'person standing'
    return f"person {tag}"

def similar(a, b):
    if not a or not b:
        return False
    r = SequenceMatcher(None, a, b).ratio()
    return r > 0.75

def merge_entries(entries, gap_tolerance=0.6):
    if not entries:
        return []
    merged = []
    cur_start, cur_end, cur_text = entries[0]
    for start, end, text in entries[1:]:
        if text is None:
            continue
        if similar(cur_text, text) and start - cur_end <= gap_tolerance:
            cur_end = end
        else:
            merged.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = start, end, text
    merged.append((cur_start, cur_end, cur_text))
    merged = [(s,e,t) for s,e,t in merged if t and len(t.strip())>2]
    return merged

def make_srt(entries, out_path='out.srt'):
    def fmt_time(t):
        h = int(t//3600)
        m = int((t%3600)//60)
        s = int(t%60)
        ms = int((t - int(t))*1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(out_path,'w',encoding='utf-8') as f:
        for i,(start,end,text) in enumerate(entries, start=1):
            f.write(f"{i}\n")
            f.write(fmt_time(start)+ ' --> ' + fmt_time(end)+ '\n')
            f.write(text + '\n\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--fps',type=float,default=2.0, help='sampling fps for frame captions (try 1-4)')
    parser.add_argument('--out','-o',default='out.srt')
    parser.add_argument('--min-verb-hit', action='store_true',
                        help='require BLIP caption to contain a common verb; otherwise use pose tag')
    args = parser.parse_args()

    frames, times, dur = extract_frames(args.video, fps_out=args.fps)
    entries = []

    for f,t in zip(frames,times):
        tag = pose_movement_tag(f)
        caption = caption_frame(f)
        text = fuse_tags_and_captions(tag, caption, min_verb_hit=args.min_verb_hit)
        if not text:
            continue
        start = t
        end = t + (1.0/args.fps) + 0.5
        entries.append((start, min(end,dur), text))

    merged = merge_entries(entries, gap_tolerance=0.8)
    make_srt(merged, args.out)
    print('Wrote', args.out)
