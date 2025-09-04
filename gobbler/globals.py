import os

core_dir = os.path.dirname(__file__)
instructions_dir = os.path.join(core_dir, "instructions")
models_dir = os.path.join(core_dir, "models/checkpoints/")

clip_prob_thresh = float(os.getenv("CLIP_PROB_THRESHOLD", 0.65))
ssim_threshold = float(os.getenv("SSIM_THRESHOLD", 0.9))
color_hist_threshold = float(os.getenv("COLOR_HIST_THRESHOLD", 0.99))
video_seconds_per_frame = int(os.getenv("VIDEO_SECONDS_PER_FRAME", 4))

yolo_prob_threshold = float(os.getenv("YOLO_PROB_THRESHOLD", 0.3))
