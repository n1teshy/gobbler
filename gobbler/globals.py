import os

core_dir = os.path.dirname(__file__)
instructions_dir = os.path.join(core_dir, "instructions")

clip_prob_thresh = float(os.getenv("CLIP_PROB_THRESHOLD", 0.65))
ssim_threshold = float(os.getenv("SSIM_THRESHOLD", 0.9))
color_hist_threshold = float(os.getenv("COLOR_HIST_THRESHOLD", 0.99))
video_seconds_per_frame = int(os.getenv("VIDEO_SECONDS_PER_FRAME", 4))

yolo_prob_threshold = float(os.getenv("YOLO_PROB_THRESHOLD", 0.5))
yolo_fallback_clip_threshold = float(
    os.getenv("YOLO_FALLBACK_CLIP_THRESHOLD", 0.75)
)
filled_pixel_region_stddev = int(os.getenv("FILLED_PIXEL_REGION_STDDEV", 22))
