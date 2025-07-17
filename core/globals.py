import os

core_dir = os.path.dirname(__file__)
instructions_dir = os.path.join(core_dir, "instructions")

clip_prob_thresh = os.getenv("CLIP_THRESHOLD", 0.7)
