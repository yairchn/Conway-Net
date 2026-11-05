"""The paper's activation video (doi.org/10.6084/m9.figshare.34750569): the activation cube of paper_figures/
activation_cube.py for the first test image of each digit, 0 to 9, joined into one video (ffmpeg concat).
Writes paper_figures/out/fcn10_mnist_cube_digits_0to9.mp4 (the per-digit clips go to paper_figures/out/video_clips/).
Run: PYTHONPATH=. python paper_figures/activation_video.py
"""
from conwaynet import REPO_ROOT
import os
import subprocess
import sys

ROOT = REPO_ROOT; OUT = f"{ROOT}/paper_figures/out"; CLIPS = f"{OUT}/video_clips"
os.makedirs(CLIPS, exist_ok=True)
for d in range(10):
    subprocess.run([sys.executable, f"{ROOT}/paper_figures/activation_cube.py", "--digit", str(d), "--rectify", "signed",
                    "--fps", "2", "--out", f"{CLIPS}/digit{d}.mp4"], check=True)
with open(f"{CLIPS}/list.txt", "w") as f:
    f.writelines(f"file 'digit{d}.mp4'\n" for d in range(10))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{CLIPS}/list.txt", "-c", "copy",
                f"{OUT}/fcn10_mnist_cube_digits_0to9.mp4"], check=True)
print("WROTE", f"{OUT}/fcn10_mnist_cube_digits_0to9.mp4")
