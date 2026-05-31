"""Full clip generation script"""

import subprocess
import os

def run_pipeline(video_path, model_path, loss_type, window_len=8, overlap_len=4, threshold=0.5,
                 temp_clip_dir=None, sam3_script=None, sam3_output_dir=None):
    """
    Splits a video into overlapping clips, then runs the SAM3 prediction script
    on each clip to generate .pkl output files.

    Args:
        video_path (str):       Absolute path to the input video.
        model_path (str):       Absolute path to the model being run.
        loss_type (str):        Loss type the model uses (BCEL or LSCE).
        window_len (int):       Clip length in seconds.
        overlap_len (int):      Overlap between consecutive clips in seconds (< window_len).
        threshold (float):      Prediction threshold (0 <= x <= 1).
        temp_clip_dir (str):    Directory to write temporary video clips.
        sam3_script (str):      Absolute path to run_sam3.py.
        sam3_output_dir (str):  Directory where SAM3 will write .pkl files.
    """

    # --- Step 1: Get video duration ---
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    video_duration = int(float(probe.stdout.strip()))

    # --- Step 2: Split video into overlapping clips ---
    # NOTE: Do NOT use -c copy. Re-encoding is slower but avoids corrupted
    #       frames at the start of each clip.
    os.makedirs(temp_clip_dir, exist_ok=True)
    clip_paths = []

    for idx, start in enumerate(range(0, video_duration - window_len, window_len - overlap_len)):
        clip_path = os.path.join(temp_clip_dir, f"output_{idx:05d}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-ss", str(start),
                "-t", str(window_len),
                "-reset_timestamps", "1",
                clip_path,
            ]
        )
        clip_paths.append(clip_path)

    # --- Step 3: Run SAM3 on each clip to generate .pkl files ---
    os.makedirs(sam3_output_dir, exist_ok=True)

    for clip_path in clip_paths:
        print(f"Running SAM3 on: {clip_path}")
        subprocess.run(
            ["python", sam3_script, "--output_dir", sam3_output_dir, clip_path]
        )
        print(f"Finished: {clip_path}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

video_path     = "/fs/vulcan-projects/fsh_track/charles/aa_test_video_main_short/2_min.mp4"
model_path     = ""
loss_type      = ""

window_len     = 14
overlap_len    = 4
threshold      = 0

temp_clip_dir  = "./clips/"
sam3_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../trokens++/run_sam3.py")
sam3_output_dir = "./sam3_pkl/"

run_pipeline(
    video_path=video_path,
    model_path=model_path,
    loss_type=loss_type,
    window_len=window_len,
    overlap_len=overlap_len,
    threshold=threshold,
    temp_clip_dir=temp_clip_dir,
    sam3_script=sam3_script,
    sam3_output_dir=sam3_output_dir,
)