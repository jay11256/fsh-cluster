"""Full video prediction script"""

import subprocess

def predict_video(video_path, model_path, loss_type, window_len=8, overlap_len=4, threshold=0.5):
    """
    Generates the prediction matrix for a given video
    Saves the tensor into a .pt file

    Args:
        video_path (str): absolute path to the video
        model_path (str): absolute path to the model being run
        loss_type (str): type of loss the model uses (BCEL or LSCE)
        window_len (num): length of sliding window in seconds
        overlap_len (num): length of overlap on clips in seconds (< window_len)
        threshold (num): threshold for a true prediction (0 <= x <= 1)

    Returns:
        prediction_matrix (torch.Tensor): 6xN matrix of prediction values
    """
    # BEFORE RUNNING
        # Navigate to fsh-cluster/pipeline
        # Run "module load ffmpeg" before calling!
        # Maybe specify temp video path?    
            # ^Empty contents of this directory
        # Maybe specify output video naming format?
 
    ''' TESTING VIDEO SPLITTING ON 
        first two minutes of
        /fs/vulcan-projects/fsh_track/raw_data/box/AR_natural_spawns_JB/080225_spawn_B1-5_ARdoublehet/080225_spawn_B1-5.mp4

        stored at /fs/vulcan-projects/fsh_track/charles/aa_test_video_main_short/2_min.mp4

        outputs are in folder /fs/vulcan-projects/fsh_track/charles/aa_test_video_temp_clips
            rm -rf /fs/vulcan-projects/fsh_track/charles/aa_test_video_temp_clips/*                
    '''

    temp_video_path = "/fs/vulcan-projects/fsh_track/charles/aa_test_video_temp_clips"

    len_result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", 
                             "/fs/vulcan-projects/fsh_track/charles/aa_test_video_main_short/2_min.mp4"], capture_output=True, text=True).stdout
    len_result = int(float(len_result))

    # DO NOT include -c copy
        # Re-encodes video, which is much slower but avoids the corrupted(?) first few seconds of each clip
    for idx, start in enumerate(range(0, len_result - window_len, window_len - overlap_len)):
        subprocess.run(["ffmpeg", "-i", video_path, "-ss", str(start), "-t", str(window_len), "-reset_timestamps", "1", f"{temp_video_path}/output_{idx:05d}.mp4"])


    return


video_path = "/fs/vulcan-projects/fsh_track/charles/aa_test_video_main_short/2_min.mp4"
model_path = ""
loss_type = ""

window_len = 14
overlap_len = 4
threshold = 0

predict_video(video_path, model_path, loss_type, window_len, overlap_len, threshold)
