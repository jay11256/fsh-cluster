"""Prediction matrix visualization"""

def visualize_matrix(video_path, ground_truth_path, window_len=8, overlap_len=4):
    """
    Generates a visualization based on the prediction matrix

    Args:
        video_path (str): absolute path to the video
        ground_truth_path (str): absolute path to the ground truth label csv
        window_len (num): length of sliding window in seconds
        overlap_len (num): length of overlap on clips in seconds (< window_len)

    """
    return