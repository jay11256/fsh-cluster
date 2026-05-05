"""Full video prediction script"""

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
    return