"""Clip prediction script"""

def predict_clip(video_path, model_path):
    """
    Generates a prediction vector given a clip

    Args:
        video_path (str): absolute path to the video
        model_path (str): absolute path to the model being run

    Returns:
        prediction_vector (torch.Tensor): 6x1 vector of prediction values
    """
    return