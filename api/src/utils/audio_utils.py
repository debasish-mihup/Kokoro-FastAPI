import torch
import torchaudio.functional as F

def resample_audio(audio: torch.Tensor, orig_rate: int = 24000, target_rate: int = 24000) -> torch.Tensor:
    """Resample audio tensor to target sample rate.
    
    Args:
        audio: Audio tensor [num_samples] or [channels, num_samples]
        orig_rate: Original sample rate (model native is 24000)
        target_rate: Target sample rate
        
    Returns:
        Resampled audio tensor
    """
    if orig_rate == target_rate:
        return audio
    
    # Ensure audio is 2D for resampling
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        
    resampled = F.resample(audio, orig_freq=orig_rate, new_freq=target_rate)
    
    # Return to original shape
    if resampled.shape[0] == 1:
        resampled = resampled.squeeze(0)
        
    return resampled
