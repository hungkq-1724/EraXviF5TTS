# https://pypi.org/project/phonemizer/3.0.1/
# https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
# USE this: https://github.com/thewh1teagle/phonemizer-fork.git
import torch
import numpy as np
from viphoneme import vi2IPA
from tqdm import tqdm

def text_to_phonemes(text):
    """
    Convert Vietnamese text to IPA phonemes using viphoneme.
    Improved handling of edge cases and different input types.
    
    Args:
        text: String or list of strings/lists to convert
        
    Returns:
        List of phoneme sequences
    """
    # Handle different input types
    if isinstance(text, list):
        result = []
        for t in tqdm(text):
            # Handle case where t is itself a list
            if isinstance(t, list):
                # Convert nested list to string
                t_str = ' '.join(str(item) for item in t if item is not None)
                result.append(vi2IPA(t_str).strip().split(" "))
            else:
                # Make sure t is a string before passing to vi2IPA
                result.append(vi2IPA(str(t)).strip().split(" "))
        return result
    # Single string case
    return vi2IPA(str(text).strip().split(" "))

from phonemizer import phonemize
def text_to_phonemes_espeak(text):

    phoneme_seq = []
    # Step 1: Convert text to phonemes using phonemizer instead of viphoneme
    if isinstance(text_input, str):
        # Phonemize text with espeak
        phoneme_seq = phonemize(str(text_input), language="vi").strip().split(" ")
        # Make it a list of lists for batch processing
        phoneme_seq = [phoneme_seq]
    elif isinstance(text_input, list):
        # Handle batch of texts
        phoneme_seq = []
        for text in text_input:
            if text:
                phoneme_seq.append(phonemize(str(text), language="vi").strip().split(" "))
            else:
                phoneme_seq.append([])
    else:
        raise ValueError(f"Unsupported input type: {type(text_input)}")
        
    return phoneme_seq
    
def phoneme_to_indices(phoneme_seq, phoneme_map=None):
    """
    Convert phoneme sequence to indices, handling edge cases better.
    If phoneme_map is provided, uses it. Otherwise creates a new one.
    
    Args:
        phoneme_seq: List of phoneme sequences or a single sequence
        phoneme_map: Optional mapping of phonemes to indices
        
    Returns:
        Tuple of (indices, phoneme_map)
    """
    # Build phoneme map if not provided
    if phoneme_map is None:
        # Create a set of unique phonemes from all sequences
        unique_phonemes = set()
        
        # Handle different input types
        if isinstance(phoneme_seq, list):
            for seq in phoneme_seq:
                if seq: # Check for empty sequences
                    unique_phonemes.update(seq)
        elif phoneme_seq:  # Single sequence
            unique_phonemes.update(phoneme_seq)
            
        # Create mapping with 0 reserved for padding/unknown
        phoneme_map = {p: i+1 for i, p in enumerate(sorted(unique_phonemes))}
    
    # Convert to indices
    if isinstance(phoneme_seq, list):
        # Multiple sequences
        indices = []
        for seq in phoneme_seq:
            # Handle empty sequences with a placeholder
            if not seq:
                indices.append([0])  # Use padding token
            else:
                indices.append([phoneme_map.get(p, 0) for p in seq])
    else:
        # Single sequence
        indices = [phoneme_map.get(p, 0) for p in (phoneme_seq or [])]

    return indices, phoneme_map
    
def create_phoneme_embedding(phoneme_map, embed_dim=192):
    """
    Create embedding layer for phonemes.
    
    Args:
        phoneme_map: Dictionary mapping phonemes to indices
        embed_dim: Embedding dimension
        
    Returns:
        torch.nn.Embedding layer
    """
    num_phonemes = len(phoneme_map) + 1  # +1 for padding/unknown
    return torch.nn.Embedding(num_phonemes, embed_dim)

def get_durations_from_alignment(alignment):
    """
    Extract duration of each token from alignment matrix.
    
    Args:
        alignment: Alignment matrix [b, nt, mel_len]
    
    Returns:
        durations: Duration of each token [b, nt]
    """
    return alignment.sum(dim=2)  # Sum across the mel dimension

def phonemes_to_mel_alignment(phoneme_tensor, phoneme_mask, mel_spec, model):
    """
    Calculate alignment between phonemes and mel-spectrogram frames.
    
    Args:
        phoneme_tensor: Tensor of phoneme indices [b, nt]
        phoneme_mask: Mask for phoneme tensor [b, nt]
        mel_spec: Mel spectrogram [b, n_mels, mel_len]
        model: Model with alignment network
        
    Returns:
        alignment: Hard alignment matrix [b, nt, mel_len]
        similarity: Similarity matrix [b, nt, mel_len]
    """
    # Get similarity from alignment network
    similarity = model.alignment_network(phoneme_tensor, phoneme_mask, mel_spec.permute(0, 2, 1))
    
    # Find monotonic alignment
    alignment = monotonic_alignment_search(similarity)
    
    return alignment, similarity

def monotonic_alignment_search(similarity_matrix):
    b, nt, mel_len = similarity_matrix.shape
    device = similarity_matrix.device
    dtype = similarity_matrix.dtype
    
    # Sử dụng phép tính GPU-accelerated: cumulative sum
    # Thay vì vòng lặp lồng nhau tính DP
    cum_sim = torch.cumsum(similarity_matrix, dim=2)  # Tính tổng tích lũy theo chiều mel
    
    # Khởi tạo ma trận alignment
    alignments = torch.zeros_like(similarity_matrix)
    
    # Tính durations trực tiếp bằng argmax
    # Tìm vị trí tốt nhất để chuyển sang phoneme tiếp theo
    for i in range(b):
        start_frame = 0
        
        for n in range(nt - 1):  # Xử lý tất cả phoneme trừ phoneme cuối
            # Tìm vị trí tốt nhất để kết thúc phoneme hiện tại
            scores = torch.zeros(mel_len, device=device, dtype=dtype)
            scores[:start_frame] = -float('inf')  # Đảm bảo không quay lại
            
            if n < nt - 1:
                # Tính điểm cho mỗi vị trí kết thúc có thể
                for end_frame in range(start_frame, mel_len):
                    # Điểm cho phoneme hiện tại kết thúc tại end_frame
                    curr_score = cum_sim[i, n, end_frame] - (cum_sim[i, n, start_frame-1] if start_frame > 0 else 0)
                    # Điểm cho phoneme tiếp theo bắt đầu từ end_frame+1
                    next_score = cum_sim[i, n+1, -1] - (cum_sim[i, n+1, end_frame] if end_frame < mel_len-1 else 0)
                    scores[end_frame] = curr_score + next_score
            
            # Tìm vị trí tốt nhất
            best_end = torch.argmax(scores).item()
            
            # Đánh dấu tất cả frames từ start_frame đến best_end thuộc về phoneme hiện tại
            alignments[i, n, start_frame:best_end+1] = 1
            
            # Cập nhật start_frame cho phoneme tiếp theo
            start_frame = best_end + 1
            if start_frame >= mel_len:
                break  # Không còn frames
        
        # Phoneme cuối cùng lấy tất cả frames còn lại
        if start_frame < mel_len:
            alignments[i, -1, start_frame:] = 1
    
    return alignments

'''
def monotonic_alignment_search(similarity_matrix):
    """
    Lightning-fast monotonic alignment search algorithm that remains on GPU.
    
    Args:
        similarity_matrix: Similarity matrix [b, nt, mel_len]
    
    Returns:
        hard_alignment: Alignments [b, nt, mel_len]
    """
    b, nt, mel_len = similarity_matrix.shape
    device = similarity_matrix.device
    dtype = similarity_matrix.dtype
    
    # Initialize forward pass
    forward = torch.full((b, nt, mel_len), float('-inf'), device=device, dtype=dtype)
    forward[:, 0, 0] = similarity_matrix[:, 0, 0]
    
    # Fill first row and column - vectorized across batch dimension
    for t in range(1, mel_len):
        forward[:, 0, t] = forward[:, 0, t-1] + similarity_matrix[:, 0, t]
    
    for n in range(1, nt):
        forward[:, n, 0] = forward[:, n-1, 0] + similarity_matrix[:, n, 0]
    
    # Fill the rest of the DP table - vectorized across batch
    for n in range(1, nt):
        for t in range(1, mel_len):
            forward[:, n, t] = similarity_matrix[:, n, t] + torch.maximum(
                forward[:, n-1, t],  # from above
                forward[:, n, t-1]   # from left
            )
    
    # Create output - directly on GPU
    alignments = torch.zeros_like(similarity_matrix)
    
    # Process each sample in the batch - sequentially but all on GPU
    for i in range(b):
        # Store the path for this sample
        path = []
        
        # Start from the end position
        n, t = nt-1, mel_len-1
        path.append((n, t))
        
        # Use a fixed number of iterations for safety
        max_steps = nt + mel_len
        
        # Backtracking on GPU to find the path
        for step in range(max_steps):
            if n == 0 and t == 0:
                break
                
            if n == 0:
                t -= 1
            elif t == 0:
                n -= 1
            else:
                # Compare values and move accordingly
                if forward[i, n-1, t] > forward[i, n, t-1]:
                    n -= 1
                else:
                    t -= 1
            
            if n >= 0 and t >= 0:
                path.append((n, t))
        
        # Reverse the path to start from (0,0)
        path.reverse()
        
        # Process the path to create hard alignment
        prev_n = -1
        
        for p_n, p_t in path:
            # When we move to a new phoneme, all frames from prev_t to current t
            # are assigned to the current phoneme
            if p_n != prev_n:
                # Mark this position as belonging to current phoneme
                alignments[i, p_n, p_t] = 1
                prev_n = p_n
            else:
                # Continue marking frames for the current phoneme
                alignments[i, p_n, p_t] = 1
        
        # Ensure we don't have any phoneme with zero duration by checking each phoneme
        for n in range(nt):
            if alignments[i, n].sum() == 0:
                # Find a position to assign to this phoneme
                # Assign it the frame right after the previous phoneme
                assigned = False
                
                # Try to find where the previous phoneme ends
                if n > 0:
                    for t in range(mel_len-1, -1, -1):
                        if alignments[i, n-1, t] == 1:
                            # Assign the next frame (if available) to current phoneme
                            if t+1 < mel_len:
                                alignments[i, n, t+1] = 1
                                assigned = True
                            # If no next frame, share the last frame with previous phoneme
                            else:
                                alignments[i, n, t] = 1
                                assigned = True
                            break
                
                # If still not assigned (first phoneme or couldn't find prev end),
                # assign the first available frame
                if not assigned:
                    for t in range(mel_len):
                        if not any(alignments[i, :, t] > 0):
                            alignments[i, n, t] = 1
                            break
                    else:
                        # If all frames are taken, assign to the first frame
                        # This is a fallback to ensure non-zero duration
                        alignments[i, n, 0] = 1
    
    return alignments
'''