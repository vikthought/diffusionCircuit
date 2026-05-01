#!/usr/bin/env python3
"""
Stimulus ON/OFF Classification Utility

Creates binary labels for each timepoint indicating whether a stimulus is ON or OFF.
This allows analyzing connectivity specifically during stimulus transitions.

Usage:
    from pipeline.utils.stimulus_periods import get_stimulus_mask, segment_by_stimulus
    
    # Get binary mask for a trace
    mask = get_stimulus_mask(trace_length, fps=4, worm_stim_order=[1, 2, 3])
    
    # Segment traces into ON/OFF periods
    on_traces, off_traces = segment_by_stimulus(X, mask)
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

# Default stimulus timing from the NeuroPAL dataset
# stim_times[event_position] = [start_sec, end_sec]
# Event position 0 = first stimulus period, 1 = second, 2 = third
# The actual stimulus TYPE at each position varies by worm (see stims array)
STIM_TIMES_SEC = np.array([
    [60.5, 70.5],    # Event position 0
    [120.5, 130.5],  # Event position 1
    [180.5, 190.5],  # Event position 2
])

DEFAULT_FPS = 4.0


def get_stimulus_mask(
    n_frames: int,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
) -> np.ndarray:
    """
    Create a binary mask indicating stimulus ON (1) vs OFF (0) for each frame.
    
    The mask is stimulus-agnostic: any stimulus = ON, no stimulus = OFF.
    This gives us a binary classification where:
      - ON = during any stimulus presentation period
      - OFF = between stimulus periods (baseline)
    
    Args:
        n_frames: Total number of frames in the trace
        fps: Sampling rate (frames per second)
        stim_times_sec: Array of shape (n_events, 2) with [start, end] in seconds
        
    Returns:
        Binary mask of shape (n_frames,) where 1 = stimulus ON, 0 = OFF
    """
    mask = np.zeros(n_frames, dtype=np.int32)
    
    for start_sec, end_sec in stim_times_sec:
        start_frame = int(np.floor(start_sec * fps))
        end_frame = int(np.ceil(end_sec * fps))
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        
        if start_frame < end_frame:
            mask[start_frame:end_frame] = 1
    
    return mask


def get_transition_frames(
    n_frames: int,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
    window_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get frame indices around stimulus ON and OFF transitions.
    
    These are the "impulse" moments where information about the stimulus
    must propagate through the network.
    
    Args:
        n_frames: Total number of frames
        fps: Sampling rate
        stim_times_sec: Stimulus timing array
        window_frames: Number of frames before/after transition to include
        
    Returns:
        on_frames: Array of frame indices around stimulus onsets
        off_frames: Array of frame indices around stimulus offsets
    """
    on_frames = []
    off_frames = []
    
    for start_sec, end_sec in stim_times_sec:
        on_frame = int(np.round(start_sec * fps))
        off_frame = int(np.round(end_sec * fps))
        
        # Get window around each transition
        for f in range(on_frame - window_frames, on_frame + window_frames + 1):
            if 0 <= f < n_frames:
                on_frames.append(f)
        
        for f in range(off_frame - window_frames, off_frame + window_frames + 1):
            if 0 <= f < n_frames:
                off_frames.append(f)
    
    return np.unique(on_frames), np.unique(off_frames)


def segment_traces_by_stimulus(
    X: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a trace matrix into ON and OFF periods.
    
    Args:
        X: Array of shape (T, n_neurons) - time series
        mask: Binary mask of shape (T,) from get_stimulus_mask
        
    Returns:
        X_on: Traces during stimulus ON periods (T_on, n_neurons)
        X_off: Traces during stimulus OFF periods (T_off, n_neurons)
    """
    T = X.shape[0]
    
    if len(mask) != T:
        raise ValueError(f"Mask length {len(mask)} != trace length {T}")
    
    X_on = X[mask == 1]
    X_off = X[mask == 0]
    
    return X_on, X_off


def summarize_stimulus_periods(
    n_frames: int,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
) -> dict:
    """
    Summarize the stimulus periods in a recording.
    
    Returns:
        Dictionary with summary statistics
    """
    mask = get_stimulus_mask(n_frames, fps, stim_times_sec)
    on_frames, off_frames = get_transition_frames(n_frames, fps, stim_times_sec)
    
    total_duration = n_frames / fps
    on_duration = mask.sum() / fps
    off_duration = (n_frames - mask.sum()) / fps
    
    return {
        'total_frames': n_frames,
        'total_duration_sec': total_duration,
        'on_frames': int(mask.sum()),
        'on_duration_sec': on_duration,
        'off_frames': int(n_frames - mask.sum()),
        'off_duration_sec': off_duration,
        'on_fraction': mask.sum() / n_frames,
        'n_stimuli': len(stim_times_sec),
        'n_transition_frames': len(on_frames) + len(off_frames),
    }


# =============================================================================
# 4-PERIOD SEGMENTATION: NOTHING, ON, SHOWING, OFF
# =============================================================================

class StimulusPeriod:
    """Enum-like class for stimulus period types."""
    NOTHING = 0   # Baseline (no stimulus, not a transition)
    ON = 1        # Onset transition (stimulus just started)
    SHOWING = 2   # Stimulus active (after onset, before offset)
    OFF = 3       # Offset transition (stimulus just ended)
    SKIP = -1     # Initial period to skip (not trusted)


def get_4period_mask(
    n_frames: int,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
    transition_window_sec: float = 2.0,
    skip_initial_sec: float = 10.0,
) -> np.ndarray:
    """
    Create a mask labeling each frame as NOTHING, ON, SHOWING, or OFF.
    
    The 4 periods are:
    - NOTHING (0): Baseline between stimuli, after skip period
    - ON (1): Transition window around stimulus onset
    - SHOWING (2): Stimulus active, after onset transition, before offset
    - OFF (3): Transition window around stimulus offset
    - SKIP (-1): Initial frames to skip (untrusted)
    
    Args:
        n_frames: Total number of frames in the trace
        fps: Sampling rate (frames per second)
        stim_times_sec: Array of shape (n_events, 2) with [start, end] in seconds
        transition_window_sec: Duration of transition window on each side (default: 1s)
        skip_initial_sec: Initial seconds to skip (default: 10s)
        
    Returns:
        Integer mask of shape (n_frames,) with values from StimulusPeriod
    """
    mask = np.full(n_frames, StimulusPeriod.NOTHING, dtype=np.int32)
    
    # Mark initial skip period
    skip_frames = int(np.ceil(skip_initial_sec * fps))
    mask[:skip_frames] = StimulusPeriod.SKIP
    
    # Calculate transition window in frames
    trans_frames = int(np.ceil(transition_window_sec * fps))
    
    for start_sec, end_sec in stim_times_sec:
        # Convert to frame indices
        start_frame = int(np.round(start_sec * fps))
        end_frame = int(np.round(end_sec * fps))
        
        # ON transition: [start - trans, start + trans]
        on_start = max(0, start_frame - trans_frames)
        on_end = min(n_frames, start_frame + trans_frames)
        
        # OFF transition: [end - trans, end + trans]
        off_start = max(0, end_frame - trans_frames)
        off_end = min(n_frames, end_frame + trans_frames)
        
        # SHOWING: between ON and OFF transitions
        showing_start = on_end
        showing_end = off_start
        
        # Apply labels (order matters: more specific overrides general)
        if showing_start < showing_end:
            mask[showing_start:showing_end] = StimulusPeriod.SHOWING
        
        mask[on_start:on_end] = StimulusPeriod.ON
        mask[off_start:off_end] = StimulusPeriod.OFF
    
    return mask


def get_4period_segments(
    n_frames: int,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
    transition_window_sec: float = 2.0,
    skip_initial_sec: float = 10.0,
) -> dict:
    """
    Get contiguous segment ranges for each period type.
    
    Returns:
        Dictionary mapping period names to lists of (start_frame, end_frame) tuples
    """
    mask = get_4period_mask(n_frames, fps, stim_times_sec, 
                            transition_window_sec, skip_initial_sec)
    
    segments = {
        'NOTHING': [],
        'ON': [],
        'SHOWING': [],
        'OFF': [],
    }
    
    # Find contiguous runs of each period type
    for period_name, period_val in [
        ('NOTHING', StimulusPeriod.NOTHING),
        ('ON', StimulusPeriod.ON),
        ('SHOWING', StimulusPeriod.SHOWING),
        ('OFF', StimulusPeriod.OFF),
    ]:
        # Find where this period starts and ends
        is_period = (mask == period_val)
        
        if not is_period.any():
            continue
            
        # Find transitions
        diff = np.diff(is_period.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if is_period[0]:
            starts = np.concatenate([[0], starts])
        if is_period[-1]:
            ends = np.concatenate([ends, [n_frames]])
        
        for s, e in zip(starts, ends):
            segments[period_name].append((s, e))
    
    return segments


def segment_trace_4periods(
    X: np.ndarray,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
    transition_window_sec: float = 2.0,
    skip_initial_sec: float = 10.0,
    min_segment_frames: int = 4,
) -> dict:
    """
    Segment a trace matrix into 4 period types, returning contiguous segments.
    
    Args:
        X: Array of shape (T, n_neurons) - time series
        fps: Sampling rate
        stim_times_sec: Stimulus timing array
        transition_window_sec: Duration of transition window (default: 1s)
        skip_initial_sec: Initial seconds to skip (default: 10s)
        min_segment_frames: Minimum frames for a valid segment (default: 4)
        
    Returns:
        Dictionary mapping period names to lists of arrays, where each array
        is a contiguous segment of shape (T_seg, n_neurons)
    """
    T = X.shape[0]
    segments = get_4period_segments(T, fps, stim_times_sec,
                                    transition_window_sec, skip_initial_sec)
    
    result = {
        'NOTHING': [],
        'ON': [],
        'SHOWING': [],
        'OFF': [],
    }
    
    for period_name, seg_list in segments.items():
        for start, end in seg_list:
            if end - start >= min_segment_frames:
                result[period_name].append(X[start:end])
    
    return result


def summarize_4period_segmentation(
    n_frames: int,
    fps: float = DEFAULT_FPS,
    stim_times_sec: np.ndarray = STIM_TIMES_SEC,
    transition_window_sec: float = 2.0,
    skip_initial_sec: float = 10.0,
) -> dict:
    """
    Summarize the 4-period segmentation for a recording.
    
    Returns:
        Dictionary with summary statistics for each period
    """
    mask = get_4period_mask(n_frames, fps, stim_times_sec,
                            transition_window_sec, skip_initial_sec)
    segments = get_4period_segments(n_frames, fps, stim_times_sec,
                                    transition_window_sec, skip_initial_sec)
    
    summary = {
        'n_frames': n_frames,
        'fps': fps,
        'total_duration_sec': n_frames / fps,
        'skip_initial_sec': skip_initial_sec,
        'transition_window_sec': transition_window_sec,
    }
    
    for period_name, period_val in [
        ('NOTHING', StimulusPeriod.NOTHING),
        ('ON', StimulusPeriod.ON),
        ('SHOWING', StimulusPeriod.SHOWING),
        ('OFF', StimulusPeriod.OFF),
        ('SKIP', StimulusPeriod.SKIP),
    ]:
        n_frames_period = (mask == period_val).sum()
        summary[f'{period_name.lower()}_frames'] = int(n_frames_period)
        summary[f'{period_name.lower()}_duration_sec'] = n_frames_period / fps
        if period_name != 'SKIP':
            summary[f'{period_name.lower()}_n_segments'] = len(segments.get(period_name, []))
    
    return summary


if __name__ == '__main__':
    # Demo/test
    print("Stimulus Period Utility Demo")
    print("=" * 60)
    
    # Typical recording: 232 seconds at 4 Hz = 928 frames
    n_frames = 928
    fps = 4.0
    
    # Test old 2-period mask
    print("\n--- Original 2-Period Mask (ON/OFF) ---")
    mask = get_stimulus_mask(n_frames, fps)
    summary = summarize_stimulus_periods(n_frames, fps)
    print(f"Recording: {n_frames} frames ({n_frames/fps:.1f}s) @ {fps} Hz")
    print(f"  ON duration:  {summary['on_duration_sec']:.1f}s ({summary['on_fraction']*100:.1f}%)")
    print(f"  OFF duration: {summary['off_duration_sec']:.1f}s")
    
    # Test new 4-period segmentation
    print("\n--- New 4-Period Segmentation ---")
    summary_4p = summarize_4period_segmentation(n_frames, fps)
    print(f"Skip initial: {summary_4p['skip_initial_sec']:.1f}s")
    print(f"Transition window: ±{summary_4p['transition_window_sec']:.1f}s")
    print()
    for period in ['NOTHING', 'ON', 'SHOWING', 'OFF', 'SKIP']:
        dur = summary_4p[f'{period.lower()}_duration_sec']
        frames = summary_4p[f'{period.lower()}_frames']
        if period != 'SKIP':
            n_seg = summary_4p[f'{period.lower()}_n_segments']
            print(f"  {period:8s}: {dur:6.1f}s ({frames:4d} frames, {n_seg} segments)")
        else:
            print(f"  {period:8s}: {dur:6.1f}s ({frames:4d} frames)")
    
    # Show actual segments
    print("\n--- Segment Details ---")
    segments = get_4period_segments(n_frames, fps)
    for period_name in ['NOTHING', 'ON', 'SHOWING', 'OFF']:
        segs = segments[period_name]
        print(f"\n{period_name}:")
        for i, (s, e) in enumerate(segs):
            print(f"  {i+1}. frames [{s:4d}-{e:4d}] = [{s/fps:6.1f}s - {e/fps:6.1f}s] ({e-s} frames)")

