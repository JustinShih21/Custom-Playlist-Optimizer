#!/usr/bin/env python3
"""
playlist_comparison_real.py

Generate and compare playlists from Base Model and Advanced Model using real data.
Produces visualizations for audio features and demonstrates the novelty-familiarity curve.
Saves all visualizations to the 'final_visuals' folder.

Modified to handle the base model returning a list instead of DataFrame.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional

# ---------- LOAD DATASET ----------
# Ensure the cleaned CSV is in the same directory or provide full path
df = pd.read_csv('Final_Cleaned_Song_Predictions.csv')

# ---------- IMPORT PLAYLIST GENERATORS ----------
# Base Model expects the DataFrame as the first argument and returns a list
from baseModel import create_playlist as base_create_playlist  
# Advanced Model from new.py returns a DataFrame
from new import create_playlist as advanced_create_playlist, arrange_playlist_by_mood

# ---------- CREATE OUTPUT DIRECTORY ----------
output_dir = 'final_visuals'
os.makedirs(output_dir, exist_ok=True)

def save_figure(title):
    """Save the current figure to the output directory with a cleaned filename"""
    # Clean the title to create a valid filename
    filename = title.lower().replace(' ', '_').replace(':', '').replace(',', '')
    filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '-'])
    filepath = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def convert_base_playlist_to_dataframe(base_results: List, songs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert base model results (list of lists) to DataFrame with audio features.
    
    Args:
        base_results: List of lists from base model
        songs_df: Main songs DataFrame with audio features
    
    Returns:
        DataFrame with base model playlist and audio features
    """
    # Convert results to DataFrame
    # base_results format: [[#, "Song", "Artist", Rating], ...]
    result_df = pd.DataFrame(base_results, columns=["Position", "track_name_x", "artist_name_clean_x", "Rating"])
    
    # Merge with songs dataframe to get audio features
    merged_df = pd.merge(
        result_df,
        songs_df,
        on=['track_name_x', 'artist_name_clean_x'],
        how='left'
    )
    
    return merged_df

def visualize_playlist_flow(base_playlist: pd.DataFrame, advanced_playlist: pd.DataFrame, 
                           feature: str, advanced_mood: str, title: str):
    """
    Create a visualization showing how a specific feature flows through playlists.
    
    Args:
        base_playlist: Base model playlist DataFrame
        advanced_playlist: Advanced model playlist DataFrame
        feature: Feature to visualize (e.g., 'energy', 'tempo')
        advanced_mood: The mood of the advanced playlist
        title: Title for the plot
    """
    plt.figure(figsize=(14, 7))
    
    # Calculate normalized position (0-100%) for varied playlist lengths
    # Base Model
    base_positions = np.linspace(0, 100, len(base_playlist))
    plt.plot(base_positions, base_playlist[feature], 
             marker='o', linestyle='-', linewidth=3,
             label='Base Model', alpha=0.8, color='blue')
    
    # Advanced Model
    adv_positions = np.linspace(0, 100, len(advanced_playlist))
    plt.plot(adv_positions, advanced_playlist[feature], 
             marker='s', linestyle='--', linewidth=3,
             label=f'Advanced Model ({advanced_mood})', alpha=0.8, color='orange')
    
    # Add section markers for advanced model
    # Extract mood from the playlist metadata
    mood = advanced_mood.lower()
    
    # Determine which flow type this is based on mood
    flow_type = None
    if mood in ['party', 'pregame', 'karaoke', 'euphoric']:
        flow_type = 'party'
    elif mood in ['cardio', 'weight_training', 'running']:
        flow_type = 'exercise'
    elif mood in ['deep_work', 'study', 'creative_work']:
        flow_type = 'focus'
    elif mood in ['heartbreak', 'melancholy', 'romantic', 'angry']:
        flow_type = 'emotional'
    elif mood in ['sleep', 'meditation', 'chill', 'sunday_morning', 'beach', 'yoga']:
        flow_type = 'relaxation'
    
    # Section markers based on flow type
    sections = {}
    n = len(advanced_playlist)
    
    if flow_type == 'party':
        sections = {
            'Hook': int(0.15 * n),
            'Ramp-Up': int(0.35 * n),
            'Peak': int(0.65 * n),
            'Cooldown': int(0.8 * n),
            'Finale': n
        }
    elif flow_type == 'exercise':
        sections = {
            'Warm-Up': int(0.2 * n),
            'Workout Core': int(0.65 * n),
            'Power Sprint': int(0.8 * n),
            'Cool-Down': n
        }
    elif flow_type == 'focus':
        sections = {
            'Anchor': int(0.15 * n),
            'Subtle Shift': int(0.4 * n),
            'Deep Flow': int(0.8 * n),
            'Wind-Down': n
        }
    elif flow_type == 'emotional':
        sections = {
            'Prologue': int(0.15 * n),
            'Rising Action': int(0.35 * n),
            'Climax': int(0.65 * n),
            'Resolution': int(0.8 * n),
            'Denouement': n
        }
    elif flow_type == 'relaxation':
        sections = {
            'Drift-In': int(0.15 * n),
            'Soothing Flow': int(0.4 * n),
            'Deep Rest': int(0.8 * n),
            'Comfort Outro': n
        }
    else:
        # Generic sections if flow type can't be determined
        sections = {
            'Section 1': int(0.25 * n),
            'Section 2': int(0.5 * n),
            'Section 3': int(0.75 * n),
            'Section 4': n
        }
    
    # Add vertical lines at section boundaries
    last_pos = 0
    for section, pos in sections.items():
        # Convert to percentage
        pct_pos = (pos / n) * 100
        
        # Add vertical line at section boundary
        if pos < n:  # Don't add line at the very end
            plt.axvline(x=pct_pos, color='gray', linestyle=':', alpha=0.7)
        
        # Add section label in the middle of the section
        mid_pct = (last_pos + pct_pos) / 2
        y_range = plt.ylim()
        plt.text(mid_pct, y_range[0] - 0.05 * (y_range[1] - y_range[0]), 
                section, ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        
        last_pos = pct_pos
    
    plt.title(title, fontsize=15)
    plt.xlabel('Playlist Position (%)', fontsize=12)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure and show
    save_figure(title)
    plt.show()

def visualize_popularity_curve(base_playlist: pd.DataFrame, advanced_playlist: pd.DataFrame, 
                              advanced_mood: str, title: str):
    """
    Create a visualization showing the popularity curve through playlists.
    
    Args:
        base_playlist: Base model playlist DataFrame
        advanced_playlist: Advanced model playlist DataFrame
        advanced_mood: The mood of the advanced playlist
        title: Title for the plot
    """
    plt.figure(figsize=(14, 7))
    
    # Set up color palette
    colors = {'Base Model': '#1f77b4', 'Advanced Model': '#ff7f0e'}
    
    # Add popularity curve bands for ideal reference
    positions = np.linspace(0, 100, 100)
    
    # Create "ideal" popularity bands
    plt.fill_between(positions, 
                     [85 if x < 20 or x > 80 else 40 if 20 <= x < 50 or 65 < x <= 80 else 10 for x in positions],
                     [100 if x < 20 or x > 80 else 70 if 20 <= x < 50 or 65 < x <= 80 else 40 for x in positions],
                     color='lightgray', alpha=0.3, label='Ideal Popularity Zones')
    
    # Annotate the zones
    plt.text(10, 95, "FAMILIAR\nHOOK", ha='center', fontsize=10, weight='bold')
    plt.text(50, 35, "DISCOVERY ZONE", ha='center', fontsize=10, weight='bold')
    plt.text(90, 95, "COMFORT\nCLOSE", ha='center', fontsize=10, weight='bold')
    
    # Calculate normalized position (0-100%) for varied playlist lengths
    # Base Model
    base_positions = np.linspace(0, 100, len(base_playlist))
    plt.plot(base_positions, base_playlist['popularity'], 
             marker='o', linestyle='-', linewidth=3,
             label='Base Model', color=colors['Base Model'], alpha=0.8)
    
    # Advanced Model
    adv_positions = np.linspace(0, 100, len(advanced_playlist))
    plt.plot(adv_positions, advanced_playlist['popularity'], 
             marker='s', linestyle='--', linewidth=3,
             label=f'Advanced Model ({advanced_mood})', color=colors['Advanced Model'], alpha=0.8)
    
    # Add smoothed trendlines
    for positions, data, label, color in [(base_positions, base_playlist['popularity'], 'Base Model', colors['Base Model']),
                                         (adv_positions, advanced_playlist['popularity'], f'Advanced Model ({advanced_mood})', colors['Advanced Model'])]:
        z = np.polyfit(positions, data, 3)
        p = np.poly1d(z)
        smooth_x = np.linspace(0, 100, 100)
        plt.plot(smooth_x, p(smooth_x), 
                linestyle='-', linewidth=2, color=color, alpha=0.4)
    
    plt.title(title, fontsize=15)
    plt.xlabel('Playlist Position (%)', fontsize=12)
    plt.ylabel('Popularity', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure and show
    save_figure(title)
    plt.show()

# Keep all the other functions similar to the original...
# (compare_song_position_shift, generate_feature_distribution, visualize_novelty_familiarity_distribution, etc.)
# Note: These functions will work with DataFrames, no changes needed

def main():
    # ---------- PARAMETERS ----------
    user_id   = 'user_b4beed9bf653604a876fdfd9df59e19c'  # replace with a real user ID
    moods     = ['party', 'cardio', 'chill', 'study', 'heartbreak']
    num_songs = 20

    print(f"Visualizations will be saved to: {os.path.abspath(output_dir)}")

    # ---------- GENERATE SINGLE BASE MODEL PLAYLIST ----------
    print("Generating single Base Model playlist...")
    # Convert the DataFrame to dict before passing to base model
    songs_dict = df.to_dict(orient="list")
    
    # The base model expects user_id, mood (ignored), num_songs
    base_results = base_create_playlist(songs_dict, user_id, num_songs=num_songs)
    if base_results is None:
        print("Error: Could not generate base model playlist!")
        return
    
    # Convert base model results to DataFrame with audio features
    base_playlist = convert_base_playlist_to_dataframe(base_results, df)
    
    # Add metadata
    base_playlist['Model'] = 'Base Model'
    base_playlist['Mood'] = 'baseline'  # Generic label for the mood-agnostic base playlist
    
    # ---------- GENERATE ADVANCED MODEL PLAYLISTS ----------
    advanced_playlists = []
    flow_visualizations = {}  # Store playlists for flow visualization

    for mood in moods:
        print(f"Generating Advanced Model playlist for mood={mood}…")
        # Generate playlist
        playlist = advanced_create_playlist(user_id, mood, num_songs)
        if playlist is None:
            print(f"Warning: Could not generate advanced model playlist for mood={mood}")
            continue
        
        # Arrange the playlist based on mood
        arranged_playlist = arrange_playlist_by_mood(playlist, mood)
        
        # Store metadata
        pl = arranged_playlist.copy()
        pl['Model'] = 'Advanced Model'
        pl['Mood'] = mood
        advanced_playlists.append(pl)
        
        # Store for flow visualization
        flow_visualizations[mood] = pl

    # ---------- COMBINE ALL RESULTS ----------
    # Create a list to hold all playlists for combined analysis
    all_playlists = []
    
    # Add base playlist once for each mood comparison
    for mood in moods:
        if mood in flow_visualizations:  # Only compare if advanced playlist exists
            base_copy = base_playlist.copy()
            base_copy['Comparison_Mood'] = mood  # Track which mood it's being compared against
            all_playlists.append(base_copy)
    
    # Add all advanced playlists
    all_playlists.extend(advanced_playlists)
    
    # Combine into single DataFrame
    combined_df = pd.concat(all_playlists, ignore_index=True)

    # Handle missing columns and score mapping
    score_col = None
    if 'expected_ratings' in combined_df.columns:
        score_col = 'expected_ratings'
        combined_df = combined_df.rename(columns={'expected_ratings': 'score'})
    elif 'predicted_score' in combined_df.columns:
        score_col = 'predicted_score'
        combined_df = combined_df.rename(columns={'predicted_score': 'score'})
    elif 'Rating' in combined_df.columns:
        score_col = 'Rating'
        combined_df = combined_df.rename(columns={'Rating': 'score'})

    # Ensure popularity column exists - use it if available, otherwise create a default
    if 'popularity' not in combined_df.columns:
        print("Warning: 'popularity' column not found, using default values")
        combined_df['popularity'] = 50  # Default value

    # ---------- MAP MOOD TO CATEGORY LABELS ----------
    mood_map = {
        'party': 'Party',
        'cardio': 'Exercise',
        'study': 'Focus & Productivity',
        'heartbreak': 'Emotional',
        'chill': 'Relaxation',
        'baseline': 'Baseline'  # For base model
    }
    combined_df['Category'] = combined_df['Mood'].map(mood_map)

    # ---------- FEATURES TO COMPARE ----------
    # Only include features that exist in both datasets
    features = []
    for feat in ['energy', 'tempo', 'danceability', 'valence', 'popularity', 'score']:
        if feat in combined_df.columns:
            features.append(feat)
    
    print(f"Available features for comparison: {features}")

    # ---------- PLOTTING ----------
    
    # 1. Basic feature boxplots by mood and model
    for feat in features:
        plt.figure(figsize=(16, 8))
        
        # Create separate plots for each mood comparison
        fig, axes = plt.subplots(1, len(moods), figsize=(20, 6), sharey=True)
        
        # Handle the case where we might have fewer moods than expected
        if len(moods) == 1:
            axes = [axes]
        
        for i, mood in enumerate(moods):
            if mood in flow_visualizations and i < len(axes):
                # Get data for this mood comparison
                comparison_data = combined_df[
                    (combined_df['Model'] == 'Advanced Model') & (combined_df['Mood'] == mood) |
                    (combined_df['Model'] == 'Base Model') & (combined_df['Comparison_Mood'] == mood)
                ]
                
                if len(comparison_data) > 0:
                    sns.boxplot(
                        data=comparison_data,
                        x='Model',
                        y=feat,
                        ax=axes[i],
                        palette='Set2'
                    )
                    axes[i].set_title(f"{mood_map.get(mood, mood.title())}")
                    axes[i].set_xlabel('')
                    if i == 0:
                        axes[i].set_ylabel(feat.replace('_', ' ').title())
                    else:
                        axes[i].set_ylabel('')
        
        fig.suptitle(f"{feat.replace('_', ' ').title()} Comparison Across Moods", fontsize=16)
        plt.tight_layout()
        
        # Save figure and show
        save_figure(f"{feat}_comparison_across_moods")
        plt.show()
    
    # 2. Flow visualizations for each mood
    for mood, advanced_playlist in flow_visualizations.items():
        # Only create visualizations if the features exist
        if 'energy' in base_playlist.columns and 'energy' in advanced_playlist.columns:
            visualize_playlist_flow(
                base_playlist,
                advanced_playlist,
                'energy',
                mood,
                f'Energy Flow Comparison: Base vs Advanced ({mood_map.get(mood, mood.title())})'
            )
        
        if 'tempo' in base_playlist.columns and 'tempo' in advanced_playlist.columns:
            visualize_playlist_flow(
                base_playlist,
                advanced_playlist,
                'tempo',
                mood,
                f'Tempo Flow Comparison: Base vs Advanced ({mood_map.get(mood, mood.title())})'
            )
        
        if 'valence' in base_playlist.columns and 'valence' in advanced_playlist.columns:
            visualize_playlist_flow(
                base_playlist,
                advanced_playlist,
                'valence',
                mood,
                f'Valence Flow Comparison: Base vs Advanced ({mood_map.get(mood, mood.title())})'
            )
        
        if 'popularity' in base_playlist.columns and 'popularity' in advanced_playlist.columns:
            visualize_popularity_curve(
                base_playlist,
                advanced_playlist,
                mood,
                f'Popularity Curve Comparison: Base vs Advanced ({mood_map.get(mood, mood.title())})'
            )
    
    # 3. Create a comprehensive popularity distribution comparison if possible
    if 'popularity' in combined_df.columns:
        plt.figure(figsize=(15, 8))
        
        # Plot all advanced model playlists
        for mood, advanced_pl in flow_visualizations.items():
            if 'popularity' in advanced_pl.columns:
                adv_positions = np.linspace(0, 100, len(advanced_pl))
                plt.plot(adv_positions, advanced_pl['popularity'], 
                        linestyle='--', linewidth=2,
                        label=f'Advanced ({mood})', alpha=0.7)
        
        # Plot base model (only once)
        if 'popularity' in base_playlist.columns:
            base_positions = np.linspace(0, 100, len(base_playlist))
            plt.plot(base_positions, base_playlist['popularity'], 
                    linestyle='-', linewidth=3,
                    label='Base Model', color='black', alpha=0.9)
        
        # Add ideal popularity zones
        positions = np.linspace(0, 100, 100)
        plt.fill_between(positions, 
                         [85 if x < 20 or x > 80 else 40 if 20 <= x < 50 or 65 < x <= 80 else 10 for x in positions],
                         [100 if x < 20 or x > 80 else 70 if 20 <= x < 50 or 65 < x <= 80 else 40 for x in positions],
                         color='lightgray', alpha=0.3, label='Ideal Popularity Zones')
        
        plt.title('Comprehensive Popularity Curve Comparison', fontsize=16)
        plt.xlabel('Playlist Position (%)', fontsize=12)
        plt.ylabel('Popularity', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save figure and show
        save_figure('comprehensive_popularity_comparison')
        plt.show()
    
    print(f"\nAll visualizations and data saved to: {os.path.abspath(output_dir)}")
    print("You can use these files for presentations or further analysis.")

if __name__ == '__main__':
    main()