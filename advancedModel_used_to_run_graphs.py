# -*- coding: utf-8 -*-
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tabulate import tabulate
import argparse
import os
from datetime import datetime
from typing import List, Dict, Optional

class AdvancedPlaylistArranger:
    def __init__(self, playlist: pd.DataFrame):
        self.playlist = playlist.copy()
        self.original_order = playlist.copy()
        
    def arrange_by_tempo(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by tempo (BPM)"""
        return self.playlist.sort_values('tempo', ascending=ascending)
    
    def arrange_by_energy(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by energy level"""
        return self.playlist.sort_values('energy', ascending=ascending)
    
    def arrange_by_popularity(self, ascending: bool = False) -> pd.DataFrame:
        """Arrange songs by popularity"""
        return self.playlist.sort_values('popularity', ascending=ascending)
    
    def arrange_by_key(self) -> pd.DataFrame:
        """Arrange songs by musical key (if available)"""
        if 'key' in self.playlist.columns:
            # Convert key numbers to musical notation
            key_map = {
                0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E',
                5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A',
                10: 'A#', 11: 'B'
            }
            self.playlist['key_name'] = self.playlist['key'].map(key_map)
            return self.playlist.sort_values('key_name')
        return self.playlist
    
    def arrange_by_danceability(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by danceability"""
        return self.playlist.sort_values('danceability', ascending=ascending)
    
    def arrange_by_valence(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by valence (mood)"""
        return self.playlist.sort_values('valence', ascending=ascending)
    
    def arrange_by_acousticness(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by acousticness"""
        return self.playlist.sort_values('acousticness', ascending=ascending)
    
    def arrange_by_instrumentalness(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by instrumentalness"""
        return self.playlist.sort_values('instrumentalness', ascending=ascending)
    
    def arrange_by_loudness(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by loudness"""
        return self.playlist.sort_values('loudness', ascending=ascending)
    
    def arrange_by_speechiness(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by speechiness"""
        return self.playlist.sort_values('speechiness', ascending=ascending)
    
    def arrange_by_duration(self, ascending: bool = True) -> pd.DataFrame:
        """Arrange songs by duration"""
        if 'duration_ms' in self.playlist.columns:
            return self.playlist.sort_values('duration_ms', ascending=ascending)
        return self.playlist
    
    def arrange_by_genre(self) -> pd.DataFrame:
        """Arrange songs by genre"""
        return self.playlist.sort_values('track_genre')
    
    def arrange_by_artist(self) -> pd.DataFrame:
        """Arrange songs by artist name"""
        return self.playlist.sort_values('artist_name_clean_x')
    
    def arrange_by_name(self) -> pd.DataFrame:
        """Arrange songs by track name"""
        return self.playlist.sort_values('track_name_x')
    
    def arrange_by_advanced_party_flow(self) -> pd.DataFrame:
        """
        Arrange songs for an optimized party experience with 5 sections:
        1. Hook (1-3): Familiar bangers (high popularity)
        2. Ramp-Up (4-7): Energy build with high valence
        3. Peak (8-12): Discovery zone with mid-popularity
        4. Cooldown Tease (13-15): Brief energy dip
        5. Finale (16-18): Comfort finale with high popularity
        """
        if len(self.playlist) < 10:
            # For small playlists, just do basic energy flow
            return self.arrange_by_flow()
            
        # Sort by popularity to pick top songs
        sorted_by_popularity = self.playlist.sort_values('popularity', ascending=False)
        
        # Get popularity percentiles
        self.playlist['popularity_percentile'] = self.playlist['popularity'].rank(pct=True) * 100
        
        # Create the sections
        num_songs = len(self.playlist)
        
        # Calculate section sizes proportionally to the playlist length
        hook_size = max(1, int(0.15 * num_songs))
        ramp_size = max(1, int(0.20 * num_songs))
        peak_size = max(2, int(0.30 * num_songs))
        cooldown_size = max(1, int(0.15 * num_songs))
        finale_size = max(1, num_songs - hook_size - ramp_size - peak_size - cooldown_size)
        
        # 1. Hook: High popularity bangers
        hook = self.playlist[self.playlist['popularity_percentile'] >= 90].sort_values('popularity', ascending=False).head(hook_size)
        if len(hook) < hook_size:  # If not enough high popularity songs
            hook = sorted_by_popularity.head(hook_size)
        
        # 2. Ramp-Up: Energy build with high valence
        ramp_candidates = self.playlist[
            ~self.playlist.index.isin(hook.index) & 
            (self.playlist['valence'] >= 0.7)
        ].sort_values('energy')
        if len(ramp_candidates) < ramp_size:
            # If not enough high valence songs, use any remaining songs
            ramp_candidates = self.playlist[~self.playlist.index.isin(hook.index)]
        ramp_up = ramp_candidates.head(ramp_size)
        
        # 3. Peak: Mid-popularity discovery tracks
        peak_candidates = self.playlist[
            ~self.playlist.index.isin(hook.index) & 
            ~self.playlist.index.isin(ramp_up.index) & 
            (self.playlist['popularity_percentile'].between(40, 70))
        ].sort_values(['energy', 'danceability'], ascending=False)
        if len(peak_candidates) < peak_size:
            # If not enough mid-popularity songs, use any high energy songs
            peak_candidates = self.playlist[
                ~self.playlist.index.isin(hook.index) & 
                ~self.playlist.index.isin(ramp_up.index)
            ].sort_values('energy', ascending=False)
        peak = peak_candidates.head(peak_size)
        
        # 4. Cooldown: Brief energy dip
        cooldown_candidates = self.playlist[
            ~self.playlist.index.isin(hook.index) & 
            ~self.playlist.index.isin(ramp_up.index) & 
            ~self.playlist.index.isin(peak.index)
        ].sort_values('energy')
        cooldown = cooldown_candidates.head(cooldown_size)
        
        # 5. Finale: High popularity comfort tracks
        finale_candidates = self.playlist[
            ~self.playlist.index.isin(hook.index) & 
            ~self.playlist.index.isin(ramp_up.index) & 
            ~self.playlist.index.isin(peak.index) & 
            ~self.playlist.index.isin(cooldown.index) & 
            (self.playlist['popularity_percentile'] >= 85)
        ].sort_values('popularity', ascending=False)
        if len(finale_candidates) < finale_size:
            # If not enough high popularity songs left, use any remaining songs
            finale_candidates = self.playlist[
                ~self.playlist.index.isin(hook.index) & 
                ~self.playlist.index.isin(ramp_up.index) & 
                ~self.playlist.index.isin(peak.index) & 
                ~self.playlist.index.isin(cooldown.index)
            ].sort_values('popularity', ascending=False)
        finale = finale_candidates.head(finale_size)
        
        hook = hook.sort_values('tempo', ascending=True)           # 1. Hook: low to high
        ramp_up = ramp_up.sort_values('tempo', ascending=False)    # 2. Ramp-Up: high to low
        peak = peak.sort_values('tempo', ascending=True)           # 3. Peak: low to high
        cooldown = cooldown.sort_values('tempo', ascending=False)  # 4. Cooldown: high to low
        finale = finale.sort_values('tempo', ascending=True)
        # Combine all sections
        return pd.concat([hook, ramp_up, peak, cooldown, finale])
        
    def arrange_by_advanced_exercise_flow(self) -> pd.DataFrame:
        """
        Arrange songs for an optimized exercise experience with 4 sections:
        1. Warm-Up (1-4): Progressive tempo (100-110 BPM)
        2. Workout Core (5-12): Alternating blocks of high-BPM power songs and novelty tracks
        3. Power Sprint (13-15): Ultra-high BPM tracks for final push
        4. Cool-Down (16-18): Descending BPM
        """
        if len(self.playlist) < 10:
            # For small playlists, just do basic workout flow
            return self.arrange_by_workout_flow()
            
        # Get popularity percentiles
        self.playlist['popularity_percentile'] = self.playlist['popularity'].rank(pct=True) * 100
        
        # Create the sections
        num_songs = len(self.playlist)
        
        # Calculate section sizes proportionally to the playlist length
        warmup_size = max(1, int(0.20 * num_songs))
        core_size = max(3, int(0.45 * num_songs))
        sprint_size = max(1, int(0.15 * num_songs))
        cooldown_size = max(1, num_songs - warmup_size - core_size - sprint_size)
        
        # 1. Warm-Up: Progressive tempo (100-110 BPM)
        warmup_candidates = self.playlist[
            (self.playlist['tempo'].between(100, 110))
        ].sort_values('tempo')
        if len(warmup_candidates) < warmup_size:
            # If not enough songs in ideal tempo range, use lowest tempo songs
            warmup_candidates = self.playlist.sort_values('tempo')
        warmup = warmup_candidates.head(warmup_size)
        
        # 2. Workout Core: Alternating blocks of high-BPM power songs and novelty tracks
        remaining = self.playlist[~self.playlist.index.isin(warmup.index)]
        
        # High-BPM power songs
        power_candidates = remaining[remaining['tempo'] > 130].sort_values('energy', ascending=False)
        if len(power_candidates) < core_size * 0.75:  # If not enough high-BPM songs
            power_candidates = remaining.sort_values(['tempo', 'energy'], ascending=False)
        
        # Novel tracks (mid-popularity)
        novel_candidates = remaining[
            (remaining['popularity_percentile'].between(40, 70)) & 
            ~remaining.index.isin(power_candidates.index)
        ]
        if len(novel_candidates) < core_size * 0.25:  # If not enough novel tracks
            novel_candidates = remaining[~remaining.index.isin(power_candidates.index)]
        
        # Combine power and novel tracks in alternating blocks
        power_block_size = 3
        novel_block_size = 1
        core = pd.DataFrame()
        power_songs_used = 0
        novel_songs_used = 0
        
        while len(core) < core_size:
            # Add a block of power songs
            power_block = power_candidates.iloc[power_songs_used:power_songs_used+power_block_size]
            power_songs_used += len(power_block)
            core = pd.concat([core, power_block])
            
            # Add a novel track if we haven't reached the core size yet
            if len(core) < core_size and novel_songs_used < len(novel_candidates):
                novel_block = novel_candidates.iloc[novel_songs_used:novel_songs_used+novel_block_size]
                novel_songs_used += len(novel_block)
                core = pd.concat([core, novel_block])
            
            # Break if we've used all available songs
            if power_songs_used >= len(power_candidates) and novel_songs_used >= len(novel_candidates):
                break
        
        # Trim to core_size if needed
        core = core.head(core_size)
        
        # 3. Power Sprint: Ultra-high BPM tracks for final push
        sprint_candidates = self.playlist[
            ~self.playlist.index.isin(warmup.index) & 
            ~self.playlist.index.isin(core.index) & 
            (self.playlist['tempo'] >= 140)
        ].sort_values('energy', ascending=False)
        if len(sprint_candidates) < sprint_size:
            # If not enough ultra-high BPM tracks, use high energy songs
            sprint_candidates = self.playlist[
                ~self.playlist.index.isin(warmup.index) & 
                ~self.playlist.index.isin(core.index)
            ].sort_values(['tempo', 'energy'], ascending=False)
        sprint = sprint_candidates.head(sprint_size)
        
        # 4. Cool-Down: Descending BPM
        cooldown_candidates = self.playlist[
            ~self.playlist.index.isin(warmup.index) & 
            ~self.playlist.index.isin(core.index) & 
            ~self.playlist.index.isin(sprint.index)
        ].sort_values('tempo', ascending=False)  # Sort descending to get gradually lower BPM
        cooldown = cooldown_candidates.head(cooldown_size)
        
        warmup = warmup.sort_values('tempo', ascending=True)           # 1. Hook: low to high
        core = core.sort_values('tempo', ascending=False)    # 2. Ramp-Up: high to low
        sprint = sprint.sort_values('tempo', ascending=True)           # 3. Peak: low to high
        cooldown = cooldown.sort_values('tempo', ascending=False)  # 4. Cooldown: high to low
        # Combine all sections
        return pd.concat([warmup, core, sprint, cooldown])
    
    def arrange_by_advanced_focus_flow(self) -> pd.DataFrame:
        """
        Arrange songs for an optimized focus experience with 4 sections:
        1. Anchor (1-3): Consistent groove (90-110 BPM)
        2. Subtle Shift (4-8): Textural novelty with mid-high popularity
        3. Deep Flow (9-14): Steady-state with minimal variation
        4. Wind-Down (15-18): Familiar return with top focus tracks
        """
        if len(self.playlist) < 10:
            # For small playlists, just do basic deep work flow
            return self.arrange_by_flow()
            
        # Get popularity percentiles
        self.playlist['popularity_percentile'] = self.playlist['popularity'].rank(pct=True) * 100
        
        # Create the sections
        num_songs = len(self.playlist)
        
        # Calculate section sizes proportionally to the playlist length
        anchor_size = max(1, int(0.15 * num_songs))
        shift_size = max(1, int(0.25 * num_songs))
        flow_size = max(2, int(0.40 * num_songs))
        winddown_size = max(1, num_songs - anchor_size - shift_size - flow_size)
        
        # 1. Anchor: Consistent groove (90-110 BPM)
        anchor_candidates = self.playlist[
            (self.playlist['tempo'].between(90, 110)) &
            (self.playlist['instrumentalness'] >= 0.3)
        ].sort_values(['tempo', 'instrumentalness'], ascending=False)
        if len(anchor_candidates) < anchor_size:
            # If not enough instrumental tracks in ideal tempo range, use any instrumental tracks
            anchor_candidates = self.playlist.sort_values('instrumentalness', ascending=False)
        anchor = anchor_candidates.head(anchor_size)
        
        # 2. Subtle Shift: Textural novelty with mid-high popularity
        shift_candidates = self.playlist[
            ~self.playlist.index.isin(anchor.index) & 
            (self.playlist['popularity_percentile'].between(50, 80))
        ].sort_values('instrumentalness', ascending=False)
        if len(shift_candidates) < shift_size:
            # If not enough mid-high popularity songs, use any remaining instrumental tracks
            shift_candidates = self.playlist[
                ~self.playlist.index.isin(anchor.index)
            ].sort_values('instrumentalness', ascending=False)
        shift = shift_candidates.head(shift_size)
        
        # 3. Deep Flow: Steady-state with minimal variation
        # Find tracks with similar energy/tempo to maintain flow
        avg_energy = self.playlist['energy'].mean()
        avg_tempo = self.playlist['tempo'].mean()
        
        # Calculate distance from average energy/tempo
        self.playlist['energy_distance'] = (self.playlist['energy'] - avg_energy).abs()
        self.playlist['tempo_distance'] = (self.playlist['tempo'] - avg_tempo).abs()
        self.playlist['flow_score'] = self.playlist['energy_distance'] + self.playlist['tempo_distance']
        
        flow_candidates = self.playlist[
            ~self.playlist.index.isin(anchor.index) & 
            ~self.playlist.index.isin(shift.index)
        ].sort_values('flow_score')  # Sort by lowest deviation from average energy/tempo
        flow = flow_candidates.head(flow_size)
        
        # 4. Wind-Down: Familiar return with top focus tracks
        winddown_candidates = self.playlist[
            ~self.playlist.index.isin(anchor.index) & 
            ~self.playlist.index.isin(shift.index) & 
            ~self.playlist.index.isin(flow.index) & 
            (self.playlist['popularity_percentile'] >= 80)
        ].sort_values(['instrumentalness', 'popularity'], ascending=[False, False])
        if len(winddown_candidates) < winddown_size:
            # If not enough popular songs left, use any remaining songs
            winddown_candidates = self.playlist[
                ~self.playlist.index.isin(anchor.index) & 
                ~self.playlist.index.isin(shift.index) & 
                ~self.playlist.index.isin(flow.index)
            ].sort_values('popularity', ascending=False)
        winddown = winddown_candidates.head(winddown_size)
        
        anchor = anchor.sort_values('tempo', ascending=True)           # 1. Hook: low to high
        shift = shift.sort_values('tempo', ascending=False)    # 2. Ramp-Up: high to low
        flow = flow.sort_values('tempo', ascending=True)           # 3. Peak: low to high
        winddown = winddown.sort_values('tempo', ascending=False)  # 4. Cooldown: high to low
        # Combine all sections
        return pd.concat([anchor, shift, flow, winddown])
    
    def arrange_by_advanced_emotional_flow(self) -> pd.DataFrame:
        """
        Arrange songs for an optimized emotional experience with 5 sections:
        1. Prologue (1-3): Safe neutral (mid-valence)
        2. Rising Action (4-7): Intensifying mood (decreasing valence)
        3. Climax (8-12): Peak emotion
        4. Resolution (13-15): Gentle lift (increasing valence)
        5. Denouement (16-18): Comfort close (familiar, hopeful)
        """
        if len(self.playlist) < 10:
            # For small playlists, just do basic emotional flow
            return self.arrange_by_flow()
            
        # Get popularity percentiles
        self.playlist['popularity_percentile'] = self.playlist['popularity'].rank(pct=True) * 100
        
        # Create the sections
        num_songs = len(self.playlist)
        
        # Calculate section sizes proportionally to the playlist length
        prologue_size = max(1, int(0.15 * num_songs))
        rising_size = max(1, int(0.20 * num_songs))
        climax_size = max(1, int(0.30 * num_songs))
        resolution_size = max(1, int(0.15 * num_songs))
        denouement_size = max(1, num_songs - prologue_size - rising_size - climax_size - resolution_size)
        
        # 1. Prologue: Safe neutral (mid-valence)
        prologue_candidates = self.playlist[
            (self.playlist['valence'].between(0.4, 0.6))
        ].sort_values('popularity', ascending=False)
        if len(prologue_candidates) < prologue_size:
            # If not enough mid-valence songs, use popular songs
            prologue_candidates = self.playlist.sort_values('popularity', ascending=False)
        prologue = prologue_candidates.head(prologue_size)
        
        # 2. Rising Action: Intensifying mood (decreasing valence)
        rising_candidates = self.playlist[
            ~self.playlist.index.isin(prologue.index)
        ].sort_values('valence', ascending=False)  # Start with higher valence and will descend
        rising = rising_candidates.head(rising_size)
        
        # 3. Climax: Peak emotion
        # Look for songs with low valence and high energy for emotional impact
        climax_candidates = self.playlist[
            ~self.playlist.index.isin(prologue.index) & 
            ~self.playlist.index.isin(rising.index) & 
            (self.playlist['valence'] < 0.4) &
            (self.playlist['energy'] > 0.6)
        ]
        if len(climax_candidates) < climax_size:
            # If not enough peak emotion songs, use any low valence songs
            climax_candidates = self.playlist[
                ~self.playlist.index.isin(prologue.index) & 
                ~self.playlist.index.isin(rising.index)
            ].sort_values('valence')
        climax = climax_candidates.head(climax_size)
        
        # 4. Resolution: Gentle lift (increasing valence)
        resolution_candidates = self.playlist[
            ~self.playlist.index.isin(prologue.index) & 
            ~self.playlist.index.isin(rising.index) & 
            ~self.playlist.index.isin(climax.index) & 
            (self.playlist['acousticness'] > 0.3)  # Prefer acoustic tracks for resolution
        ].sort_values('valence')  # Start with lower valence and will ascend
        if len(resolution_candidates) < resolution_size:
            # If not enough acoustic songs, use any remaining songs
            resolution_candidates = self.playlist[
                ~self.playlist.index.isin(prologue.index) & 
                ~self.playlist.index.isin(rising.index) & 
                ~self.playlist.index.isin(climax.index)
            ].sort_values('valence')
        resolution = resolution_candidates.head(resolution_size)
        
        # 5. Denouement: Comfort close (familiar, hopeful)
        denouement_candidates = self.playlist[
            ~self.playlist.index.isin(prologue.index) & 
            ~self.playlist.index.isin(rising.index) & 
            ~self.playlist.index.isin(climax.index) & 
            ~self.playlist.index.isin(resolution.index) & 
            (self.playlist['valence'] > 0.5) &  # Hopeful
            (self.playlist['popularity_percentile'] >= 80)  # Familiar
        ].sort_values('popularity', ascending=False)
        if len(denouement_candidates) < denouement_size:
            # If not enough hopeful, familiar songs, use any remaining songs
            denouement_candidates = self.playlist[
                ~self.playlist.index.isin(prologue.index) & 
                ~self.playlist.index.isin(rising.index) & 
                ~self.playlist.index.isin(climax.index) & 
                ~self.playlist.index.isin(resolution.index)
            ].sort_values(['valence', 'popularity'], ascending=[False, False])
        denouement = denouement_candidates.head(denouement_size)
        
        prologue = prologue.sort_values('tempo', ascending=True)           # 1. Hook: low to high
        rising = rising.sort_values('tempo', ascending=False)    # 2. Ramp-Up: high to low
        climax = climax.sort_values('tempo', ascending=True)           # 3. Peak: low to high
        resolution = resolution.sort_values('tempo', ascending=False)  # 4. Cooldown: high to low
        # Combine all sections
        return pd.concat([prologue, rising, climax, resolution, denouement])
    
    def arrange_by_advanced_relaxation_flow(self) -> pd.DataFrame:
        """
        Arrange songs for an optimized relaxation experience with 4 sections:
        1. Drift-In (1-3): Slow fade (60-80 BPM, low dynamics)
        2. Soothing Flow (4-8): Textural novelties
        3. Deep Rest (9-14): Uniform warmth (constant energy/tempo)
        4. Comfort Outro (15-18): Safe return to familiar relaxation tracks
        """
        if len(self.playlist) < 10:
            # For small playlists, just do basic flow
            return self.arrange_by_flow()
            
        # Get popularity percentiles
        self.playlist['popularity_percentile'] = self.playlist['popularity'].rank(pct=True) * 100
        
        # Create the sections
        num_songs = len(self.playlist)
        
        # Calculate section sizes proportionally to the playlist length
        drift_size = max(1, int(0.15 * num_songs))
        soothing_size = max(1, int(0.25 * num_songs))
        deep_size = max(2, int(0.40 * num_songs))
        comfort_size = max(1, num_songs - drift_size - soothing_size - deep_size)
        
        # 1. Drift-In: Slow fade (60-80 BPM, low dynamics)
        drift_candidates = self.playlist[
            (self.playlist['tempo'].between(60, 80)) &
            (self.playlist['energy'] < 0.5)
        ].sort_values(['tempo', 'energy'])
        if len(drift_candidates) < drift_size:
            # If not enough slow, low-energy songs, use any low-energy songs
            drift_candidates = self.playlist.sort_values(['energy', 'tempo'])
        drift = drift_candidates.head(drift_size)
        
        # 2. Soothing Flow: Textural novelties
        # Look for instrumental tracks with some uniqueness/novelty
        self.playlist['uniqueness'] = (
            self.playlist['instrumentalness'] * 0.7 + 
            (1 - self.playlist['popularity_percentile']/100) * 0.3  # Lower popularity = more unique
        )
        soothing_candidates = self.playlist[
            ~self.playlist.index.isin(drift.index) & 
            (self.playlist['energy'] < 0.6)  # Keep it relaxing
        ].sort_values('uniqueness', ascending=False)
        soothing = soothing_candidates.head(soothing_size)
        
        # 3. Deep Rest: Uniform warmth (constant energy/tempo)
        # Find tracks with similar energy/tempo to maintain uniform feeling
        avg_energy = self.playlist['energy'].mean()
        avg_tempo = self.playlist['tempo'].mean()
        
        # Calculate distance from average energy/tempo but prioritize lower energy
        self.playlist['energy_consistency'] = (
            (self.playlist['energy'] - min(avg_energy, 0.4)).abs() * 2 +  # Lower energy is better for relaxation
            (self.playlist['tempo'] - min(avg_tempo, 90)).abs() / 10  # Lower tempo is better
        )
        
        deep_candidates = self.playlist[
            ~self.playlist.index.isin(drift.index) & 
            ~self.playlist.index.isin(soothing.index) & 
            (self.playlist['energy'] < 0.5)  # Keep energy low for deep rest
        ].sort_values('energy_consistency')
        deep = deep_candidates.head(deep_size)
        
        # 4. Comfort Outro: Safe return to familiar relaxation tracks
        comfort_candidates = self.playlist[
            ~self.playlist.index.isin(drift.index) & 
            ~self.playlist.index.isin(soothing.index) & 
            ~self.playlist.index.isin(deep.index) & 
            (self.playlist['popularity_percentile'] >= 80)  # Familiar
        ].sort_values(['energy', 'popularity'])  # Low energy, high popularity
        if len(comfort_candidates) < comfort_size:
            # If not enough familiar songs left, use any remaining songs
            comfort_candidates = self.playlist[
                ~self.playlist.index.isin(drift.index) & 
                ~self.playlist.index.isin(soothing.index) & 
                ~self.playlist.index.isin(deep.index)
            ].sort_values('energy')
        comfort = comfort_candidates.head(comfort_size)
        
        drift = drift.sort_values('tempo', ascending=True)           # 1. Hook: low to high
        soothing = soothing.sort_values('tempo', ascending=False)    # 2. Ramp-Up: high to low
        deep = deep.sort_values('tempo', ascending=True)           # 3. Peak: low to high
        comfort = comfort.sort_values('tempo', ascending=False)  # 4. Cooldown: high to low
        # Combine all sections
        return pd.concat([drift, soothing, deep, comfort])
    

    def print_playlist(self, title: str = "Arranged Playlist", mood: str = "party"):
        """Print the playlist in a nice format with section labels"""
        table_data = []
        num_songs = len(self.playlist)

        # Section breakpoints and titles per mood - based on exact specifications
        section_boundaries = {
            'party': [1, 3, 7, 12, 17],  # Hook(1-3), Ramp-Up(4-7), Peak(8-12), Cooldown(13-17), Finale(18-20)
            'exercise': [1, 4, 12, 15],  # Warm-Up(1-4), Workout Core(5-12), Power Sprint(13-15), Cool-Down(16-18)
            'focus': [1, 3, 8, 14],      # Anchor(1-3), Subtle Shift(4-8), Deep Flow(9-14), Wind-Down(15-18)
            'emotional': [1, 3, 7, 12, 15],  # Prologue(1-3), Rising Action(4-7), Climax(8-12), Resolution(13-15), Denouement(16-18)
            'relaxation': [1, 3, 8, 14]   # Drift-In(1-3), Soothing Flow(4-8), Deep Rest(9-14), Comfort Outro(15-18)
        }

        section_titles_map = {
            'party': ['🎉 HOOK', '📈 RAMP-UP', '🔥 PEAK', '😌 COOLDOWN', '🎵 FINALE'],
            'exercise': ['🔆 WARM-UP', '💪 WORKOUT CORE', '⚡ POWER SPRINT', '🧘 COOL-DOWN'],
            'focus': ['🎯 ANCHOR', '🌊 SUBTLE SHIFT', '🧠 DEEP FLOW', '🌅 WIND-DOWN'],
            'emotional': ['📖 PROLOGUE', '📈 RISING ACTION', '💫 CLIMAX', '💭 RESOLUTION', '🍃 DENOUEMENT'],
            'relaxation': ['☁️ DRIFT-IN', '🌊 SOOTHING FLOW', '💤 DEEP REST', '🏡 COMFORT OUTRO']
        }

        # Get current mood boundaries and titles
        section_titles = section_titles_map.get(mood, ['SECTION 1', 'SECTION 2', 'SECTION 3'])
        
        # Calculate boundaries based on number of sections
        num_sections = len(section_titles)
        if mood in section_boundaries:
            # Use predefined boundaries if they exist
            boundaries = section_boundaries[mood]
        else:
            # Calculate even boundaries for unknown moods
            boundaries = [int((i + 1) * num_songs / num_sections) for i in range(num_sections - 1)]

        current_section = 0

        for idx, (_, song) in enumerate(self.playlist.iterrows(), 1):
            # if this is the very first song, or we just crossed a boundary
            if idx == 1 or (current_section < len(boundaries) and idx > boundaries[current_section]):
                table_data.append([
                    '---',
                    section_titles[current_section],
                    '---','---','---','---','---'
                ])
                current_section += 1

            # Determine energy indicator
            energy_val = song.get('energy', 0)
            if energy_val < 0.4:
                energy_indicator = '🔋'
            elif energy_val < 0.7:
                energy_indicator = '🔋🔋'
            else:
                energy_indicator = '🔋🔋🔋'

            row = [
                idx,
                song.get('track_name_x', 'N/A'),
                song.get('artist_name_clean_x', 'N/A'),
                song.get('track_genre', 'N/A'),
                f"{song.get('popularity', 0):.0f}",
                f"{energy_indicator} {energy_val:.2f}",
                f"{song.get('tempo', 0):.0f}"
            ]
            table_data.append(row)

        # Add the last section title if we haven't shown all sections
        if current_section < len(section_titles):
            table_data.append([
                '---',
                section_titles[current_section],
                '---','---','---','---','---'
            ])

        print(f"\n{title}:")
        print(tabulate(
            table_data,
            headers=['#', 'Track', 'Artist', 'Genre', 'Popularity', 'Energy', 'BPM'],
            tablefmt='fancy_grid'
        ))


    def arrange_by_flow(self) -> pd.DataFrame:
        """
        Basic flow arrangement that works for smaller playlists:
        1. Start with moderate energy
        2. Build up to high energy
        3. Peak in the middle
        4. Gradually decrease energy
        5. End with calm songs
        """
        # Create energy-based segments
        low_energy = self.playlist[self.playlist['energy'] < 0.4]
        mid_energy = self.playlist[(self.playlist['energy'] >= 0.4) & (self.playlist['energy'] < 0.7)]
        high_energy = self.playlist[self.playlist['energy'] >= 0.7]
        
        # Arrange each segment
        low_energy = low_energy.sort_values('energy', ascending=True)
        mid_energy = mid_energy.sort_values('energy', ascending=True)
        high_energy = high_energy.sort_values('energy', ascending=True)
        
        # Combine segments in the desired order
        return pd.concat([
            mid_energy.iloc[:len(mid_energy)//2],  # Start with moderate energy
            high_energy,                           # Build up to high energy
            mid_energy.iloc[len(mid_energy)//2:],  # Moderate energy
            low_energy                            # End with calm songs
        ])
    
    def arrange_by_workout_flow(self) -> pd.DataFrame:
        """Arrange songs for a workout playlist:
        1. Warm-up (moderate energy)
        2. High-intensity (high energy)
        3. Cool-down (decreasing energy)
        """
        # Create energy-based segments
        warmup = self.playlist[(self.playlist['energy'] >= 0.5) & (self.playlist['energy'] < 0.7)]
        high_intensity = self.playlist[self.playlist['energy'] >= 0.7]
        cooldown = self.playlist[self.playlist['energy'] < 0.5]
        
        # Arrange each segment
        warmup = warmup.sort_values('energy', ascending=True)
        high_intensity = high_intensity.sort_values('energy', ascending=False)
        cooldown = cooldown.sort_values('energy', ascending=False)
        
        # Combine segments
        return pd.concat([warmup, high_intensity, cooldown])
    
    def arrange_by_party_flow(self) -> pd.DataFrame:
        """Simple party flow (less sophisticated than advanced_party_flow):
        1. Start with popular, high-energy songs
        2. Mix in some variety
        3. End with well-known hits
        """
        # Create segments based on popularity and energy
        high_energy = self.playlist[self.playlist['energy'] >= 0.7]
        popular = self.playlist[self.playlist['popularity'] >= 70]
        others = self.playlist[
            (self.playlist['energy'] < 0.7) & 
            (self.playlist['popularity'] < 70)
        ]
        
        # Arrange each segment
        high_energy = high_energy.sort_values('popularity', ascending=False)
        popular = popular.sort_values('energy', ascending=False)
        others = others.sort_values('popularity', ascending=False)
        
        # Combine segments
        return pd.concat([high_energy, others, popular])

# Define mood/activity categories and their characteristics
mood_categories = {
    'pregame': {
        'min_danceability': 0.7,
        'min_energy': 0.8,
        'min_valence': 0.6,
        'min_popularity': 60,
        'description': '🍻 Getting ready to go out!',
        'flow': 'party'
    },
    'party': {
        'min_danceability': 0.8,
        'min_energy': 0.7,
        'min_valence': 0.6,
        'min_tempo': 115,
        'description': '🎉 Full party mode!',
        'flow': 'party'
    },
    'karaoke': {
        'min_popularity': 70,
        'max_instrumentalness': 0.2,
        'min_valence': 0.5,
        'description': '🎤 Popular sing-along hits',
        'flow': 'party'
    },
    'cardio': {
        'min_tempo': 130,
        'min_energy': 0.8,
        'min_danceability': 0.7,
        'description': '🏃‍♀️ High-tempo cardio workout',
        'flow': 'exercise'
    },
    'weight_training': {
        'min_energy': 0.8,
        'min_valence': 0.6,
        'min_tempo': 100,
        'description': '🏋️‍♀️ Pumped up for lifting',
        'flow': 'exercise'
    },
    'yoga': {
        'max_energy': 0.4,
        'min_instrumentalness': 0.4,
        'max_tempo': 100,
        'description': '🧘‍♀️ Peaceful yoga flow',
        'flow': 'relaxation'
    },
    'running': {
        'min_tempo': 140,
        'min_energy': 0.8,
        'min_valence': 0.5,
        'description': '🏃 Perfect running rhythm',
        'flow': 'exercise'
    },
    'deep_work': {
        'min_instrumentalness': 0.5,
        'max_energy': 0.5,
        'max_speechiness': 0.1,
        'description': '💻 Deep focus and concentration',
        'flow': 'focus'
    },
    'study': {
        'max_energy': 0.5,
        'min_instrumentalness': 0.3,
        'max_tempo': 120,
        'description': '📚 Background music for studying',
        'flow': 'focus'
    },
    'creative_work': {
        'min_instrumentalness': 0.3,
        'max_energy': 0.6,
        'min_valence': 0.4,
        'description': '🎨 Inspiring creative flow',
        'flow': 'focus'
    },
    'heartbreak': {
        'max_valence': 0.3,
        'min_acousticness': 0.4,
        'max_energy': 0.5,
        'description': '💔 Processing heartbreak',
        'flow': 'emotional'
    },
    'melancholy': {
        'max_valence': 0.4,
        'max_energy': 0.4,
        'min_acousticness': 0.3,
        'description': '😢 Feeling blue and reflective',
        'flow': 'emotional'
    },
    'euphoric': {
        'min_valence': 0.8,
        'min_energy': 0.7,
        'min_danceability': 0.6,
        'description': '🌟 Pure joy and excitement',
        'flow': 'party'
    },
    'romantic': {
        'min_valence': 0.5,
        'max_energy': 0.6,
        'min_acousticness': 0.3,
        'description': '❤️ Love and romance',
        'flow': 'emotional'
    },
    'angry': {
        'min_energy': 0.8,
        'max_valence': 0.4,
        'min_loudness': -7.0,
        'description': '😠 Release that anger',
        'flow': 'emotional'
    },
    'sleep': {
        'max_energy': 0.2,
        'max_loudness': -12.0,
        'min_instrumentalness': 0.4,
        'description': '😴 Peaceful sleep sounds',
        'flow': 'relaxation'
    },
    'meditation': {
        'max_energy': 0.3,
        'min_instrumentalness': 0.5,
        'max_tempo': 80,
        'description': '🧘 Mindful meditation',
        'flow': 'relaxation'
    },
    'chill': {
        'max_energy': 0.5,
        'min_valence': 0.4,
        'max_tempo': 110,
        'description': '😌 Relaxed vibes',
        'flow': 'relaxation'
    },
    'sunday_morning': {
        'min_valence': 0.5,
        'max_energy': 0.5,
        'min_acousticness': 0.3,
        'description': '☀️ Lazy Sunday morning',
        'flow': 'relaxation'
    },
    'beach': {
        'min_valence': 0.6,
        'max_energy': 0.7,
        'min_acousticness': 0.2,
        'description': '🏖️ Beach day relaxation',
        'flow': 'relaxation'
    }
}

# Expanded mood mapping for user input
mood_keywords = {
    'pregame': ['pregame', 'pre-game', 'pre game', 'getting ready', 'pre-party', 'pre party', 'fireball', 'vodka'],
    'party': ['party', 'club', 'dancing', 'celebration', 'festive', 'dance', 'drinks', 'drunk', 'drinking'],
    'karaoke': ['karaoke', 'singing', 'sing along', 'singalong', 'sing'],
    'cardio': ['cardio', 'hiit', 'spinning', 'cycling', 'elliptical', 'running', 'jog', 'jogging', 'sprint'],
    'weight_training': ['weights', 'lifting', 'gym', 'strength', 'workout', 'weightlifting', 'weight training', 'weight_training'],
    'yoga': ['yoga', 'stretching', 'flexibility', 'stretch', 'yoga class', 'yoga session', 'yoga practice'],
    'running': ['running', 'jog', 'jogging', 'sprint', 'run', 'running shoes', 'running shoes', 'running shoes'],
    'deep_work': ['deep_work','focus', 'concentrate', 'coding', 'programming', 'writing', 'work', 'deep work', 'concentration', 'focused'],
    'study': ['study', 'studying', 'homework', 'learning', 'reading', 'book', 'books', 'textbook', 'textbooks'],
    'creative_work': ['creative_work', 'creative','creating', 'drawing', 'painting', 'art', 'design', 'sketch', 'sketching', 'sketchpad', 'sketchpad'],
    'heartbreak': ['breakup', 'break up', 'heartbroken', 'missing', 'miss', 'missed', 'ex', 'exes', 'ex girlfriend', 'ex boyfriend', 'heartbreak',],
    'melancholy': ['sad', 'down', 'blue', 'depressed', 'lonely', 'alone', 'disappointed', 'melancholy'],
    'euphoric': ['happy', 'excited', 'joy', 'ecstatic', 'amazing', 'great', 'amazing', 'lit', 'euphoric'],
    'romantic': ['love', 'romantic', 'date', 'romance', 'intimate', 'boyfriend', 'marriage', 'romantic'],
    'angry': ['angry', 'mad', 'furious', 'rage', 'upset', 'frustrated', 'pissed', 'hate', 'angry'],
    'sleep': ['sleep', 'bedtime', 'night', 'rest', 'sleeping', 'night'],
    'meditation': ['meditate', 'meditation', 'mindfulness', 'zen', 'peace'],
    'chill': ['chill', 'relax', 'relaxing', 'mellow', 'unwind'],
    'sunday_morning': ['morning', 'breakfast', 'sunday', 'wake up', 'sunrise', 'lazy', 'sunday_morning'],
    'beach': ['beach', 'summer', 'sunny', 'pool', 'vacation', 'sun', 'waves', 'sand', 'ocean', 'high uv']
}

def get_mood_from_input(user_input):
    user_input = user_input.lower().strip()
    # First try to parse as a number
    try:
        choice = int(user_input)
        if 1 <= choice <= len(mood_categories):
            return list(mood_categories.keys())[choice - 1]
    except ValueError:
        pass
    
    # Split input into words
    input_words = set(user_input.split())
    
    # Try to match keywords as whole words
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            # Split multi-word keywords
            keyword_words = set(keyword.split())
            if keyword_words.issubset(input_words):
                return mood
    
    return None

def analyze_playlist(playlist):
    """Analyze and print statistics about the generated playlist"""
    stats = {
        'Average Energy': playlist['energy'].mean(),
        'Average Tempo': playlist['tempo'].mean(),
        'Average Danceability': playlist['danceability'].mean(),
        'Average Valence': playlist['valence'].mean(),
        'Average Popularity': playlist['popularity'].mean(),
        'Duration (minutes)': playlist['duration_ms'].sum() / 60000,
        'Genre Distribution': playlist['track_genre'].value_counts().to_dict()
    }
    
    print("\n📊 Playlist Analysis:")
    print("=" * 40)
    for stat, value in stats.items():
        if stat != 'Genre Distribution':
            print(f"{stat}: {value:.2f}")
    
    print("\n🎵 Top Genres:")
    for genre, count in sorted(stats['Genre Distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {genre}: {count} songs")

def create_playlist(user_id, mood, num_songs=20):
    # Read the cleaned dataset
    df = pd.read_csv('Final_Cleaned_Song_Predictions.csv')
    
    # 1. Filter songs based on mood
    filtered_songs = df.copy()
    for param, value in mood_categories[mood].items():
        if param in ['description', 'flow']:  # Skip non-filter parameters
            continue
        feature = param.split('_', 1)[1]
        if param.startswith('min_'):
            filtered_songs = filtered_songs[filtered_songs[feature] >= value]
        else:
            filtered_songs = filtered_songs[filtered_songs[feature] <= value]
    filtered_songs = filtered_songs.reset_index(drop=True)

    # Adjust playlist size if too few
    if len(filtered_songs) < num_songs:
        print(f"Warning: Only found {len(filtered_songs)} songs matching the criteria. Adjusting to {len(filtered_songs)} songs.")
        num_songs = len(filtered_songs)

    # 2. Compute expected ratings
    user_ratings = filtered_songs[user_id]
    #avg_ratings = filtered_songs[[c for c in filtered_songs.columns if c.startswith('user_')]].mean(axis=1)
    #expected_ratings = 0.7 * user_ratings + 0.3 * avg_ratings

    # 3. Build and solve the Gurobi model
    model = gp.Model("playlist_optimization")
    model.setParam('OutputFlag', 1)
    x = model.addVars(len(filtered_songs), vtype=GRB.BINARY, name="song_selection")

    # Objective: maximize sum(expected_ratings[i] * x[i])
    model.setObjective(gp.quicksum(user_ratings[i] * x[i] for i in range(len(filtered_songs))),
                       GRB.MAXIMIZE)

    # Constraint 1: exactly num_songs selected
    model.addConstr(gp.quicksum(x[i] for i in range(len(filtered_songs))) == num_songs,
                    name="size_exact")

    # Constraint 2: at most 2 per artist
    for artist in filtered_songs['artist_name_clean_x'].unique():
        idxs = filtered_songs[filtered_songs['artist_name_clean_x'] == artist].index.tolist()
        model.addConstr(gp.quicksum(x[i] for i in idxs) <= 2,
                        name=f"artist_{artist}")

    model.optimize()

    # 4. Fallback if infeasible
    if model.status == GRB.INFEASIBLE:
        print("⚠️ Model infeasible—relaxing to size-only constraint.")
        model.remove(model.getConstrs())
        model.update()
        model.addConstr(gp.quicksum(x[i] for i in range(len(filtered_songs))) == num_songs,
                        name="size_only")
        model.optimize()

    # 5. Extract solution if available
    if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        selected_indices = [i for i in range(len(filtered_songs)) if x[i].X > 0.5]
        playlist = filtered_songs.iloc[selected_indices]
        analyze_playlist(playlist)
        return playlist
    else:
        print("❌ No feasible solution found even after fallback.")
        return None

def save_playlist(playlist, mood, output_dir='playlists'):
    """Save the playlist to a CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/{mood}_playlist_{timestamp}.csv"
    playlist.to_csv(filename, index=False)
    return filename

def arrange_playlist_by_mood(playlist, mood):
    """
    Arrange the playlist using the appropriate flow pattern based on the mood
    """
    # Map the mood to its flow type
    flow_mapping = {
        'pregame': 'party',
        'party': 'party',
        'karaoke': 'party',
        'cardio': 'exercise',
        'weight_training': 'exercise',
        'yoga': 'relaxation',
        'running': 'exercise',
        'deep_work': 'focus',
        'study': 'focus',
        'creative_work': 'focus',
        'heartbreak': 'emotional',
        'melancholy': 'emotional',
        'euphoric': 'party',
        'romantic': 'emotional',
        'angry': 'emotional',
        'sleep': 'relaxation',
        'meditation': 'relaxation',
        'chill': 'relaxation',
        'sunday_morning': 'relaxation',
        'beach': 'relaxation'
    }
    
    # Get the flow pattern for the mood
    flow_type = flow_mapping.get(mood, 'flow')
    
    arranger = AdvancedPlaylistArranger(playlist)
    
    # Use the appropriate arrangement method based on flow type
    if flow_type == 'party':
        arranged = arranger.arrange_by_advanced_party_flow()
    elif flow_type == 'exercise':
        arranged = arranger.arrange_by_advanced_exercise_flow()
    elif flow_type == 'focus':
        arranged = arranger.arrange_by_advanced_focus_flow()
    elif flow_type == 'emotional':
        arranged = arranger.arrange_by_advanced_emotional_flow()
    elif flow_type == 'relaxation':
        arranged = arranger.arrange_by_advanced_relaxation_flow()
    else:
        arranged = arranger.arrange_by_flow()  # Default to basic flow
    
    # Create a new arranger with the arranged playlist
    display_arranger = AdvancedPlaylistArranger(arranged)
    # Print with the correct mood
    display_arranger.print_playlist(f"Playlist arranged for {mood}", flow_type)
    
    return arranged

def main():
    parser = argparse.ArgumentParser(description='Generate and arrange a personalized playlist')
    
    # Main command options
    parser.add_argument('--mode', choices=['generate', 'arrange', 'both'], default='both',
                      help='Operation mode: generate a new playlist, arrange an existing one, or both')
    
    # Generate playlist options
    parser.add_argument('--mood', type=str, help='Specify a mood or activity (e.g., workout, party, study)')
    parser.add_argument('--user_id', type=str, help='Specify a user ID')
    parser.add_argument('--num_songs', type=int, default=20, help='Number of songs in the playlist (default: 20)')
    parser.add_argument('--output_dir', type=str, default='playlists', help='Directory to save playlists (default: playlists)')
    parser.add_argument('--no_save', action='store_true', help='Do not save the playlist to a file')
    
    # Arrange playlist options
    parser.add_argument('--input', type=str, help='Input CSV file for arrangement (required if mode is "arrange")')
    parser.add_argument('--arrangement', type=str, 
                      choices=['tempo', 'energy', 'popularity', 'key', 'danceability',
                              'valence', 'acousticness', 'instrumentalness', 'loudness',
                              'speechiness', 'duration', 'genre', 'artist', 'name',
                              'flow', 'workout', 'party', 'advanced_party', 'advanced_exercise',
                              'advanced_focus', 'advanced_emotional', 'advanced_relaxation', 'smart'],
                      default='smart',
                      help='How to arrange the playlist (default: smart)')
    parser.add_argument('--ascending', action='store_true', help='Sort in ascending order (if applicable)')
    
    args = parser.parse_args()
    
    # Check for required arguments based on mode
    if args.mode == 'arrange' and not args.input:
        parser.error("--input is required when mode is 'arrange'")
    
    playlist = None
    filename = None
    
    print("\n🎵 Welcome to the Advanced Mood & Activity Playlist Generator & Arranger! 🎵")
    
    # Generate a playlist
    if args.mode in ['generate', 'both']:
        # Read the dataset to get valid user IDs
        df = pd.read_csv('Final_Cleaned_Song_Predictions.csv')
        valid_user_ids = [col for col in df.columns if col.startswith('user_')]
        
        if args.user_id:
            user_id = args.user_id
        else:
            print("\nAvailable user IDs:")
            for uid in valid_user_ids:
                print(f"- {uid}")
            user_id = input("\nPlease enter the user ID: ").strip()
        
        if user_id not in valid_user_ids:
            print(f"\nError: Invalid user ID: {user_id}")
            return
        
        print(f"\nGenerating playlist for user: {user_id}")
        
        if args.mood:
            mood = args.mood.lower()
            if mood not in mood_categories:
                print(f"\nInvalid mood: {mood}")
                print("\nAvailable moods:")
                for m, params in mood_categories.items():
                    print(f"- {m}: {params['description']}")
                return
        else:
            print("\nCategories:")
            categories = {
                '🎉 PARTY & SOCIAL': ['pregame', 'party', 'karaoke', 'euphoric'],
                '💪 EXERCISE': ['cardio', 'weight_training', 'running'],
                '🧠 FOCUS & PRODUCTIVITY': ['deep_work', 'study', 'creative_work'],
                '💭 EMOTIONAL': ['heartbreak', 'melancholy', 'romantic', 'angry'],
                '😌 RELAXATION': ['sleep', 'meditation', 'chill', 'sunday_morning', 'beach', 'yoga']
            }
            for cat, moods in categories.items():
                print(f"\n{cat}:")
                for m in moods:
                    print(f"- {mood_categories[m]['description']}")
            
            user_feeling = input(f"\nHow does {user_id} feel or what are they doing? ").strip()
            mood = get_mood_from_input(user_feeling)
            if not mood:
                print("\nI couldn't quite match their mood. Here are all options:")
                for idx, (m, params) in enumerate(mood_categories.items(), 1):
                    print(f"{idx}. {m}: {params['description']}")
                choice = input("\nEnter the number or name of your choice: ").strip()
                mood = get_mood_from_input(choice) or 'chill'
                print(f"Defaulting to '{mood}'.")
        
        playlist = create_playlist(user_id, mood, args.num_songs)
        
        if playlist is not None:
            # If in 'both' mode, we'll arrange it before saving
            if args.mode == 'generate' and not args.no_save:
                filename = save_playlist(playlist, mood, args.output_dir)
                print(f"\nPlaylist saved to: {filename}")
        else:
            print("\nFailed to generate a playlist. Please try again with different parameters.")
            return
    
    # Arrange an existing playlist
    if args.mode in ['arrange', 'both']:
        if args.mode == 'arrange':
            # Read the playlist from file
            try:
                playlist = pd.read_csv(args.input)
                filename = args.input
                
                # Try to extract mood from filename
                mood = os.path.basename(filename).split('_')[0]
                if mood not in mood_categories:
                    mood = None
            except Exception as e:
                print(f"\nError reading playlist file: {e}")
                return
        
        # For 'smart' arrangement, use the mood-based flow
        if args.arrangement == 'smart' and 'mood' in locals() and mood in mood_categories:
            print(f"\nUsing smart arrangement based on '{mood}' mood...")
            arranged_playlist = arrange_playlist_by_mood(playlist, mood)
            arrangement_description = f"optimized {mood_categories[mood]['flow']} pattern"
        else:
            # Create arranger with the playlist
            arranger = AdvancedPlaylistArranger(playlist)
            
            # Get arrangement method
            arrangement = args.arrangement if args.arrangement != 'smart' else 'flow'
            
            # Apply the selected arrangement
            arrangement_methods = {
                'tempo': arranger.arrange_by_tempo,
                'energy': arranger.arrange_by_energy,
                'popularity': arranger.arrange_by_popularity,
                'key': arranger.arrange_by_key,
                'danceability': arranger.arrange_by_danceability,
                'valence': arranger.arrange_by_valence,
                'acousticness': arranger.arrange_by_acousticness,
                'instrumentalness': arranger.arrange_by_instrumentalness,
                'loudness': arranger.arrange_by_loudness,
                'speechiness': arranger.arrange_by_speechiness,
                'duration': arranger.arrange_by_duration,
                'genre': arranger.arrange_by_genre,
                'artist': arranger.arrange_by_artist,
                'name': arranger.arrange_by_name,
                'flow': arranger.arrange_by_flow,
                'workout': arranger.arrange_by_workout_flow,
                'party': arranger.arrange_by_party_flow,
                'advanced_party': arranger.arrange_by_advanced_party_flow,
                'advanced_exercise': arranger.arrange_by_advanced_exercise_flow,
                'advanced_focus': arranger.arrange_by_advanced_focus_flow,
                'advanced_emotional': arranger.arrange_by_advanced_emotional_flow,
                'advanced_relaxation': arranger.arrange_by_advanced_relaxation_flow
            }
            
            if arrangement in ['flow', 'workout', 'party', 
                              'advanced_party', 'advanced_exercise', 'advanced_focus', 
                              'advanced_emotional', 'advanced_relaxation']:
                arranged_playlist = arrangement_methods[arrangement]()
            else:
                arranged_playlist = arrangement_methods[arrangement](args.ascending)
            
            arrangement_description = arrangement
            
            # Create arranger object with arranged playlist for printing
            display_arranger = AdvancedPlaylistArranger(arranged_playlist)
            
            # Print the arranged playlist with appropriate mood
            if 'mood' in locals() and mood in mood_categories:
                display_arranger.print_playlist(f"Playlist arranged by {arrangement_description}", mood)
            else:
                display_arranger.print_playlist(f"Playlist arranged by {arrangement_description}")
        
        # Save the arranged playlist if needed
        if not args.no_save:
            if args.mode == 'both':
                # For 'both' mode, we've already created a playlist and know the mood
                arranged_filename = save_playlist(arranged_playlist, f"{mood}_arranged", args.output_dir)
            else:
                # For 'arrange' mode, extract filename base or use input filename
                base_name = os.path.splitext(os.path.basename(filename))[0]
                arranged_filename = f"{args.output_dir}/{base_name}_arranged_{args.arrangement}.csv"
                arranged_playlist.to_csv(arranged_filename, index=False)
            
            print(f"\nArranged playlist saved to: {arranged_filename}")

if __name__ == "__main__":
    main()