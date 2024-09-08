import random

class CommentaryGenerator:
    def __init__(self):
        self.rally_count = 0
    
    def generate_frame_commentary(self, ball_hit_frames, shot_types, rally_count, player_stats, frame_index):
        commentary_lines = []
        
        # Sample commentary based on shot types
        shot_type_player_1, shot_type_player_2 = shot_types
        if shot_type_player_1:
            commentary_lines.append(f"Player 1 executes a {shot_type_player_1}.")
        if shot_type_player_2:
            commentary_lines.append(f"Player 2 responds with a {shot_type_player_2}.")
        
        # Commentary based on rally count
        if rally_count > 0:
            commentary_lines.append(f"Rally count: {rally_count}.")
        
        # Add player statistics commentary
        if player_stats is not None and not player_stats.empty:
            # Print column names for debugging
            print("Player stats columns:", player_stats.columns)
            
            for player_id, stats in player_stats.iterrows():
                # Safeguard against missing columns
                shots = stats.get('shots', 'N/A')
                hits = stats.get('hits', 'N/A')
                commentary_lines.append(f"Player {player_id} Stats - Shots: {shots}, Hits: {hits}.")
        
        # Randomly add some general commentary to make it more engaging
        if random.random() < 0.1:  # 10% chance to add a general comment
            commentary_lines.append("What a thrilling game!")

        full_commentary = " ".join(commentary_lines)
        
        return full_commentary

    def save_commentary(self, filepath, commentary):
        with open(filepath, "w") as f:
            f.write(commentary)
            