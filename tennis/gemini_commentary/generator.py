import random
import time
import google.generativeai as genai
from collections import deque

class CommentaryGenerator:
    def __init__(self, api_key):
        self.rally_count = 0
        self.use_fallback = False
        
        if api_key is None or api_key == "":
            print("Warning: No API key provided. Using fallback commentary.")
            self.use_fallback = True
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Using deque to store the timestamps of API requests for rate limiting
        self.request_times = deque()
        self.max_requests_per_minute = 12
        self.request_interval = 60 / self.max_requests_per_minute
        

    def generate_frame_commentary(self, ball_hit_frames, shot_types, rally_count, frame_index):
        if self.use_fallback:
            return self.fallback_commentary(ball_hit_frames, shot_types, rally_count, frame_index)
        
        self._wait_for_rate_limit()
        context = f"""
        Frame: {frame_index}
        Rally count: {rally_count}
        Player 1 shot: {shot_types[0]}
        Player 2 shot: {shot_types[1]}
        """
        prompt = f"""
        Generate a brief, engaging tennis commentary based on the following context:
        {context}
        The commentary should be dynamic, reflect the current state of the game, and sound natural.
        Focus on the most interesting aspects of the current play.
        """
        try:
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.7
            ))
            commentary = response.text.strip()
            self._record_request_time()
        except genai.types.BlockedPromptException as e:
            print(f"Daily limit exceeded or content blocked: {e}")
            commentary = self.fallback_commentary(ball_hit_frames, shot_types, rally_count, frame_index)
        except Exception as e:
            print(f"Error generating commentary: {e}")
            commentary = self.fallback_commentary(ball_hit_frames, shot_types, rally_count, frame_index)
        return commentary
        return commentary

    def _wait_for_rate_limit(self):
        current_time = time.time()
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest_request_time = self.request_times[0]
            time_since_oldest_request = current_time - oldest_request_time
            if time_since_oldest_request < 60:
                sleep_time = 60 - time_since_oldest_request
                time.sleep(sleep_time)

    def _record_request_time(self):
        current_time = time.time()
        self.request_times.append(current_time)
        if len(self.request_times) > self.max_requests_per_minute:
            self.request_times.popleft()

    def fallback_commentary(self, ball_hit_frames, shot_types, rally_count,  frame_index):
        
        commentary_lines = []
        shot_type_player_1, shot_type_player_2 = shot_types
        if shot_type_player_1:
            commentary_lines.append(f"Player 1 executes a {shot_type_player_1}.")
        if shot_type_player_2:
            commentary_lines.append(f"Player 2 responds with a {shot_type_player_2}.")
        if rally_count > 0:
            commentary_lines.append(f"Rally count: {rally_count}.")

        if random.random() < 0.1:
            commentary_lines.append("What a thrilling game!")
        return " ".join(commentary_lines)

    