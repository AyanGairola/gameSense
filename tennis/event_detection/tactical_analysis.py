class TacticalAnalysis:
    def __init__(self, court_keypoints):
        self.zones = self.define_zones(court_keypoints)

    def define_zones(self, court_keypoints):
        """
        Define zones on the court using the court keypoints.
        """
        # Ensure court_keypoints is a list of alternating x, y coordinates
        assert len(court_keypoints) >= 20, "Not enough keypoints to define the court zones"  # Ensure at least 10 (x, y) pairs

        # Define zones based on the keypoints from the image:
        zones = {
            # Baseline (near the top of the court)
            "baseline": [(court_keypoints[0], court_keypoints[1]), (court_keypoints[8], court_keypoints[9])],
            
            # Service line (middle area of the court)
            "service_line": [(court_keypoints[10], court_keypoints[11]), (court_keypoints[14], court_keypoints[15])],
            
            # Net (straight vertical line between keypoints 8 and 10)
            "net": self.calculate_net_line(
                (court_keypoints[16], court_keypoints[17]),  # Keypoint for the left side of the net
                (court_keypoints[18], court_keypoints[19])   # Keypoint for the right side of the net
            )
        }

        print("Defined zones:", zones)  # Debugging print statement to verify zone coordinates
        return zones

    def calculate_net_line(self, net_left, net_right):
        """
        Calculate the x-coordinate of the net line based on two keypoints (net_left and net_right).
        """
        if isinstance(net_left, tuple) and isinstance(net_right, tuple):  # Ensure both keypoints have x and y coordinates
            net_x = (net_left[0] + net_right[0]) / 2  # Middle x-coordinate of the net
            return [(net_x, 0), (net_x, 1080)]  # Assume the net is a vertical line spanning the court's height
        else:
            raise ValueError("Invalid net keypoints")

    def analyze_ball_position(self, ball_position):
        """
        Analyze the ball's position and determine which zone it's in.
        """
        # Print ball position for debugging
        print(f"Ball Position: {ball_position}")

        # Check if the ball is in any defined zone
        for zone_name, zone_coords in self.zones.items():
            if self.is_in_zone(ball_position, zone_coords):
                print(f"Ball is in {zone_name}")
                return zone_name

        # If no zone matched, consider the ball "out"
        print("Ball is out of bounds")
        return "out"

    def analyze_ball_bounce(self, bounce_position):
        """
        Analyze the ball's bounce position and determine which zone it bounced in.
        """
        return self.analyze_ball_position(bounce_position)

    def analyze_player_positions(self, player_positions):
        """
        Analyze player positions relative to zones.
        """
        player_zone_stats = {}
        for player_id, player_position in enumerate(player_positions, start=1):
            if self.is_in_zone(player_position, self.zones["baseline"]):
                player_zone_stats[f"player_{player_id}"] = "baseline"
            elif self.is_in_zone(player_position, self.zones["service_line"]):
                player_zone_stats[f"player_{player_id}"] = "service_line"
            elif self.is_in_zone(player_position, self.zones["net"]):  # Check if player is near the net
                player_zone_stats[f"player_{player_id}"] = "net"
            else:
                player_zone_stats[f"player_{player_id}"] = "out"
        return player_zone_stats

    def is_in_zone(self, position, zone_coords):
        """
        Check if the ball or player position is inside the zone defined by two coordinates.
        """
        (x_min, y_min), (x_max, y_max) = zone_coords
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max
