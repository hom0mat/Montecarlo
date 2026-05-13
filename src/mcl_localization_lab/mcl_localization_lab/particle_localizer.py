from collections import deque
from pathlib import Path
import math

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


MAP_FILE = "warehouse_layout.png"
MAP_RESOLUTION = 0.05


def yaw_from_quat(qx, qy, qz, qw):
    """Return planar yaw from a ROS quaternion."""
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ParticleLocalizationNode(Node):
    """Monte Carlo Localization node using odometry, LiDAR, and a known grid map."""

    def __init__(self):
        super().__init__("particle_localization_node")

        self.grid_resolution = MAP_RESOLUTION
        self.known_map = self._load_map_image()
        self.map_rows, self.map_cols = self.known_map.shape
        self.score_grid = self._build_score_grid(self.known_map)

        self.particle_count = 320
        self.particle_cloud = np.zeros((self.particle_count, 4), dtype=float)
        self.ray_budget = 72
        self.low_likelihood_log = -10.0

        self.odom_xy_sigma = 0.48
        self.odom_yaw_sigma = 0.65
        self.scan_yaw_correction = 0.0
        self.random_injection_rate = 0.10

        self.pose_estimate = None
        self.smoothed_pose = None
        self.smoothing_gain = 0.80
        self.path_history = deque(maxlen=90)

        self.last_odom_pose = None
        self.robot_odom_pose = None
        self.seeded_from_odom = False

        self.seed_particles_over_map()

        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)

        self.get_logger().info("Particle localizer ready. Waiting for /odom and /scan.")

    def _load_map_image(self):
        package_root = Path(__file__).resolve().parents[1]
        map_path = package_root / "maps" / MAP_FILE
        image = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(
                f"Map image not found: {map_path}. Run: ros2 run mcl_localization_lab build_warehouse_map"
            )
        return image

    def _build_score_grid(self, image):
        occupied = (image < 128).astype(np.float64)
        score_grid = cv2.GaussianBlur(occupied, (0, 0), sigmaX=5.0, sigmaY=5.0)
        max_score = score_grid.max()
        if max_score > 0:
            score_grid /= max_score
        self.get_logger().info(
            f"Sensor score grid ready. Range: {score_grid.min():.4f} to {score_grid.max():.4f}"
        )
        return score_grid

    def world_to_map(self, x_m, y_m):
        col = int((x_m / self.grid_resolution) + self.map_cols / 2.0)
        row = int(self.map_rows / 2.0 - (y_m / self.grid_resolution))
        return col, row

    def map_to_world(self, col, row):
        x_m = (col - self.map_cols / 2.0) * self.grid_resolution
        y_m = (self.map_rows / 2.0 - row) * self.grid_resolution
        return x_m, y_m

    def is_free_cell_world(self, x_m, y_m):
        col, row = self.world_to_map(x_m, y_m)
        if col < 0 or col >= self.map_cols or row < 0 or row >= self.map_rows:
            return False
        return self.known_map[row, col] > 200

    def seed_particles_over_map(self):
        for idx in range(self.particle_count):
            while True:
                col = np.random.randint(0, self.map_cols)
                row = np.random.randint(0, self.map_rows)
                if self.known_map[row, col] > 200:
                    x_m, y_m = self.map_to_world(col, row)
                    yaw = np.random.uniform(-np.pi, np.pi)
                    self.particle_cloud[idx] = [x_m, y_m, yaw, 1.0 / self.particle_count]
                    break
        self.get_logger().info(f"Initial global sampling completed with {self.particle_count} candidates.")

    def seed_particles_near_pose(self, x_m, y_m, yaw):
        for idx in range(self.particle_count):
            sample_x = x_m + np.random.normal(0, 0.35)
            sample_y = y_m + np.random.normal(0, 0.35)
            sample_yaw = yaw + np.random.normal(0, 0.25)

            tries = 0
            while not self.is_free_cell_world(sample_x, sample_y) and tries < 30:
                sample_x = x_m + np.random.normal(0, 0.35)
                sample_y = y_m + np.random.normal(0, 0.35)
                sample_yaw = yaw + np.random.normal(0, 0.25)
                tries += 1

            self.particle_cloud[idx] = [sample_x, sample_y, wrap_angle(sample_yaw), 1.0 / self.particle_count]

        self.pose_estimate = (x_m, y_m, yaw)
        self.smoothed_pose = (x_m, y_m, yaw)
        self.get_logger().info(f"Particles seeded around odom: x={x_m:.2f}, y={y_m:.2f}, yaw={yaw:.2f}")

    def on_odom(self, msg):
        x_m = msg.pose.pose.position.x
        y_m = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        current_pose = np.array([x_m, y_m, yaw])
        self.robot_odom_pose = current_pose

        if self.last_odom_pose is None:
            self.last_odom_pose = current_pose
            if not self.seeded_from_odom:
                self.seed_particles_near_pose(x_m, y_m, yaw)
                self.seeded_from_odom = True
            return

        dx = current_pose[0] - self.last_odom_pose[0]
        dy = current_pose[1] - self.last_odom_pose[1]
        dyaw = wrap_angle(current_pose[2] - self.last_odom_pose[2])

        if abs(dx) > 0.002 or abs(dy) > 0.002 or abs(dyaw) > 0.005:
            self.apply_dead_reckoning(dx, dy, dyaw)
            self.last_odom_pose = current_pose

    def apply_dead_reckoning(self, dx, dy, dyaw):
        displacement = math.hypot(dx, dy)
        trans_noise = 0.005 + 0.04 * displacement
        yaw_noise = 0.005 + 0.04 * abs(dyaw)

        for idx in range(self.particle_count):
            old_x, old_y, old_yaw, _ = self.particle_cloud[idx]
            candidate_x = old_x + dx + np.random.normal(0, trans_noise)
            candidate_y = old_y + dy + np.random.normal(0, trans_noise)
            candidate_yaw = wrap_angle(old_yaw + dyaw + np.random.normal(0, yaw_noise))

            if self.is_free_cell_world(old_x, old_y) and not self.is_free_cell_world(candidate_x, candidate_y):
                self.particle_cloud[idx, 0:3] = [old_x, old_y, old_yaw]
            else:
                self.particle_cloud[idx, 0:3] = [candidate_x, candidate_y, candidate_yaw]

    def on_scan(self, msg):
        if self.last_odom_pose is None:
            return

        scan_angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        scan_ranges = np.asarray(msg.ranges)
        sample_count = min(len(scan_angles), len(scan_ranges))
        scan_angles = scan_angles[:sample_count]
        scan_ranges = scan_ranges[:sample_count]

        valid = (
            (scan_ranges > msg.range_min)
            & (scan_ranges < msg.range_max - 0.1)
            & ~np.isinf(scan_ranges)
            & ~np.isnan(scan_ranges)
        )
        scan_angles = scan_angles[valid]
        scan_ranges = scan_ranges[valid]

        if len(scan_ranges) == 0:
            self.update_estimate()
            self.draw_debug_view()
            return

        stride = max(1, len(scan_ranges) // self.ray_budget)
        scan_angles = scan_angles[::stride]
        scan_ranges = scan_ranges[::stride]

        log_weights = np.zeros(self.particle_count)
        half_cols = self.map_cols / 2.0
        half_rows = self.map_rows / 2.0

        for idx in range(self.particle_count):
            x_m, y_m, yaw, _ = self.particle_cloud[idx]

            if not self.is_free_cell_world(x_m, y_m):
                log_weights[idx] = -1e6
                continue

            world_angles = yaw + scan_angles + self.scan_yaw_correction
            hit_x = x_m + scan_ranges * np.cos(world_angles)
            hit_y = y_m + scan_ranges * np.sin(world_angles)

            hit_cols = (hit_x / self.grid_resolution + half_cols).astype(int)
            hit_rows = (half_rows - hit_y / self.grid_resolution).astype(int)
            inside = (hit_cols >= 0) & (hit_cols < self.map_cols) & (hit_rows >= 0) & (hit_rows < self.map_rows)

            scan_log_score = 0.0
            for ray_idx in range(len(scan_ranges)):
                field_score = self.score_grid[hit_rows[ray_idx], hit_cols[ray_idx]] if inside[ray_idx] else 0.0
                scan_log_score += math.log(field_score) if field_score > 1e-6 else self.low_likelihood_log

            if self.robot_odom_pose is not None:
                odom_x, odom_y, odom_yaw = self.robot_odom_pose
                xy_error = math.hypot(x_m - odom_x, y_m - odom_y)
                yaw_error = wrap_angle(yaw - odom_yaw)
                odom_prior = -0.5 * (xy_error / self.odom_xy_sigma) ** 2 - 0.5 * (yaw_error / self.odom_yaw_sigma) ** 2
                log_weights[idx] = scan_log_score + odom_prior
            else:
                log_weights[idx] = scan_log_score

        max_log = np.max(log_weights)
        weights = np.exp(log_weights - max_log)
        total_weight = np.sum(weights)
        self.particle_cloud[:, 3] = weights / total_weight if total_weight > 0 else 1.0 / self.particle_count

        self.update_estimate()
        self.keep_best_candidates()
        self.draw_debug_view()

    def update_estimate(self):
        if self.robot_odom_pose is not None:
            odom_x, odom_y, _ = self.robot_odom_pose
            radius = np.hypot(self.particle_cloud[:, 0] - odom_x, self.particle_cloud[:, 1] - odom_y)
            candidate_idx = np.where(radius < 1.5)[0]
            candidates = self.particle_cloud[candidate_idx] if len(candidate_idx) > 0 else self.particle_cloud
        else:
            candidates = self.particle_cloud

        weights = candidates[:, 3].copy()
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(candidates)) / len(candidates)

        x_hat = np.sum(candidates[:, 0] * weights)
        y_hat = np.sum(candidates[:, 1] * weights)
        yaw_hat = math.atan2(np.sum(np.sin(candidates[:, 2]) * weights), np.sum(np.cos(candidates[:, 2]) * weights))

        if self.smoothed_pose is None:
            self.smoothed_pose = (x_hat, y_hat, yaw_hat)
        else:
            prev_x, prev_y, prev_yaw = self.smoothed_pose
            smooth_x = self.smoothing_gain * prev_x + (1.0 - self.smoothing_gain) * x_hat
            smooth_y = self.smoothing_gain * prev_y + (1.0 - self.smoothing_gain) * y_hat
            smooth_yaw = wrap_angle(prev_yaw + (1.0 - self.smoothing_gain) * wrap_angle(yaw_hat - prev_yaw))
            self.smoothed_pose = (smooth_x, smooth_y, smooth_yaw)

        self.pose_estimate = self.smoothed_pose
        self.path_history.append((self.pose_estimate[0], self.pose_estimate[1]))

    def keep_best_candidates(self):
        weights = self.particle_cloud[:, 3].copy()
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(self.particle_count) / self.particle_count

        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        start = np.random.uniform(0, 1.0 / self.particle_count)
        probes = start + np.arange(self.particle_count) / self.particle_count
        selected_idx = np.searchsorted(cumulative, probes)
        selected_idx = np.clip(selected_idx, 0, self.particle_count - 1)

        next_cloud = self.particle_cloud[selected_idx].copy()
        next_cloud[:, 0] += np.random.normal(0, 0.02, self.particle_count)
        next_cloud[:, 1] += np.random.normal(0, 0.02, self.particle_count)
        next_cloud[:, 2] = np.arctan2(
            np.sin(next_cloud[:, 2] + np.random.normal(0, 0.03, self.particle_count)),
            np.cos(next_cloud[:, 2] + np.random.normal(0, 0.03, self.particle_count)),
        )

        injected = max(1, int(self.particle_count * self.random_injection_rate))
        if self.robot_odom_pose is not None:
            odom_x, odom_y, odom_yaw = self.robot_odom_pose
            for k in range(injected):
                idx = self.particle_count - 1 - k
                sx = odom_x + np.random.normal(0, 0.35)
                sy = odom_y + np.random.normal(0, 0.35)
                syaw = odom_yaw + np.random.normal(0, 0.25)
                tries = 0
                while not self.is_free_cell_world(sx, sy) and tries < 15:
                    sx = odom_x + np.random.normal(0, 0.35)
                    sy = odom_y + np.random.normal(0, 0.35)
                    tries += 1
                next_cloud[idx] = [sx, sy, wrap_angle(syaw), 1.0 / self.particle_count]

        for idx in range(self.particle_count):
            if not self.is_free_cell_world(next_cloud[idx, 0], next_cloud[idx, 1]):
                parent = selected_idx[idx]
                next_cloud[idx, 0:3] = self.particle_cloud[parent, 0:3]

        next_cloud[:, 3] = 1.0 / self.particle_count
        self.particle_cloud = next_cloud

    def draw_debug_view(self):
        frame = cv2.cvtColor(self.known_map, cv2.COLOR_GRAY2BGR)

        for j, (x_m, y_m) in enumerate(self.path_history):
            col, row = self.world_to_map(x_m, y_m)
            if 0 <= col < self.map_cols and 0 <= row < self.map_rows:
                intensity = int(80 + 175 * (j / max(len(self.path_history), 1)))
                cv2.circle(frame, (col, row), 2, (0, intensity, intensity), -1)

        for idx in range(self.particle_count):
            col, row = self.world_to_map(self.particle_cloud[idx, 0], self.particle_cloud[idx, 1])
            if 0 <= col < self.map_cols and 0 <= row < self.map_rows:
                cv2.circle(frame, (col, row), 2, (0, 255, 0), -1)

        if self.pose_estimate is not None:
            x_m, y_m, yaw = self.pose_estimate
            col, row = self.world_to_map(x_m, y_m)
            if 0 <= col < self.map_cols and 0 <= row < self.map_rows:
                cv2.circle(frame, (col, row), 8, (0, 0, 255), 2)
                cv2.line(frame, (col, row), (int(col + 22 * math.cos(yaw)), int(row - 22 * math.sin(yaw))), (0, 0, 255), 2)

        if self.robot_odom_pose is not None:
            x_m, y_m, yaw = self.robot_odom_pose
            col, row = self.world_to_map(x_m, y_m)
            if 0 <= col < self.map_cols and 0 <= row < self.map_rows:
                cv2.circle(frame, (col, row), 7, (255, 0, 0), -1)
                cv2.line(frame, (col, row), (int(col + 22 * math.cos(yaw)), int(row - 22 * math.sin(yaw))), (255, 0, 0), 2)

        cv2.imshow("MCL warehouse lab", cv2.resize(frame, (700, 700)))
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping particle localization node.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
