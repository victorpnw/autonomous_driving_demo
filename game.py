import math
import random
import bisect
from collections import deque
from dataclasses import dataclass

import pygame
import torch
import torch.nn as nn
import torch.optim as optim


WIDTH, HEIGHT = 1100, 720
FPS = 60
BG_COLOR = (17, 23, 31)
TEXT_COLOR = (238, 241, 245)
ACCENT = (53, 168, 255)
TRACK_COLOR = (85, 94, 105)
TRACK_EDGE = (165, 175, 186)
CAR_COLOR = (255, 179, 71)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


@dataclass
class Button:
    rect: pygame.Rect
    label: str

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, hover: bool) -> None:
        base = (37, 121, 189) if hover else ACCENT
        pygame.draw.rect(surface, base, self.rect, border_radius=12)
        pygame.draw.rect(surface, (210, 231, 255), self.rect, 2, border_radius=12)
        text = font.render(self.label, True, (12, 22, 34))
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)


class Slider:
    def __init__(self, x: int, y: int, width: int, label: str, minimum: float, maximum: float, value: float):
        self.rect = pygame.Rect(x, y, width, 8)
        self.knob_radius = 14
        self.label = label
        self.min = minimum
        self.max = maximum
        self.value = value
        self.dragging = False

    def _knob_x(self) -> int:
        ratio = (self.value - self.min) / (self.max - self.min)
        return int(self.rect.left + ratio * self.rect.width)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            knob_pos = pygame.Vector2(self._knob_x(), self.rect.centery)
            if knob_pos.distance_to(pygame.Vector2(event.pos)) <= self.knob_radius + 3:
                self.dragging = True
            elif self.rect.inflate(0, 20).collidepoint(event.pos):
                self._set_from_mouse(event.pos[0])
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_mouse(event.pos[0])

    def _set_from_mouse(self, x_pos: int) -> None:
        ratio = clamp((x_pos - self.rect.left) / self.rect.width, 0.0, 1.0)
        self.value = self.min + ratio * (self.max - self.min)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, tiny: pygame.font.Font) -> None:
        pygame.draw.rect(surface, (58, 72, 87), self.rect.inflate(0, 12), border_radius=6)
        fill_rect = pygame.Rect(self.rect.left, self.rect.top, self._knob_x() - self.rect.left, self.rect.height)
        pygame.draw.rect(surface, ACCENT, fill_rect.inflate(0, 12), border_radius=6)
        pygame.draw.circle(surface, (244, 249, 255), (self._knob_x(), self.rect.centery), self.knob_radius)

        label_text = font.render(self.label, True, TEXT_COLOR)
        value_text = tiny.render(f"{self.value:.5f}", True, (202, 212, 222))
        surface.blit(label_text, (self.rect.left, self.rect.top - 44))
        surface.blit(value_text, (self.rect.right - value_text.get_width(), self.rect.top - 40))


class IrregularTrackEnv:
    def __init__(self):
        self.center = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
        self.track_width = 106.0
        self.off_track_margin = 24.0
        self.max_speed = 4.2
        self.max_steps = 1400

        self.actions = [(steer, accel) for steer in (-1, 0, 1) for accel in (-1, 0, 1)]

        self.centerline = self._generate_centerline()
        self.segment_unit = []
        self.segment_lengths = []
        self.cumulative_lengths = []
        self.total_length = 0.0
        self.world_min_x = 0.0
        self.world_max_x = 0.0
        self.world_min_y = 0.0
        self.world_max_y = 0.0
        self._build_track_cache()

        self.position = pygame.Vector2(0, 0)
        self.heading = 0.0
        self.speed = 0.0
        self.prev_progress = 0.0
        self.accumulated_progress = 0.0
        self.steps = 0
        self.closest_segment_idx = None

    def shift_track(self, dx: float, dy: float):
        delta = pygame.Vector2(dx, dy)
        self.center += delta
        self.centerline = [pt + delta for pt in self.centerline]
        self.position += delta
        self.world_min_x += dx
        self.world_max_x += dx
        self.world_min_y += dy
        self.world_max_y += dy

    def _generate_centerline(self, num_points: int = 560):
        points = []
        for i in range(num_points):
            theta = (2.0 * math.pi * i) / num_points
            radius = 214.0 + 44.0 * math.sin(2.0 * theta + 0.6) + 34.0 * math.sin(3.0 * theta - 1.1) + 20.0 * math.sin(5.0 * theta + 1.9)
            x = self.center.x + math.cos(theta) * (radius * 1.11 + 26.0 * math.sin(4.0 * theta + 0.5))
            y = self.center.y + math.sin(theta) * (radius * 0.82 + 22.0 * math.cos(3.0 * theta - 0.9))
            points.append(pygame.Vector2(x, y))

        for _ in range(2):
            smoothed = []
            n = len(points)
            for i in range(n):
                smoothed.append(points[(i - 1) % n] * 0.2 + points[i] * 0.6 + points[(i + 1) % n] * 0.2)
            points = smoothed

        return points

    def _build_track_cache(self):
        n = len(self.centerline)
        self.segment_unit = []
        self.segment_lengths = []
        self.cumulative_lengths = [0.0]

        for i in range(n):
            a = self.centerline[i]
            b = self.centerline[(i + 1) % n]
            seg = b - a
            seg_len = max(seg.length(), 1e-6)
            self.segment_unit.append(seg / seg_len)
            self.segment_lengths.append(seg_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + seg_len)

        self.total_length = self.cumulative_lengths[-1]

        pad = self.track_width / 2 + self.off_track_margin + 18.0
        xs = [p.x for p in self.centerline]
        ys = [p.y for p in self.centerline]
        self.world_min_x = min(xs) - pad
        self.world_max_x = max(xs) + pad
        self.world_min_y = min(ys) - pad
        self.world_max_y = max(ys) + pad

    def reset(self):
        self.position = self.centerline[0].copy()
        start_tangent = self.segment_unit[0]
        self.heading = math.atan2(start_tangent.y, start_tangent.x)
        self.speed = 0.45
        self.prev_progress = 0.0
        self.accumulated_progress = 0.0
        self.steps = 0
        self.closest_segment_idx = 0
        return self.observe()

    def _track_projection(self, point: pygame.Vector2):
        n = len(self.centerline)
        if self.closest_segment_idx is None:
            candidates = range(n)
        else:
            window = 120
            candidates = [((self.closest_segment_idx + delta) % n) for delta in range(-window, window + 1)]

        best_dist2 = float("inf")
        best_i = 0
        best_t = 0.0
        best_proj = self.centerline[0]

        for i in candidates:
            a = self.centerline[i]
            b = self.centerline[(i + 1) % n]
            ab = b - a
            ab_len2 = ab.length_squared()
            if ab_len2 < 1e-8:
                continue
            t = clamp((point - a).dot(ab) / ab_len2, 0.0, 1.0)
            proj = a + ab * t
            dist2 = (point - proj).length_squared()
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_t = t
                best_proj = proj

        tangent = self.segment_unit[best_i]
        normal = pygame.Vector2(-tangent.y, tangent.x)
        lateral_offset = (point - best_proj).dot(normal)
        tangent_angle = math.atan2(tangent.y, tangent.x)
        progress = self.cumulative_lengths[best_i] + best_t * self.segment_lengths[best_i]
        if progress >= self.total_length:
            progress -= self.total_length

        self.closest_segment_idx = best_i
        return lateral_offset, tangent_angle, progress

    def _track_metrics(self):
        lateral_offset, tangent_angle, progress = self._track_projection(self.position)
        heading_error = wrap_angle(self.heading - tangent_angle)
        return lateral_offset, heading_error, progress

    def _point_tangent_at_progress(self, progress: float):
        if self.total_length <= 0.0:
            return self.centerline[0], self.segment_unit[0]

        p = progress % self.total_length
        idx = bisect.bisect_right(self.cumulative_lengths, p) - 1
        idx = max(0, min(idx, len(self.segment_lengths) - 1))
        seg_start = self.centerline[idx]
        seg_tangent = self.segment_unit[idx]
        seg_len = self.segment_lengths[idx]
        local = p - self.cumulative_lengths[idx]
        t = clamp(local / max(seg_len, 1e-6), 0.0, 1.0)
        seg_end = self.centerline[(idx + 1) % len(self.centerline)]
        point = seg_start + (seg_end - seg_start) * t
        return point, seg_tangent

    def preview_centerline_local(self, heading: float, max_forward: float = 320.0, sample_count: int = 72):
        _, _, current_progress = self._track_metrics()
        forward_vec = pygame.Vector2(math.cos(heading), math.sin(heading))
        right_vec = pygame.Vector2(-math.sin(heading), math.cos(heading))

        samples = []
        for i in range(sample_count):
            p = i / max(sample_count - 1, 1)
            dist = 3.0 + (p ** 1.55) * max_forward
            world_point, _ = self._point_tangent_at_progress(current_progress + dist)
            rel = world_point - self.position
            fwd = rel.dot(forward_vec)
            side = rel.dot(right_vec)
            if fwd > 1.0:
                samples.append((fwd, side))

        if not samples:
            samples.append((1.0, 0.0))
        return samples

    def observe(self):
        offset, heading_error, _ = self._track_metrics()
        offset_limit = self.track_width / 2 + self.off_track_margin

        heading_norm = clamp(heading_error / math.pi, -1.0, 1.0)
        offset_norm = clamp(offset / offset_limit, -1.0, 1.0)
        speed_norm = clamp(self.speed / self.max_speed, 0.0, 1.0)

        return [heading_norm, offset_norm, speed_norm]

    def step(self, action_index: int):
        steer, accel = self.actions[action_index]

        self.heading += steer * (0.045 + 0.03 * (self.speed / max(self.max_speed, 1e-6)))
        self.speed += accel * 0.085
        self.speed -= 0.05 * self.speed
        self.speed *= 1.0 - 0.02 * abs(steer)
        self.speed = clamp(self.speed, 0.0, self.max_speed)

        velocity = pygame.Vector2(math.cos(self.heading), math.sin(self.heading)) * self.speed
        self.position += velocity
        self.steps += 1

        offset, heading_error, track_progress = self._track_metrics()
        delta_progress = track_progress - self.prev_progress
        if delta_progress < -self.total_length * 0.5:
            delta_progress += self.total_length
        elif delta_progress > self.total_length * 0.5:
            delta_progress -= self.total_length
        self.prev_progress = track_progress

        forward_progress = max(0.0, delta_progress)
        backward_progress = max(0.0, -delta_progress)
        self.accumulated_progress += forward_progress

        reward = forward_progress * 0.95
        reward -= abs(offset) / (self.track_width / 2) * 0.70
        reward -= abs(heading_error) / math.pi * 0.45
        reward -= backward_progress * 1.30
        reward -= 0.02

        if self.speed < 0.28:
            reward -= 0.09

        off_track = abs(offset) > (self.track_width / 2 + self.off_track_margin)
        done = False

        if off_track:
            reward -= 24.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        laps = int(self.accumulated_progress / self.total_length)
        lap_complete = laps >= 1

        if lap_complete:
            reward += 30.0
            done = True

        info = {
            "offset": offset,
            "heading_error": heading_error,
            "progress": forward_progress,
            "off_track": off_track,
            "laps": laps,
            "lap_complete": lap_complete,
        }

        return self.observe(), reward, done, info

    def draw_track(self, surface: pygame.Surface) -> None:
        pygame.draw.lines(surface, TRACK_COLOR, True, self.centerline, int(self.track_width))
        pygame.draw.lines(surface, TRACK_EDGE, True, self.centerline, 2)

        start = self.centerline[0]
        tangent = self.segment_unit[0]
        normal = pygame.Vector2(-tangent.y, tangent.x)
        start_outer = start + normal * (self.track_width / 2)
        start_inner = start - normal * (self.track_width / 2)
        pygame.draw.line(surface, (245, 245, 245), start_outer, start_inner, 4)


class DQNNet(nn.Module):
    def __init__(self, state_dim: int, action_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((list(state), int(action), float(reward), list(next_state), float(done)))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_count: int,
        lr: float,
        gamma: float,
        epsilon: float,
        target_sync_steps: int,
        batch_size: int = 64,
        replay_capacity: int = 50000,
    ):
        self.state_dim = state_dim
        self.action_count = action_count
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQNNet(state_dim, action_count).to(self.device)
        self.target_net = DQNNet(state_dim, action_count).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(replay_capacity)
        self.train_steps = 0
        self.last_loss = 0.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_count)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        states = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device)

        q_selected = self.online_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 5.0)
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.train_steps += 1

        if self.train_steps % self.target_sync_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay(self, decay_rate: float, min_epsilon: float = 0.05):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


class AutonomousDrivingDemo:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Autonomous Driving Demo (Deep Q-Learning)")
        self.min_window_w = 960
        self.min_window_h = 640
        self.window_w = WIDTH
        self.window_h = HEIGHT
        self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("consolas", 52, bold=True)
        self.large_font = pygame.font.SysFont("consolas", 32, bold=True)
        self.body_font = pygame.font.SysFont("consolas", 24)
        self.tiny_font = pygame.font.SysFont("consolas", 18)

        self.state = "home"
        self.running = True

        self.play_button = Button(pygame.Rect(0, 0, 240, 72), "Play")
        self.start_training_button = Button(pygame.Rect(0, 0, 300, 64), "Start Learning")
        self.back_button = Button(pygame.Rect(30, 28, 130, 48), "Back")
        self.exit_sim_button = Button(pygame.Rect(0, 0, 132, 44), "Exit Sim")

        self.learning_slider = Slider(WIDTH // 2 - 260, 310, 520, "Learning Rate", 0.0003, 0.003, 0.001)

        self.env = IrregularTrackEnv()
        self.agent = None

        self.current_state = None
        self.steps_per_frame = 1
        self.epsilon_decay = 0.998
        self.prev_steer = 0

        self.episode = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.recent_rewards = deque(maxlen=25)
        self.recent_off_track = deque(maxlen=25)
        self.total_laps = 0
        self.last_info = {"offset": 0.0, "heading_error": 0.0, "progress": 0.0, "off_track": False, "laps": 0, "lap_complete": False}

        self.path_points = deque(maxlen=850)
        self.visual_tick = 0.0
        self.crash_timer = 0
        self.crash_side = 0
        self.finish_line_timer = 0
        self.pending_episode_reset = False

        self.result_timer = 0
        self.result_message = ""

        self.home_tick = 0.0
        self.home_stars = [(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.uniform(0.3, 1.0)) for _ in range(120)]

        self._layout_ui(self.window_w, self.window_h)

    def _layout_ui(self, w: int, h: int):
        self.play_button.rect = pygame.Rect(w // 2 - 120, h // 2 + 40, 240, 72)
        self.start_training_button.rect = pygame.Rect(w // 2 - 150, h - 130, 300, 64)
        self.back_button.rect = pygame.Rect(30, 28, 130, 48)
        self.exit_sim_button.rect = pygame.Rect(w - 302, h - 314, 132, 44)

        self.learning_slider.rect = pygame.Rect(w // 2 - 260, 310, 520, 8)

    def _handle_resize(self, width: int, height: int):
        new_w = max(self.min_window_w, int(width))
        new_h = max(self.min_window_h, int(height))
        if new_w == self.window_w and new_h == self.window_h:
            return

        old_center = self.env.center.copy()
        self.window_w = new_w
        self.window_h = new_h
        self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
        self._layout_ui(self.window_w, self.window_h)

        target_center = pygame.Vector2(self.window_w * 0.5, self.window_h * 0.5)
        shift = target_center - old_center
        self.env.shift_track(shift.x, shift.y)

    def reset_training_metrics(self):
        self.episode = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.recent_rewards.clear()
        self.recent_off_track.clear()
        self.total_laps = 0
        self.path_points.clear()
        self.crash_timer = 0
        self.crash_side = 0
        self.finish_line_timer = 0
        self.pending_episode_reset = False

    def configure_training(self):
        lr = self.learning_slider.value
        gamma = 0.97
        epsilon_start = 1.0

        self.steps_per_frame = 3
        self.epsilon_decay = 0.99846
        target_sync_steps = 196

        self.agent = DQNAgent(
            state_dim=3,
            action_count=len(self.env.actions),
            lr=lr,
            gamma=gamma,
            epsilon=epsilon_start,
            target_sync_steps=target_sync_steps,
        )

        self.reset_training_metrics()
        self.current_state = self.env.reset()
        self.prev_steer = 0
        self.state = "training"

    def finish_episode(self, off_track: bool):
        self.episode += 1
        self.recent_rewards.append(self.episode_reward)
        self.recent_off_track.append(1 if off_track else 0)

        self.episode_reward = 0.0
        self.episode_steps = 0

        if self.last_info["laps"] > 0:
            self.total_laps += self.last_info["laps"]

    def _training_success(self) -> bool:
        if len(self.recent_rewards) < 20:
            return False

        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        off_track_rate = sum(self.recent_off_track) / len(self.recent_off_track)

        return self.episode >= 45 and avg_reward > 120.0 and off_track_rate < 0.30 and self.agent.epsilon < 0.22

    def _start_new_episode(self):
        self.current_state = self.env.reset()
        self.path_points.clear()
        self.prev_steer = 0
        self.visual_tick = 0.0
        self.crash_timer = 0
        self.crash_side = 0
        self.pending_episode_reset = False

    def _exit_simulation_to_home(self):
        self.state = "home"
        self.crash_timer = 0
        self.crash_side = 0
        self.finish_line_timer = 0
        self.pending_episode_reset = False
        self.result_timer = 0

    def update_training(self):
        if self.finish_line_timer > 0:
            self.finish_line_timer -= 1

        if self.crash_timer > 0:
            self.crash_timer -= 1
            if self.crash_timer <= 0 and self.pending_episode_reset:
                self.pending_episode_reset = False
                self._start_new_episode()
            return

        for _ in range(self.steps_per_frame):
            action = self.agent.select_action(self.current_state)
            steer, _ = self.env.actions[action]

            next_state, reward, done, info = self.env.step(action)

            steer_change = abs(steer - self.prev_steer)
            reward -= 0.078 * steer_change

            self.agent.update(self.current_state, action, reward, next_state, done)
            self.current_state = next_state
            self.prev_steer = steer
            self.agent.decay(self.epsilon_decay)

            self.episode_reward += reward
            self.episode_steps += 1
            self.last_info = info
            self.path_points.append(self.env.position.copy())
            self.visual_tick += 0.03 + self.env.speed * 0.055

            if done:
                self.finish_episode(info["off_track"])
                if self._training_success():
                    self.result_message = "Great driving! The DQN learned a stable policy."
                    self.result_timer = FPS * 4
                    self.state = "result"
                    return
                if info.get("lap_complete"):
                    self.finish_line_timer = int(FPS * 1.5)
                    self._start_new_episode()
                    continue
                if info["off_track"]:
                    self.crash_side = -1 if info["offset"] > 0 else 1
                    self.crash_timer = int(FPS * 0.8)
                    self.pending_episode_reset = True
                    return
                self._start_new_episode()

    def draw_car(self, surface: pygame.Surface, pos: pygame.Vector2, heading: float):
        front = pos + pygame.Vector2(math.cos(heading), math.sin(heading)) * 15
        rear_left = pos + pygame.Vector2(math.cos(heading + 2.55), math.sin(heading + 2.55)) * 10
        rear_right = pos + pygame.Vector2(math.cos(heading - 2.55), math.sin(heading - 2.55)) * 10
        pygame.draw.polygon(surface, CAR_COLOR, [front, rear_left, rear_right])
        pygame.draw.polygon(surface, (40, 30, 15), [front, rear_left, rear_right], 2)

    def draw_first_person_view(self, rect: pygame.Rect):
        heading_error = self.last_info["heading_error"]
        offset = self.last_info["offset"]

        sky_top = (55, 96, 156)
        sky_bottom = (138, 182, 234)
        ground_top = (50, 74, 50)
        ground_bottom = (24, 38, 24)

        horizon_shift = int(clamp(heading_error * 34.0, -44.0, 44.0))
        horizon_y = clamp(rect.top + int(rect.height * 0.42) + horizon_shift, rect.top + 120, rect.bottom - 170)

        for y in range(rect.top, int(horizon_y)):
            t = (y - rect.top) / max(1, int(horizon_y) - rect.top)
            color = (
                int(sky_top[0] + (sky_bottom[0] - sky_top[0]) * t),
                int(sky_top[1] + (sky_bottom[1] - sky_top[1]) * t),
                int(sky_top[2] + (sky_bottom[2] - sky_top[2]) * t),
            )
            pygame.draw.line(self.screen, color, (rect.left, y), (rect.right, y))

        for y in range(int(horizon_y), rect.bottom):
            t = (y - horizon_y) / max(1, rect.bottom - int(horizon_y))
            color = (
                int(ground_top[0] + (ground_bottom[0] - ground_top[0]) * t),
                int(ground_top[1] + (ground_bottom[1] - ground_top[1]) * t),
                int(ground_top[2] + (ground_bottom[2] - ground_top[2]) * t),
            )
            pygame.draw.line(self.screen, color, (rect.left, y), (rect.right, y))

        road_bottom_y = rect.bottom - 18
        road_far_half = 46
        road_near_half = int(rect.width * 0.44)

        offset_norm = clamp(offset / (self.env.track_width / 2), -1.0, 1.0)
        camera_heading = self.env.heading - heading_error * 0.60
        preview = self.env.preview_centerline_local(camera_heading, max_forward=340.0, sample_count=84)
        preview_f = [v[0] for v in preview]
        preview_s = [v[1] for v in preview]

        def side_at_forward(target_fwd: float) -> float:
            if target_fwd <= preview_f[0]:
                return preview_s[0]
            if target_fwd >= preview_f[-1]:
                return preview_s[-1]
            idx = bisect.bisect_right(preview_f, target_fwd)
            i0 = max(0, idx - 1)
            i1 = min(len(preview_f) - 1, idx)
            f0, f1 = preview_f[i0], preview_f[i1]
            s0, s1 = preview_s[i0], preview_s[i1]
            if abs(f1 - f0) < 1e-6:
                return s0
            t = (target_fwd - f0) / (f1 - f0)
            return s0 + (s1 - s0) * t

        def project_side(side: float, forward: float, depth: float) -> int:
            base = (side / (forward + 120.0)) * rect.width * (0.78 + 0.16 * depth)
            limit = rect.width * 0.55
            return int(clamp(base, -limit, limit))

        contact_side = 0
        if self.crash_timer > 0:
            contact_side = self.crash_side
        elif abs(offset_norm) > 0.84:
            contact_side = -1 if offset_norm > 0 else 1

        # Separate track curvature from car's lateral offset
        base_side = preview_s[0] if preview_s else 0.0
        track_half = self.env.track_width / 2

        segments = 28
        for i in range(segments):
            t0 = i / segments
            t1 = (i + 1) / segments

            p0 = t0 ** 1.55
            p1 = t1 ** 1.55

            y0 = int(horizon_y + p0 * (road_bottom_y - horizon_y))
            y1 = int(horizon_y + p1 * (road_bottom_y - horizon_y))

            w0 = int(road_far_half + p0 * (road_near_half - road_far_half))
            w1 = int(road_far_half + p1 * (road_near_half - road_far_half))

            near_forward = 20.0
            far_forward = 340.0
            fwd0 = near_forward + ((1.0 - p0) ** 1.62) * (far_forward - near_forward)
            fwd1 = near_forward + ((1.0 - p1) ** 1.62) * (far_forward - near_forward)
            curvature0 = side_at_forward(fwd0) - base_side
            curvature1 = side_at_forward(fwd1) - base_side
            curve0 = project_side(curvature0, fwd0, p0)
            curve1 = project_side(curvature1, fwd1, p1)

            offset_px0 = int((-offset / track_half) * w0)
            offset_px1 = int((-offset / track_half) * w1)
            cx0 = rect.centerx + curve0 + offset_px0
            cx1 = rect.centerx + curve1 + offset_px1

            shade = 52 + int((i % 2) * 8) + int((1.0 - t0) * 24)
            road_color = (shade, shade, shade + 2)
            pygame.draw.polygon(
                self.screen,
                road_color,
                [(cx0 - w0, y0), (cx0 + w0, y0), (cx1 + w1, y1), (cx1 - w1, y1)],
            )

            edge_color = (214, 218, 224)
            pygame.draw.line(self.screen, edge_color, (cx0 - w0, y0), (cx1 - w1, y1), 2)
            pygame.draw.line(self.screen, edge_color, (cx0 + w0, y0), (cx1 + w1, y1), 2)

            wall_gap0 = max(6, int(3 + p0 * 12))
            wall_gap1 = max(6, int(3 + p1 * 12))
            wall_th0 = max(4, int(6 + p0 * 18))
            wall_th1 = max(4, int(6 + p1 * 18))
            wall_h0 = max(2, int(2 + p0 * 18))
            wall_h1 = max(2, int(2 + p1 * 18))

            li0 = cx0 - w0 - wall_gap0
            lo0 = li0 - wall_th0
            li1 = cx1 - w1 - wall_gap1
            lo1 = li1 - wall_th1
            ri0 = cx0 + w0 + wall_gap0
            ro0 = ri0 + wall_th0
            ri1 = cx1 + w1 + wall_gap1
            ro1 = ri1 + wall_th1

            left_wall = (143, 149, 160)
            right_wall = (143, 149, 160)
            if contact_side == -1 and i > segments - 8:
                left_wall = (196, 118, 104)
            if contact_side == 1 and i > segments - 8:
                right_wall = (196, 118, 104)

            pygame.draw.polygon(self.screen, left_wall, [(li0, y0), (lo0, y0), (lo1, y1), (li1, y1)])
            pygame.draw.polygon(self.screen, right_wall, [(ri0, y0), (ro0, y0), (ro1, y1), (ri1, y1)])
            pygame.draw.polygon(self.screen, (206, 212, 222), [(li0, y0 - wall_h0), (lo0, y0 - wall_h0), (lo1, y1 - wall_h1), (li1, y1 - wall_h1)])
            pygame.draw.polygon(self.screen, (206, 212, 222), [(ri0, y0 - wall_h0), (ro0, y0 - wall_h0), (ro1, y1 - wall_h1), (ri1, y1 - wall_h1)])

            lane_anim = int(self.visual_tick * 11.0)
            if (i - lane_anim) % 6 in (0, 1, 2):
                lane_half0 = max(2, w0 // 13)
                lane_half1 = max(2, w1 // 13)
                pygame.draw.polygon(
                    self.screen,
                    (245, 244, 148),
                    [(cx0 - lane_half0, y0), (cx0 + lane_half0, y0), (cx1 + lane_half1, y1), (cx1 - lane_half1, y1)],
                )

            if i % 4 == 0:
                post_h = int(8 + p1 * 28)
                left_post_x = cx1 - w1 - 8
                right_post_x = cx1 + w1 + 8
                pygame.draw.line(self.screen, (223, 236, 255), (left_post_x, y1), (left_post_x, y1 - post_h), 2)
                pygame.draw.line(self.screen, (223, 236, 255), (right_post_x, y1), (right_post_x, y1 - post_h), 2)

        # -- start/finish line --
        cam_fwd_vec = pygame.Vector2(math.cos(camera_heading), math.sin(camera_heading))
        rel_start = self.env.centerline[0] - self.env.position
        finish_fwd = rel_start.dot(cam_fwd_vec)

        near_forward = 20.0
        far_forward = 340.0
        if near_forward < finish_fwd < far_forward:
            ratio = (finish_fwd - near_forward) / (far_forward - near_forward)
            fl_p = 1.0 - ratio ** (1.0 / 1.62)
            fl_depth = fl_p ** 1.55
            fl_y = int(horizon_y + fl_depth * (road_bottom_y - horizon_y))
            fl_w = int(road_far_half + fl_depth * (road_near_half - road_far_half))
            fl_curvature = side_at_forward(finish_fwd) - base_side
            fl_curve = project_side(fl_curvature, finish_fwd, fl_depth)
            fl_offset_px = int((-offset / track_half) * fl_w)
            fl_cx = rect.centerx + fl_curve + fl_offset_px
            fl_thickness = max(2, int(4 + fl_depth * 10))

            # checkered pattern across road
            num_checks = max(4, int(8 + fl_depth * 12))
            check_total_w = fl_w * 2
            check_w = max(2, check_total_w // num_checks)
            for ci in range(num_checks):
                cx_left = fl_cx - fl_w + ci * check_w
                cx_right = min(cx_left + check_w, fl_cx + fl_w)
                if ci % 2 == 0:
                    color = (240, 240, 240)
                else:
                    color = (20, 20, 20)
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(cx_left, fl_y - fl_thickness // 2, cx_right - cx_left, fl_thickness))

            # side markers
            marker_h = max(4, int(6 + fl_depth * 20))
            pygame.draw.rect(self.screen, (240, 240, 240),
                             pygame.Rect(fl_cx - fl_w - 6, fl_y - marker_h, 6, marker_h))
            pygame.draw.rect(self.screen, (240, 240, 240),
                             pygame.Rect(fl_cx + fl_w, fl_y - marker_h, 6, marker_h))

        # -- lap complete celebration overlay --
        if self.finish_line_timer > 0:
            flash_alpha = int(clamp(self.finish_line_timer / (FPS * 0.3) * 100, 0, 100))
            flash = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            flash.fill((255, 255, 220, flash_alpha))
            self.screen.blit(flash, (rect.left, rect.top))

        if abs(offset_norm) > 0.8:
            vignette = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            alpha = int((abs(offset_norm) - 0.8) / 0.2 * 120)
            vignette.fill((220, 44, 44, alpha))
            self.screen.blit(vignette, (rect.left, rect.top))

        self.draw_cockpit_overlay(rect, contact_side)

    def draw_cockpit_overlay(self, rect: pygame.Rect, contact_side: int):
        frame_color = (14, 18, 24)
        glass_tint = (170, 190, 210, 24)

        glass = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glass, glass_tint, pygame.Rect(0, 0, rect.width, int(rect.height * 0.64)))
        self.screen.blit(glass, (rect.left, rect.top))

        pygame.draw.rect(self.screen, frame_color, pygame.Rect(rect.left, rect.top, rect.width, 10))

        hood_top_y = rect.bottom - 132
        hood = [
            (rect.centerx - 238, rect.bottom),
            (rect.centerx + 238, rect.bottom),
            (rect.centerx + 138, hood_top_y),
            (rect.centerx - 138, hood_top_y),
        ]
        pygame.draw.polygon(self.screen, (242, 150, 56), hood)
        pygame.draw.polygon(self.screen, (84, 44, 12), hood, 3)

        stripe = [
            (rect.centerx - 18, rect.bottom),
            (rect.centerx + 18, rect.bottom),
            (rect.centerx + 10, hood_top_y),
            (rect.centerx - 10, hood_top_y),
        ]
        pygame.draw.polygon(self.screen, (52, 57, 70), stripe)
        pygame.draw.line(self.screen, (255, 208, 148), (rect.centerx - 140, hood_top_y + 4), (rect.centerx + 140, hood_top_y + 4), 2)

        if contact_side != 0:
            left_corner = pygame.Vector2(rect.centerx - 222, rect.bottom - 6)
            right_corner = pygame.Vector2(rect.centerx + 222, rect.bottom - 6)
            touch_corner = left_corner if contact_side == -1 else right_corner
            wall_touch_x = rect.left + 18 if contact_side == -1 else rect.right - 18
            pygame.draw.line(self.screen, (255, 232, 170), touch_corner, (wall_touch_x, rect.bottom - 86), 4)
            for _ in range(6):
                jitter_x = random.randint(-16, 16)
                jitter_y = random.randint(-20, 0)
                pygame.draw.circle(self.screen, (255, 204, 92), (int(touch_corner.x + jitter_x), int(touch_corner.y + jitter_y)), random.randint(2, 4))

    def draw_minimap(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, (20, 28, 38), rect, border_radius=10)
        pygame.draw.rect(self.screen, (72, 94, 118), rect, 2, border_radius=10)

        world_w = self.env.world_max_x - self.env.world_min_x
        world_h = self.env.world_max_y - self.env.world_min_y
        scale = min((rect.width - 20) / max(world_w, 1.0), (rect.height - 34) / max(world_h, 1.0))
        map_w = world_w * scale
        map_h = world_h * scale
        map_left = rect.left + (rect.width - map_w) * 0.5
        map_top = rect.top + 20 + (rect.height - 26 - map_h) * 0.5

        def map_point(pt: pygame.Vector2):
            return (
                int(map_left + (pt.x - self.env.world_min_x) * scale),
                int(map_top + (pt.y - self.env.world_min_y) * scale),
            )

        track_points = [map_point(pt) for pt in self.env.centerline]
        track_px_width = max(2, int(self.env.track_width * scale))
        pygame.draw.lines(self.screen, TRACK_COLOR, True, track_points, track_px_width)
        pygame.draw.lines(self.screen, TRACK_EDGE, True, track_points, 1)

        start = self.env.centerline[0]
        tangent = self.env.segment_unit[0]
        normal = pygame.Vector2(-tangent.y, tangent.x)
        start_outer = map_point(start + normal * (self.env.track_width / 2))
        start_inner = map_point(start - normal * (self.env.track_width / 2))
        pygame.draw.line(self.screen, (245, 245, 245), start_outer, start_inner, 2)

        if len(self.path_points) > 1:
            scaled_path = [map_point(pt) for pt in self.path_points]
            pygame.draw.lines(self.screen, (116, 231, 173), False, scaled_path, 2)

        car_center = pygame.Vector2(*map_point(self.env.position))
        mini_size = max(5, int(7 * scale * 4))
        front = car_center + pygame.Vector2(math.cos(self.env.heading), math.sin(self.env.heading)) * mini_size
        left = car_center + pygame.Vector2(math.cos(self.env.heading + 2.55), math.sin(self.env.heading + 2.55)) * (mini_size * 0.7)
        right = car_center + pygame.Vector2(math.cos(self.env.heading - 2.55), math.sin(self.env.heading - 2.55)) * (mini_size * 0.7)
        pygame.draw.polygon(self.screen, CAR_COLOR, [front, left, right])

        mini_label = self.tiny_font.render("Mini-map", True, (206, 218, 230))
        self.screen.blit(mini_label, (rect.left + 10, rect.top + 8))

    def draw_training_hud(self, panel: pygame.Rect):
        pygame.draw.rect(self.screen, (23, 31, 42), panel, border_radius=12)
        pygame.draw.rect(self.screen, (74, 95, 119), panel, 2, border_radius=12)

        avg_reward = (sum(self.recent_rewards) / len(self.recent_rewards)) if self.recent_rewards else 0.0
        off_rate = (sum(self.recent_off_track) / len(self.recent_off_track)) if self.recent_off_track else 0.0

        lines = [
            f"Episode: {self.episode + 1}",
            f"Current Reward: {self.episode_reward:7.2f}",
            f"Recent Avg Reward: {avg_reward:7.2f}",
            f"Off-track Rate: {off_rate * 100:5.1f}%",
            f"Exploration (ε): {self.agent.epsilon:0.3f}",
            f"Total Laps: {self.total_laps}",
        ]

        for idx, line in enumerate(lines):
            text = self.tiny_font.render(line, True, (216, 225, 235))
            self.screen.blit(text, (panel.left + 14, panel.top + 14 + idx * 21))

        mini_rect = pygame.Rect(panel.left + 12, panel.bottom - 250, panel.width - 24, 236)
        self.draw_minimap(mini_rect)

    def draw_home(self):
        w, h = self.window_w, self.window_h
        t = self.home_tick

        # -- gradient background --
        for y in range(h):
            p = y / max(h - 1, 1)
            r = int(8 + p * 14)
            g = int(12 + p * 20)
            b = int(22 + p * 36)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (w, y))

        # -- twinkling stars --
        for sx, sy, brightness in self.home_stars:
            x = sx % w
            y = sy % h
            flicker = 0.5 + 0.5 * math.sin(t * 1.8 * brightness + sx * 0.07)
            alpha = int(clamp(brightness * flicker * 200, 20, 210))
            size = 1 if brightness < 0.6 else 2
            color = (alpha, alpha, int(min(255, alpha * 1.15)))
            pygame.draw.circle(self.screen, color, (x, y), size)

        # -- perspective road vanishing into horizon --
        vanish_x = w // 2 + int(math.sin(t * 0.3) * 30)
        vanish_y = int(h * 0.32)
        road_bottom_left = (w // 2 - int(w * 0.38), h)
        road_bottom_right = (w // 2 + int(w * 0.38), h)

        road_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        road_segments = 22
        for i in range(road_segments):
            t0 = i / road_segments
            t1 = (i + 1) / road_segments
            p0 = t0 ** 1.8
            p1 = t1 ** 1.8
            y0 = int(vanish_y + p0 * (h - vanish_y))
            y1 = int(vanish_y + p1 * (h - vanish_y))
            lx0 = int(vanish_x + p0 * (road_bottom_left[0] - vanish_x))
            rx0 = int(vanish_x + p0 * (road_bottom_right[0] - vanish_x))
            lx1 = int(vanish_x + p1 * (road_bottom_left[0] - vanish_x))
            rx1 = int(vanish_x + p1 * (road_bottom_right[0] - vanish_x))
            shade = 28 + int((i % 2) * 6) + int((1.0 - t0) * 18)
            pygame.draw.polygon(road_surface, (shade, shade, shade + 3, 180),
                                [(lx0, y0), (rx0, y0), (rx1, y1), (lx1, y1)])
            # edge lines
            pygame.draw.line(road_surface, (180, 190, 205, 140), (lx0, y0), (lx1, y1), 2)
            pygame.draw.line(road_surface, (180, 190, 205, 140), (rx0, y0), (rx1, y1), 2)
            # animated lane dashes
            lane_anim = int(t * 6.0)
            if (i - lane_anim) % 4 in (0, 1):
                cx0 = (lx0 + rx0) // 2
                cx1 = (lx1 + rx1) // 2
                dash_w0 = max(1, (rx0 - lx0) // 28)
                dash_w1 = max(1, (rx1 - lx1) // 28)
                pygame.draw.polygon(road_surface, (245, 220, 100, 200),
                                    [(cx0 - dash_w0, y0), (cx0 + dash_w0, y0),
                                     (cx1 + dash_w1, y1), (cx1 - dash_w1, y1)])
        self.screen.blit(road_surface, (0, 0))

        # -- animated car silhouette on the road --
        car_bob = math.sin(t * 2.5) * 3
        car_cx = w // 2 + int(math.sin(t * 0.7) * 18)
        car_cy = int(h * 0.72 + car_bob)
        car_w, car_h = 48, 26
        # body
        body = [
            (car_cx - car_w, car_cy + car_h // 2),
            (car_cx + car_w, car_cy + car_h // 2),
            (car_cx + car_w - 6, car_cy - car_h // 2),
            (car_cx - car_w + 6, car_cy - car_h // 2),
        ]
        pygame.draw.polygon(self.screen, (255, 179, 71), body)
        pygame.draw.polygon(self.screen, (180, 110, 30), body, 2)
        # windshield
        ws = [
            (car_cx - 18, car_cy - car_h // 2),
            (car_cx + 18, car_cy - car_h // 2),
            (car_cx + 12, car_cy - car_h // 2 - 14),
            (car_cx - 12, car_cy - car_h // 2 - 14),
        ]
        pygame.draw.polygon(self.screen, (100, 160, 220), ws)
        pygame.draw.polygon(self.screen, (60, 100, 160), ws, 1)
        # headlight glow
        glow_surf = pygame.Surface((80, 40), pygame.SRCALPHA)
        glow_alpha = int(120 + 40 * math.sin(t * 4.0))
        pygame.draw.ellipse(glow_surf, (255, 230, 160, glow_alpha), (0, 0, 80, 40))
        self.screen.blit(glow_surf, (car_cx - 40, car_cy + car_h // 2 - 6))

        # -- subtle speed lines --
        for i in range(8):
            line_y = int(h * 0.55 + i * 22 + (t * 80 + i * 47) % (h * 0.45))
            if line_y > h:
                line_y -= int(h * 0.45)
            line_alpha = int(clamp(40 + 20 * math.sin(t * 2 + i), 15, 70))
            line_len = random.randint(30, 80)
            side = -1 if i % 2 == 0 else 1
            lx = w // 2 + side * random.randint(180, 360)
            line_surf = pygame.Surface((line_len, 2), pygame.SRCALPHA)
            line_surf.fill((200, 210, 230, line_alpha))
            self.screen.blit(line_surf, (lx, line_y))

        # -- dark overlay behind text for readability --
        overlay = pygame.Surface((w, 220), pygame.SRCALPHA)
        overlay.fill((10, 15, 22, 160))
        overlay_y = h // 2 - 170
        self.screen.blit(overlay, (0, overlay_y))

        # -- title with shadow --
        title_text = "Autonomous Driving"
        title_sub = "DQN Demo"
        shadow_offset = 3
        title_surf = self.title_font.render(title_text, True, TEXT_COLOR)
        title_shadow = self.title_font.render(title_text, True, (0, 0, 0))
        sub_surf = self.large_font.render(title_sub, True, ACCENT)
        sub_shadow = self.large_font.render(title_sub, True, (0, 0, 0))

        title_y = h // 2 - 155
        self.screen.blit(title_shadow, title_shadow.get_rect(center=(w // 2 + shadow_offset, title_y + shadow_offset)))
        self.screen.blit(title_surf, title_surf.get_rect(center=(w // 2, title_y)))
        self.screen.blit(sub_shadow, sub_shadow.get_rect(center=(w // 2 + shadow_offset, title_y + 48 + shadow_offset)))
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(w // 2, title_y + 48)))

        # -- accent line under title --
        line_w = 260
        pulse = 0.7 + 0.3 * math.sin(t * 2.0)
        line_color = (int(ACCENT[0] * pulse), int(ACCENT[1] * pulse), int(ACCENT[2] * pulse))
        pygame.draw.line(self.screen, line_color, (w // 2 - line_w // 2, title_y + 78), (w // 2 + line_w // 2, title_y + 78), 3)

        # -- tagline --
        tagline = self.body_font.render("Watch a neural network learn to race", True, (194, 204, 214))
        self.screen.blit(tagline, tagline.get_rect(center=(w // 2, title_y + 106)))

        hint = self.tiny_font.render("Tune parameters and watch the AI improve in real-time", True, (140, 155, 170))
        self.screen.blit(hint, hint.get_rect(center=(w // 2, title_y + 138)))

        # -- play button --
        hover = self.play_button.rect.collidepoint(pygame.mouse.get_pos())
        self.play_button.draw(self.screen, self.large_font, hover)

        # -- version tag --
        ver = self.tiny_font.render("v1.0  |  Deep Q-Learning  |  PyTorch + Pygame", True, (80, 95, 110))
        self.screen.blit(ver, ver.get_rect(center=(w // 2, h - 30)))

    def draw_tuning(self):
        self.screen.fill(BG_COLOR)
        w = self.window_w

        heading = self.large_font.render("Training Settings", True, TEXT_COLOR)
        blurb = self.tiny_font.render("Higher learning rate trains faster but may be less stable.", True, (181, 193, 206))

        self.screen.blit(heading, heading.get_rect(center=(w // 2, 120)))
        self.screen.blit(blurb, blurb.get_rect(center=(w // 2, 165)))

        self.learning_slider.draw(self.screen, self.body_font, self.tiny_font)

        speed_info = self.tiny_font.render(
            f"Sim steps/frame: 3   |   NN learning rate: {self.learning_slider.value:.5f}",
            True,
            (176, 189, 201),
        )

        self.screen.blit(speed_info, (w // 2 - speed_info.get_width() // 2, 418))

        start_hover = self.start_training_button.rect.collidepoint(pygame.mouse.get_pos())
        self.start_training_button.draw(self.screen, self.body_font, start_hover)

        back_hover = self.back_button.rect.collidepoint(pygame.mouse.get_pos())
        self.back_button.draw(self.screen, self.tiny_font, back_hover)

    def draw_training(self):
        self.screen.fill(BG_COLOR)
        viewport = pygame.Rect(16, 16, self.window_w - 348, self.window_h - 32)
        panel = pygame.Rect(self.window_w - 320, 16, 304, self.window_h - 32)

        self.draw_first_person_view(viewport)
        self.draw_training_hud(panel)

        title = self.tiny_font.render("First-Person Training View", True, (225, 234, 243))
        tip = self.tiny_font.render("Irregular track + slower car: early drift, then stabilization as epsilon decays.", True, (181, 196, 212))
        self.screen.blit(title, (viewport.left + 14, viewport.top + 12))
        self.screen.blit(tip, (viewport.left + 14, viewport.bottom - 28))

        if self.finish_line_timer > 0:
            lap_text = self.body_font.render("Lap Complete!", True, (160, 255, 180))
            self.screen.blit(lap_text, lap_text.get_rect(center=(viewport.centerx, viewport.top + 48)))
        elif self.crash_timer > 0:
            crash_text = self.body_font.render("Wall contact detected - resetting episode...", True, (255, 196, 166))
            self.screen.blit(crash_text, crash_text.get_rect(center=(viewport.centerx, viewport.top + 48)))

        exit_hover = self.exit_sim_button.rect.collidepoint(pygame.mouse.get_pos())
        self.exit_sim_button.draw(self.screen, self.tiny_font, exit_hover)

    def draw_result(self):
        self.draw_training()

        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((8, 13, 18, 170))
        self.screen.blit(overlay, (0, 0))

        msg = self.large_font.render(self.result_message, True, TEXT_COLOR)
        sub = self.body_font.render("Returning to home screen...", True, (204, 214, 224))

        self.screen.blit(msg, msg.get_rect(center=(self.window_w // 2, self.window_h // 2 - 14)))
        self.screen.blit(sub, sub.get_rect(center=(self.window_w // 2, self.window_h // 2 + 34)))

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
                continue
            if event.type == pygame.WINDOWSIZECHANGED:
                self._handle_resize(event.x, event.y)
                continue

            if self.state == "home":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.play_button.rect.collidepoint(event.pos):
                        self.state = "tuning"

            elif self.state == "tuning":
                self.learning_slider.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.back_button.rect.collidepoint(event.pos):
                        self.state = "home"
                    elif self.start_training_button.rect.collidepoint(event.pos):
                        self.configure_training()

            elif self.state == "training":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.exit_sim_button.rect.collidepoint(event.pos):
                        self._exit_simulation_to_home()

            elif self.state == "result":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.state = "home"

    def update(self):
        if self.state == "home":
            self.home_tick += 1.0 / FPS
        elif self.state == "training":
            self.update_training()
        elif self.state == "result":
            self.result_timer -= 1
            if self.result_timer <= 0:
                self.state = "home"

    def render(self):
        if self.state == "home":
            self.draw_home()
        elif self.state == "tuning":
            self.draw_tuning()
        elif self.state == "training":
            self.draw_training()
        elif self.state == "result":
            self.draw_result()

        pygame.display.flip()

    def run(self):
        while self.running:
            self.process_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    AutonomousDrivingDemo().run()
