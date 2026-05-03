"""
RL Pong — DQN + Target Network (PyTorch + Pygame)
==================================================
Install:  pip install torch pygame numpy
Run:      python rl_pong.py

Controls:
  SPACE   — pause / resume
  R       — reset agent
  1/2/3/4 — speed  1x / 5x / 20x / 100x
  Q / ESC — quit
"""

import sys, math, random, collections
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────
# HYPER-PARAMETERS
# ──────────────────────────────────────────────
LR             = 3e-4
GAMMA          = 0.99
EPS_START      = 1.0
EPS_MIN        = 0.05
EPS_DECAY      = 0.995      # per-episode decay (much slower → more exploration)
MEM_SIZE       = 10_000
BATCH_SIZE     = 128
TRAIN_EVERY    = 4          # env steps between gradient updates
TARGET_SYNC    = 200        # steps between copying online → target net
HIDDEN         = 128        # wider hidden layer

# ──────────────────────────────────────────────
# GAME CONSTANTS
# ──────────────────────────────────────────────
GW, GH     = 520, 380
PAD_X      = 24
PAD_W      = 14
PAD_H      = 72
PAD_SPEED  = 5
BALL_R     = 8
BALL_SPEED = 3.4
BALL_MAX   = 5.8

# ──────────────────────────────────────────────
# COLOURS
# ──────────────────────────────────────────────
BG         = (5, 10, 24)
GRID       = (15, 35, 75)
PADDLE_C   = (30, 100, 220)
PADDLE_HL  = (80, 160, 255)
BALL_C     = (210, 235, 255)
WALL_C     = (20, 55, 110)
WHITE      = (255, 255, 255)
PANEL_BG   = (10, 18, 38)
PANEL_BD   = (25, 55, 110)
TEXT_PRI   = (200, 220, 255)
TEXT_SEC   = (90, 120, 170)
BAR_BG     = (20, 35, 65)
BAR_FG     = (50, 120, 240)
CHART_LINE = (50, 130, 240)
GREEN      = (60, 200, 120)
AMBER      = (220, 165, 40)

DEVICE = torch.device("cpu")   # CPU is fine for this size

# ──────────────────────────────────────────────
# NEURAL NETWORK  (5 → 128 → 128 → 3)
# ──────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 3),
        )
    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────
# DQN AGENT  (with target network)
# ──────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.online  = QNet().to(DEVICE)
        self.target  = QNet().to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt     = optim.Adam(self.online.parameters(), lr=LR)
        self.memory  = collections.deque(maxlen=MEM_SIZE)
        self.epsilon = EPS_START
        self.steps   = 0

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(s).argmax(dim=1).item())

    def push(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch  = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)

        s  = torch.tensor(np.array(s),  dtype=torch.float32, device=DEVICE)
        a  = torch.tensor(a,            dtype=torch.long,    device=DEVICE).unsqueeze(1)
        r  = torch.tensor(r,            dtype=torch.float32, device=DEVICE)
        ns = torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE)
        d  = torch.tensor(d,            dtype=torch.float32, device=DEVICE)

        # online Q for chosen actions
        q_vals = self.online(s).gather(1, a).squeeze(1)

        # target: r + γ * max_a' Q_target(s', a')  (zero if done)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=1).values
            targets = r + GAMMA * next_q * (1 - d)

        loss = nn.SmoothL1Loss()(q_vals, targets)   # Huber loss — more stable than MSE
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # sync target network
        self.steps += 1
        if self.steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    @property
    def phase(self) -> str:
        if self.epsilon > 0.5:  return "EXPLORING"
        if self.epsilon > 0.15: return "LEARNING"
        return "EXPLOITING"

    @property
    def phase_color(self):
        if self.epsilon > 0.5:  return (50, 130, 240)
        if self.epsilon > 0.15: return GREEN
        return AMBER

# ──────────────────────────────────────────────
# GAME ENVIRONMENT
# ──────────────────────────────────────────────
class PongEnv:
    def __init__(self):
        self.trail: list = []
        self.reset()

    def reset(self):
        self.bx   = GW * 0.65
        self.by   = GH * 0.5 + random.uniform(-100, 100)
        angle     = random.uniform(-0.4, 0.4)
        self.vx   = -BALL_SPEED
        self.vy   = math.sin(angle) * BALL_SPEED
        self.py   = GH * 0.5
        self.hits = 0
        self.done = False
        self.trail.clear()

    def state(self) -> np.ndarray:
        return np.array([
            self.bx / GW,
            self.by / GH,
            self.vx / BALL_MAX,
            self.vy / BALL_MAX,
            self.py / GH,
        ], dtype=np.float32)

    def step(self, action: int) -> float:
        if action == 0:
            self.py = max(PAD_H / 2, self.py - PAD_SPEED)
        elif action == 2:
            self.py = min(GH - PAD_H / 2, self.py + PAD_SPEED)

        self.bx += self.vx
        self.by += self.vy

        if self.by - BALL_R <= 0:
            self.by = BALL_R;       self.vy =  abs(self.vy)
        if self.by + BALL_R >= GH:
            self.by = GH - BALL_R;  self.vy = -abs(self.vy)
        if self.bx + BALL_R >= GW:
            self.bx = GW - BALL_R;  self.vx = -abs(self.vx)

        pr     = PAD_X + PAD_W
        reward = 0.0

        # paddle hit
        if (self.vx < 0
                and self.bx - BALL_R <= pr
                and self.bx + BALL_R > PAD_X
                and self.py - PAD_H / 2 <= self.by <= self.py + PAD_H / 2):
            rel     = (self.by - self.py) / (PAD_H / 2)
            self.vx = min(BALL_MAX, abs(self.vx) * 1.05)
            self.vy = rel * 3.6
            spd     = math.hypot(self.vx, self.vy)
            if spd > BALL_MAX:
                self.vx = self.vx / spd * BALL_MAX
                self.vy = self.vy / spd * BALL_MAX
            self.bx  = pr + BALL_R + 1
            self.hits += 1
            reward   = 1.0 + self.hits * 0.1   # growing hit bonus

        # dense shaping: reward tracking the ball
        elif self.vx < 0:
            dist   = abs(self.by - self.py) / GH
            reward = 0.01 * max(0.0, 1.0 - dist)

        # miss → big penalty
        if self.bx - BALL_R <= 0:
            reward    = -3.0
            self.done = True

        return reward

# ──────────────────────────────────────────────
# RENDERING
# ──────────────────────────────────────────────
def draw_game(surf: pygame.Surface, env: PongEnv):
    surf.fill(BG)
    for x in range(0, GW, 40):
        pygame.draw.line(surf, GRID, (x, 0), (x, GH))
    for y in range(0, GH, 40):
        pygame.draw.line(surf, GRID, (0, y), (GW, y))
    for y in range(0, GH, 18):
        pygame.draw.line(surf, (25, 60, 120), (GW // 2, y), (GW // 2, y + 9))
    pygame.draw.rect(surf, WALL_C, (GW - 4, 0, 4, GH))

    env.trail.append((env.bx, env.by))
    if len(env.trail) > 14:
        env.trail.pop(0)
    for i, (tx, ty) in enumerate(env.trail):
        alpha  = int((i / len(env.trail)) * 100)
        radius = max(1, int(BALL_R * (i / len(env.trail)) * 0.7))
        ts = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(ts, (100, 170, 255, alpha), (radius, radius), radius)
        surf.blit(ts, (int(tx) - radius, int(ty) - radius))

    pad_rect = pygame.Rect(PAD_X, int(env.py - PAD_H / 2), PAD_W, PAD_H)
    pygame.draw.rect(surf, PADDLE_C, pad_rect, border_radius=4)
    pygame.draw.rect(surf, PADDLE_HL,
                     pygame.Rect(PAD_X, int(env.py - PAD_H / 2), 4, PAD_H),
                     border_radius=4)
    pygame.draw.circle(surf, BALL_C, (int(env.bx), int(env.by)), BALL_R)

    if env.hits > 0:
        font_size  = min(120, 38 + env.hits * 5)
        gf         = pygame.font.SysFont("monospace", font_size, bold=True)
        gs         = gf.render(str(env.hits), True, (60, 110, 220))
        gs.set_alpha(min(55, env.hits * 5))
        surf.blit(gs, gs.get_rect(center=(GW // 2, GH // 2)))


def draw_panel(surf, panel_rect, agent, env, ep, best, ep_hits, chart_pts, fonts, speed):
    x0, y0, pw, ph = panel_rect
    pygame.draw.rect(surf, PANEL_BG, panel_rect, border_radius=12)
    pygame.draw.rect(surf, PANEL_BD, panel_rect, width=1, border_radius=12)

    pad = 18;  cx = x0 + pad;  cy = y0 + pad

    # phase badge
    pc = agent.phase_color
    bb = pygame.Surface((pw - pad * 2, 26), pygame.SRCALPHA)
    bb.fill((*pc, 30));  surf.blit(bb, (cx, cy))
    bt = fonts["sm"].render(agent.phase, True, pc)
    surf.blit(bt, bt.get_rect(center=(x0 + pw // 2, cy + 13)));  cy += 36

    pygame.draw.line(surf, PANEL_BD, (cx, cy), (x0 + pw - pad, cy));  cy += 10

    half = (pw - pad * 2) // 2
    for label, val, col, ox in [
        ("Episode",     str(ep),    TEXT_PRI, 0),
        ("Best streak", str(best),  TEXT_PRI, half),
    ]:
        surf.blit(fonts["xs"].render(label, True, TEXT_SEC), (cx + ox, cy))
        surf.blit(fonts["lg"].render(val,   True, col),      (cx + ox, cy + 14))
    cy += 50

    avg = sum(ep_hits[-10:]) / len(ep_hits[-10:]) if ep_hits else 0.0
    for label, val, col, ox in [
        ("Hits now",    str(env.hits), GREEN,    0),
        ("Avg (10 ep)", f"{avg:.1f}",  TEXT_PRI, half),
    ]:
        surf.blit(fonts["xs"].render(label, True, TEXT_SEC), (cx + ox, cy))
        surf.blit(fonts["lg"].render(val,   True, col),      (cx + ox, cy + 14))
    cy += 50

    # epsilon bar
    surf.blit(fonts["xs"].render("Exploration (ε)", True, TEXT_SEC), (cx, cy))
    es = fonts["xs"].render(f"{agent.epsilon:.3f}", True, TEXT_SEC)
    surf.blit(es, (x0 + pw - pad - es.get_width(), cy));  cy += 16
    bw = pw - pad * 2
    pygame.draw.rect(surf, BAR_BG, (cx, cy, bw, 5), border_radius=3)
    fw = int(bw * agent.epsilon)
    if fw > 0:
        pygame.draw.rect(surf, BAR_FG, (cx, cy, fw, 5), border_radius=3)
    cy += 18

    pygame.draw.line(surf, PANEL_BD, (cx, cy), (x0 + pw - pad, cy));  cy += 10

    # chart
    surf.blit(fonts["xs"].render("Avg hits / episode", True, TEXT_SEC), (cx, cy))
    cy += 16;  ch = 80;  cw = pw - pad * 2
    pygame.draw.rect(surf, BAR_BG, (cx, cy, cw, ch), border_radius=4)
    if len(chart_pts) >= 2:
        mx = max(max(chart_pts), 1)
        pts = [(cx + int(i / (len(chart_pts) - 1) * (cw - 2)),
                cy + ch - 2 - int((v / mx) * (ch - 6)))
               for i, v in enumerate(chart_pts)]
        fs  = pygame.Surface((cw, ch), pygame.SRCALPHA)
        fp  = [(p[0]-cx, p[1]-cy) for p in [(cx, cy+ch)] + pts + [(pts[-1][0], cy+ch)]]
        pygame.draw.polygon(fs, (*CHART_LINE, 35), fp)
        surf.blit(fs, (cx, cy))
        pygame.draw.lines(surf, CHART_LINE, False, pts, 2)
    cy += ch + 12

    pygame.draw.line(surf, PANEL_BD, (cx, cy), (x0 + pw - pad, cy));  cy += 10

    # speed buttons
    surf.blit(fonts["xs"].render("Speed", True, TEXT_SEC), (cx, cy));  cy += 16
    btnw = (pw - pad * 2 - 9) // 4
    for i, s in enumerate([1, 5, 20, 100]):
        bx_ = cx + i * (btnw + 3)
        pygame.draw.rect(surf, BAR_FG if s == speed else BAR_BG,
                         (bx_, cy, btnw, 22), border_radius=4)
        pygame.draw.rect(surf, PANEL_BD, (bx_, cy, btnw, 22), width=1, border_radius=4)
        lbl = fonts["xs"].render(f"{s}x", True, WHITE if s == speed else TEXT_SEC)
        surf.blit(lbl, lbl.get_rect(center=(bx_ + btnw // 2, cy + 11)))
    cy += 32
    for h in ["SPACE pause  R reset", "1/2/3/4 speed  Q quit"]:
        hs = fonts["xs"].render(h, True, (50, 75, 120))
        surf.blit(hs, hs.get_rect(centerx=x0 + pw // 2, y=cy));  cy += 16


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    pygame.init()
    pygame.display.set_caption("RL Pong — DQN + Target Network")

    PANEL_W   = 230; GAP = 12
    screen    = pygame.display.set_mode((GW + GAP + PANEL_W, GH))
    clock     = pygame.time.Clock()
    game_surf = pygame.Surface((GW, GH))
    panel_rect = (GW + GAP, 0, PANEL_W, GH)

    fonts = {
        "lg": pygame.font.SysFont("monospace", 26, bold=True),
        "md": pygame.font.SysFont("monospace", 18, bold=True),
        "sm": pygame.font.SysFont("monospace", 13, bold=True),
        "xs": pygame.font.SysFont("monospace", 11),
    }

    agent      = Agent()
    env        = PongEnv()
    ep         = 0
    best       = 0
    ep_hits:   list = []
    chart_pts: list = []
    step_n     = 0
    speed      = 1
    paused     = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:  paused = not paused
                if event.key == pygame.K_r:
                    agent = Agent(); env.reset()
                    ep = best = step_n = 0
                    ep_hits.clear(); chart_pts.clear()
                if event.key == pygame.K_1: speed = 1
                if event.key == pygame.K_2: speed = 5
                if event.key == pygame.K_3: speed = 20
                if event.key == pygame.K_4: speed = 100

        if not paused:
            for _ in range(speed):
                s  = env.state()
                a  = agent.act(s)
                r  = env.step(a)
                ns = env.state()
                agent.push(s, a, r, ns, env.done)
                step_n += 1
                if step_n % TRAIN_EVERY == 0:
                    agent.train_step()
                if env.done:
                    agent.decay_epsilon()       # decay once per episode
                    best = max(best, env.hits)
                    ep_hits.append(env.hits)
                    ep += 1
                    if ep % 5 == 0:
                        sl = ep_hits[-10:]
                        chart_pts.append(sum(sl) / len(sl))
                        if len(chart_pts) > 60: chart_pts.pop(0)
                    env.reset()

        screen.fill((2, 5, 14))
        draw_game(game_surf, env)
        screen.blit(game_surf, (0, 0))
        draw_panel(screen, panel_rect, agent, env,
                   ep, best, ep_hits, chart_pts, fonts, speed)

        if paused:
            ov = pygame.Surface((GW, GH), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 120)); screen.blit(ov, (0, 0))
            msg = fonts["lg"].render("PAUSED — SPACE to resume", True, TEXT_PRI)
            screen.blit(msg, msg.get_rect(center=(GW // 2, GH // 2)))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()