# Autonomous Driving Demo (Python + Pygame + PyTorch)

A live deep reinforcement learning demo designed for high school students.
Tune the learning rate, click start, and watch a car learn to drive around an irregular race track in real time.

## What students see

- An animated splash screen with a perspective road, moving car, and twinkling stars.
- A settings screen with a `Learning Rate` slider (directly controls the neural network's learning rate).
- A fake-3D first-person training view with perspective road rendering and engine-cover cockpit overlay.
- The road visibly shifts when the car goes off-track, matching the mini-map behavior.
- A checkered start/finish line rendered in the first-person view with a lap-complete celebration flash.
- A real-time mini-map and simplified HUD showing: Episode, Current Reward, Recent Avg Reward, Off-track Rate, Exploration (epsilon), and Total Laps.
- `Exit Sim` button to leave training and return to the main screen at any time.
- The car initially drives poorly, then improves as DQN training progresses.
- Automatic success detection and return to the home screen once the policy is stable enough.
- Resizable/maximizable game window.

## How it works

- **Environment:** irregular closed 2D race track (F1-style shape) with off-track penalties and lap completion detection.
- **Agent:** Deep Q-Network (DQN) using PyTorch.
- **Network:** MLP (3 state features -> hidden layers -> Q-values for 9 actions).
- **Training:** epsilon-greedy exploration, replay buffer, target network sync, 3 simulation steps per frame.
- **State:** normalized heading error, lateral offset, and speed.
- **Actions:** combinations of steering and acceleration.
- **Reward:** forward progress + staying centered + smooth heading + lap completion bonus.

## Run

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the demo:

```powershell
python game.py
```

## Notes for presenters

- The demo is intentionally visual and fast to help students see RL behavior changes in real time.
- The `Learning Rate` slider (range 0.0003 - 0.003) directly controls the neural network's optimizer. Higher values train faster but may be less stable.
- Early episodes will show the car frequently going off-track; this is expected and visible in both the mini-map and first-person view.
- PyTorch may take longer to install than pure-Python packages.
