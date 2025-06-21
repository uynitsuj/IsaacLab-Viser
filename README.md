# IsaacLab-Viser

![IsaacLabViser](media/IsaacLabViser.gif)

## Motivation

This project enables headless development and visualization for IsaacLab robotics simulations using [Viser](https://viser.studio/). Instead of relying on IsaacLab's built-in GUI, which can be cumbersome for development workflows, IsaacLab-Viser provides a web-based interface that allows you to:

- Develop and debug robot simulations in headless environments (SSH, Docker, cloud instances)
- Visualize robot states, trajectories, and sensor data through a browser
- Interact with simulations remotely without requiring a desktop environment
- Maintain the full power of IsaacLab while gaining flexibility in visualization and control

Perfect for researchers and developers who want to work with IsaacLab on remote servers or in containerized environments.

# Install Instructions
```
git clone --recurse-submodules https://github.com/uynitsuj/IsaacLab-Viser.git

conda create -n isaaclab python==3.10
conda activate isaaclab
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

cd IsaacLab-Viser
pip install -e .

cd dependencies/IsaacLab
sudo apt install cmake build-essential
./isaaclab.sh --install none

cd ../../
python scripts/run.py # First run might error

```

Running the `scripts/run.py` should give you:

```
╭──────────────── viser ────────────────╮
│             ╷                         │
│   HTTP      │ http://localhost:8080   │
│   Websocket │ ws://localhost:8080     │
│             ╵                         │
╰───────────────────────────────────────╯

```
