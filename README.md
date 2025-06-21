# IsaacLab-Viser

![IsaacLabViser](media/IsaacLabViser.gif)

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