# PEFT-Leak: Gradient-Inversion Attacks & Defenses on VIT

This repository contains end-to-end code to reproduce the **PEFT-Leak** attacks and corresponding defenses on Vision Transformer (ViT) models fine-tuned with four parameter-efficient (PEFT) methods:

* Adapter  
* Prefix-Tuning  
* Bias-Tuning  
* LoRA  

Step 1: Clone and Navigate
# Clone the repository
git clone https://github.com/arukutiyash/Peftproject25.git

# Enter the project directory
cd Peftproject25

# Verify you're in the right place
ls -la
# You should see: attacks/, defenses/, evaluation/, federated/, main/, models/, utils/

Step 2: Install Dependencies
# Install all required Python packages
pip install -r requirements.txt

# Alternative: Install individually if requirements.txt fails
pip install torch>=1.12.0 torchvision>=0.13.0 torchaudio>=0.12.0
pip install matplotlib>=3.5.0 numpy>=1.21.0 tqdm>=4.64.0
pip install scikit-learn>=1.1.0 k_means_constrained>=0.7.0
pip install lpips>=0.1.4 pillow>=9.0.0 pandas>=1.3.0 seaborn>=0.11.0

Step 3: Verify Installation
# Quick dependency check
python -c "import torch, torchvision, matplotlib, numpy, sklearn; print('✅ All dependencies installed!')"

# Check CUDA availability use T4GPU in Colab or any other gpu for faster evaluation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

step 4: RUN the main file 
# 5-a.  Attack benchmark  
python main/run_attack_experiments.py --quick          # or omit --quick for full run

# 5-b.  Defense benchmark  (evaluates InstaHide, DP, MixUp, etc.)
python main/run_defense_experiments.py --quick         # add flags just like attacks

# 5-c.  Aggregate & pretty-print results tables
python main/generate_results.py                        # creates CSV / markdown tables





