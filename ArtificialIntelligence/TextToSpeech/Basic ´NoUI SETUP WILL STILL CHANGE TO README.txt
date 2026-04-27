#create venv
python -m venv xtts-env
#go into venv
xtts-env\Scripts\activate


#INSIDE VENV
pip install torch torchvision torchaudio
cd xtts-core
pip install -r requirements.txt

pip uninstall transformers -y
pip install transformers==4.36.2

#reinstall sentencepiece
pip install sentencepiece


pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

Now it should