IMPORTANT TTS IS BUILT ON OLD PYTHON SO WE NEED OLD PYTHON so download this version
needs to show up when you run python --version
https://www.python.org/downloads/release/python-31011/

also to avoid a later C++ issue:
had to download https://visualstudio.microsoft.com/de/visual-cpp-build-tools/
-> must select Desktop development with C++!!!


#create venv
python -m venv xtts-env
#go into venv
xtts-env\Scripts\activate

IF YOU CANT ENTER THE VENV THEN EXECUTE
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


#INSIDE VENV
cd xtts-core
pip install -r requirements.txt

pip uninstall transformers -y
pip install transformers==4.36.2

#reinstall sentencepiece
pip install sentencepiece


pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

Now it should work