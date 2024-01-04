cd AMDMIGraphX
rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
cd build
make -j && make install
cd ../../
python3 -m venv ll2_venv
. ll2_venv/bin/activate
pip install -r requirements.txt
pip install -r gradio_requirements.txt
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
huggingface-cli login
optimum-cli export onnx --model meta-llama/Llama-2-7b-chat-hf models/llama-2-7b-chat-hf --task text-generation --framework pt --library transformers --no-post-process
python gradio_app.py
