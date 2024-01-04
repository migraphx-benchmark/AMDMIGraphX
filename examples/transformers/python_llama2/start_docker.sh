git clone --depth 1 --branch rocm-6.0.0 https://github.com/ROCm/AMDMIGraphX AMDMIGraphX
cd AMDMIGraphX
docker build -t migraphx .
cd ..
docker run -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/code/python_llama2 -w /code/python_llama2 --group-add video  -it migraphx
