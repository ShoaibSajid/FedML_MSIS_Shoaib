git clone --recurse-submodules https://github.com/ShoaibSajid/Yolov5_DeepSORT_PseudoLabels.git
cd Yolov5_DeepSORT_PseudoLabels 
pip install -r requirements.txt
rm -r yolov5
git clone --recursive https://github.com/ShoaibSajid/yolov5.git

apt-get install libfreetype6-dev -y
pip uninstall pillow
pip install --no-cache-dir pillow

pip install fedml --upgrade
pip install -r requirements.txt

# Run Docker
# docker run -itd  --restart unless-stopped --name fedml_shoaib  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -e DISPLAY=$DISPLAY --volume  $PWD:$PWD --volume /media/:/media/ --workdir $PWD --volume /host:/ --gpus all --network host --ipc host --privileged shoaibs/fedml_shoaib /bin/bash

# Validate using Yolo
# python /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/val.py --data /home/shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection/config/multi_domain/data/3class_bdd_full_path.yaml --weights /home/shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection/runs/train/multidomain/MultiDomain_Server/weights/aggr_model_19.pt --device 6