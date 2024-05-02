xhost +local:root

docker run  -it \
            --detach \
            --name aifr_container \
            --gpus all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v ./:/root/aifr/bottle-classification-YOLOv8 \
            aifr_img

# docker run  -itd \
#             --name ultralytics_container \
#             --gpus all \
#             -v /tmp/.X11-unix:/tmp/.X11-unix \
#             -v ./:/root/aifr \
#             ultralytics/ultralytics
