xhost +local:root

docker run  -it \
            --detach \
            --network host \
            --ipc=host \
            --name aifr_container \
            --gpus all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v ./:/root/aifr/bottle-classification-YOLOv8 \
            -v /home/quanvu/uts/aifr/wdwyl_ros1:/root/aifr/wdwyl_ros1 \
            aifr_img

# docker run  -itd \
#             --name ultralytics_container \
#             --gpus all \
#             -v /tmp/.X11-unix:/tmp/.X11-unix \
#             -v ./:/root/aifr \
#             ultralytics/ultralytics
