docker run  -itd \
            --name aifr_container \
            --gpus all \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v ./:/root/aifr \
            aifr_img

# docker run  -itd \
#             --name ultralytics_container \
#             --gpus all \
#             -v /tmp/.X11-unix:/tmp/.X11-unix \
#             -v ./:/root/aifr \
#             ultralytics/ultralytics