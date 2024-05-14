# Run XLaunch

docker run  -it `
            --detach `
            --network host `
            --name aifr_container `
            --gpus all `
            -v ".:/root/aifr/bottle-classification-YOLOv8" `
            -e DISPLAY=host.docker.internal:0.0 `
            --workdir /root/aifr/bottle-classification-YOLOv8 `
            aifr_img