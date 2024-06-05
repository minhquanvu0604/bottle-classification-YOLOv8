# Run XLaunch

docker run  -it `
            --detach `
            --network host `
            --ipc=host `
            --name aifr_container `
            --gpus all `
            -v ".:/root/aifr/bottle-classification-YOLOv8" `
	        -v "C:\PERSONAL DATA ROG\git\wdwyl_ros1:/root/aifr/wdwyl_ros1" ` # Absolute path
            # -v ".:/root/aifr/wdwyl_ros1" ` # Relative path
            -e DISPLAY=host.docker.internal:0.0 `
            --workdir /root/aifr/bottle-classification-YOLOv8 `
            aifr_img