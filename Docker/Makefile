run:
	docker build -t deeplab .

in:
	nvidia-docker run --runtime=nvidia -v /home/hagi:/work --rm -it deeplab:latest /bin/bash
