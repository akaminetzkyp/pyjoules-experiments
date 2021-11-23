g5k-setup-nvidia-docker -t
docker build -t pytorch .
docker run -it --rm -v $PWD:/tmp -w /tmp -p 8888:8888 --shm-size 8G pytorch
