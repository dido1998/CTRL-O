# Make base directory for storing clevrtex dataset
DATA_PATH=${DATA_PATH:=./outputs}
mkdir -p ${DATA_PATH}/clevrtex

BASE_URL=https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex

# Downloads
wget --no-verbose -O- ${BASE_URL}_full_part1.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_full_part2.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_full_part3.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_full_part4.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_full_part5.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/

wget --no-verbose -O- ${BASE_URL}_outd.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_camo.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_grassbd.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_pbg.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
wget --no-verbose -O- ${BASE_URL}_vbg.tar.gz | bsdtar -xf- -C ${DATA_PATH}/clevrtex/
