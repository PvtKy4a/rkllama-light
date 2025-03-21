PROG_NAME=rkllama-light

PROG_INSTALL_PATH=/usr/local/bin

LIB_NAME=librkllmrt.so

LIB_INSTALL_PATH=/usr/local/lib

WORKSPACE_DIR=$HOME/.$PROG_NAME

python3 -m venv env

source ./env/bin/activate

pip install --no-cache-dir -r ./requirements.txt

pyinstaller --onefile --name $PROG_NAME ./src/rkllama_light.py

deactivate

if [ ! -e "./dist/$PROG_NAME" ]; then
    echo "Install error. The executable file was not compiled."
    exit 1
fi

sudo cp ./dist/$PROG_NAME $PROG_INSTALL_PATH

sudo cp ./lib/$LIB_NAME $LIB_INSTALL_PATH

if [[ ! -d "${WORKSPACE_DIR}" ]]; then
    mkdir -p ${WORKSPACE_DIR}
fi

echo "Install success."