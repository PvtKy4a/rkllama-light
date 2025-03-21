PROG_NAME=rkllama-light

PROG_INSTALL_PATH=/usr/local/bin

LIB_NAME=librkllmrt.so

LIB_INSTALL_PATH=/usr/local/lib

WORKSPACE_DIR=$HOME/.$PROG_NAME

if [[ ! -d "./env" ]]; then
    echo "Creating a Python virtual environment."
    python3 -m venv ./env
else
    echo "An existing Python virtual environment was detected."
fi

if [ -e "./dist/$PROG_NAME" ]; then
    rm ./dist/$PROG_NAME
fi

source ./env/bin/activate

pip install --no-cache-dir -r ./requirements.txt

pyinstaller --onefile --name $PROG_NAME ./src/rkllama_light.py

deactivate

if [ ! -e "./dist/$PROG_NAME" ]; then
    echo "Install error. The executable file was not compiled."
    exit 1
fi

if [ -e "$PROG_INSTALL_PATH/$PROG_NAME" ]; then
    echo "Removing a previously installed executable file."
    sudo rm $PROG_INSTALL_PATH/$PROG_NAME
fi

echo "Installing the compiled executable file."
sudo cp ./dist/$PROG_NAME $PROG_INSTALL_PATH

if [ ! -e "$LIB_INSTALL_PATH/$LIB_NAME" ]; then
    echo "Installing the RKLLM library."
    sudo cp ./lib/$LIB_NAME $LIB_INSTALL_PATH
else
    echo "The RKLLM library is already installed."
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
    echo "Creating a working directory."
    mkdir $WORKSPACE_DIR
    mkdir $WORKSPACE_DIR/models
else
    echo "An existing working directory was found."
fi

echo "Install success."