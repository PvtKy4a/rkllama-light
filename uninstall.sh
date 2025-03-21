PROG_NAME=rkllama-light

PROG_INSTALL_PATH=/usr/local/bin

LIB_NAME=librkllmrt.so

LIB_INSTALL_PATH=/usr/local/lib

MODELS_CFG_NAME=rkllama_light_models.json

MODELS_CFG_PATH=$HOME/.config/

WORKSPACE_DIR=$HOME/.$PROG_NAME

if [ -e "$PROG_INSTALL_PATH/$PROG_NAME" ]; then
    echo "Removing an executable file."
    sudo rm $PROG_INSTALL_PATH/$PROG_NAME
else
    echo "The installed executable file was not found."
fi

if [ -e "$LIB_INSTALL_PATH/$LIB_NAME" ]; then
    echo "Removing the RKLLM library."
    sudo rm $LIB_INSTALL_PATH/$LIB_NAME
else
    echo "Installed library RKLLM not found"
fi

if [ -e "$HOME/.config/rkllama_light_models.json" ]; then
    echo "Removing models config."
    sudo rm $HOME/.config/rkllama_light_models.json
else
    echo "Models config not found."
fi

if [ -d "$WORKSPACE_DIR" ]; then
    echo "Removing working directory."
    rm -r $WORKSPACE_DIR
else
    echo "Working directory not found."
fi

echo "Uninstall success."