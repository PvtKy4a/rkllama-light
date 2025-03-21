PROG_NAME=rkllama-light

PROG_INSTALL_PATH=/usr/local/bin

LIB_NAME=librkllmrt.so

LIB_INSTALL_PATH=/usr/local/lib

MODELS_CFG_NAME=rkllama_light_models.json

MODELS_CFG_PATH=$HOME/.config/

WORKSPACE_DIR=$HOME/.$PROG_NAME

if [ -e "$PROG_INSTALL_PATH/$PROG_NAME" ]; then
    sudo rm $PROG_INSTALL_PATH/$PROG_NAME
fi

if [ -e "$LIB_INSTALL_PATH/$LIB_NAME" ]; then
    sudo rm $LIB_INSTALL_PATH/$LIB_NAME
fi

if [ -e "$HOME/.config/rkllama_light_models.json" ]; then
    sudo rm $HOME/.config/rkllama_light_models.json
fi

if [ -d "$WORKSPACE_DIR" ]; then
    rm -r $WORKSPACE_DIR
fi

echo "Uninstall success."