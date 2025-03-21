PROG_NAME=rkllama-light

if [ -d "./build" ]; then
    rm -r ./build
fi

if [ -d "./dist" ]; then
    rm -r ./dist
fi

if [ -d "./env" ]; then
    rm -r ./env
fi

if [ -e "./$PROG_NAME.spec" ]; then
    sudo rm ./$PROG_NAME.spec
fi

echo "Clean success."