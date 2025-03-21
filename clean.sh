PROG_NAME=rkllama-light

if [ -d "./build" ]; then
    echo "Removing the build directory."
    rm -r ./build
fi

if [ -d "./dist" ]; then
    echo "Removing the distribution directory."
    rm -r ./dist
fi

if [ -d "./env" ]; then
    echo "Removing a virtual environment directory."
    rm -r ./env
fi

if [ -e "./$PROG_NAME.spec" ]; then
    echo "Removing a specification file."
    rm ./$PROG_NAME.spec
fi

echo "Clean success."