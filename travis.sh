# CHANGED_FILES=$(find codes/python/ -type f -name "*.py")

CHANGED_FILES=($(git diff --name-only $TRAVIS_COMMIT_RANGE))
echo "Changed files are \n $CHANGED_FILES"

for file in $CHANGED_FILES; do
    # Check if the last 3 characters are .py
    if [ ${file: -3} == ".py" ]; then
        python $file;
    else
        echo "$file is a directory"
    fi
done
