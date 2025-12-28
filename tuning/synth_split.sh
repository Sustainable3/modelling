#!/bin/bash

# splitting synth dataset at STOS
# this previously awful-looking script was adapted from a genAI output
# MD, XII 25

SRC_DIR="dane_syntetyczne"
DEST_DIR="synth_dataset"
TRAIN_PERCENT=70
VAL_PERCENT=20

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Directory '$SRC_DIR' not found."
    exit 1
fi

# Create destination directory structure
echo "Creating directory structure in '$DEST_DIR'..."
for split in training validation test; do
    mkdir -p "$DEST_DIR/$split/images"
    mkdir -p "$DEST_DIR/$split/labels"
done

# 1. Get list of file IDs
echo "Scanning files..."
# We list files in images, strip the extension to get the 'basename'
file_list=$(ls "$SRC_DIR/images" | sed 's/\.[^.]*$//')

# 2. Shuffle
shuffled_list=$(echo "$file_list" | shuf)

# 3. Calculate counts
total_files=$(echo "$shuffled_list" | wc -l)
train_limit=$(( total_files * TRAIN_PERCENT / 100 ))
val_limit=$(( total_files * VAL_PERCENT / 100 + train_limit ))

echo "Found $total_files pairs."
echo "Splitting: $train_limit Training | $(( val_limit - train_limit )) Validation | $(( total_files - val_limit )) Test"

# 4. Copy files
counter=0
echo "$shuffled_list" | while read -r basename; do
    if [ "$counter" -lt "$train_limit" ]; then
        target="training"
    elif [ "$counter" -lt "$val_limit" ]; then
        target="validation"
    else
        target="test"
    fi

    src_img=$(ls "$SRC_DIR/images/$basename".* 2>/dev/null | head -n 1)
    src_lbl="$SRC_DIR/labels/$basename.txt"

    if [[ -f "$src_img" && -f "$src_lbl" ]]; then
        cp "$src_img" "$DEST_DIR/$target/images/"
        cp "$src_lbl" "$DEST_DIR/$target/labels/"
    else
        echo "Warning: Matching pair missing for '$basename'. Skipping."
    fi

    ((counter++))
done

echo "---"
echo "Data split complete! Files stored in '$DEST_DIR'."
