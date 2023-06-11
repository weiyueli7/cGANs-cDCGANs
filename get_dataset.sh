#!/bin/bash

# Google Drive folder link
drive_folder_link="https://drive.google.com/drive/folders/1s6ZWE4FSumKG_Wc5xXGB_4Sxwme80sry?usp=sharing"

data_folder="data/INTEL"
mkdir -p "$data_folder"

folder_id=$(echo "$drive_folder_link" | awk -F'/' '{print $NF}')

# you may need to edit the line below to add the path to your local bin folder!
# export PATH="/home/wel019/.local/bin:$PATH"

if ! command -v gdown &>/dev/null; then
    pip install gdown
fi

gdown --folder "$folder_id" --output "$data_folder"

for zip_file in "$data_folder"/*.zip; do
    unzip "$zip_file" -d "$data_folder"
    rm "$zip_file"
done
