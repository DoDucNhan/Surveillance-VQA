#!/usr/bin/env bash
if [ "$1" == '' ] || [ "$2" == '' ]; then
    echo "Usage: $0 <input folder> <output folder>";
    exit;
fi
ext=mp4
for file in "${1}"/*."${ext}"; do
    destination="${2}${file:${#1}:${#file}-${#1}-${ext}-4}";
    echo $destination
    mkdir -p "$destination";
    ffmpeg -i "$file" -vsync 0 "$destination/%d.jpg";
done
