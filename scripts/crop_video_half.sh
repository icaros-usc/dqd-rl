# Crops a video to show only the left half.
#
# Usage:
#    bash scripts/crop_video_half.sh INPUT OUTPUT
ffmpeg -i $1 -vf crop=1/2*in_w:in_h:0:0 $2
