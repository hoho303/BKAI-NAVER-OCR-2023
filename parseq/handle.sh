for img_folder in $(ls ../data/results/images)
do
    img_dir=../data/results/images/$img_folder
    echo $img_dir
    mv $img_dir/* ../data/results/images
    rm -rf $img_dir
done