mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

unzip -q train2014.zip
unzip -q val2014.zip
unzip -q test2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip

cd ../
mkdir annotations
cd annotations

wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2014.zip

unzip -q annotations_trainval2014.zip
unzip -q image_info_test2014.zip

rm annotations_trainval2014.zip
rm image_info_test2014.zip