test -f BlurDatasetImage.zip || wget http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/data/BlurDatasetImage.zip
test -f BlurDatasetGT.zip || wget http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/data/BlurDatasetGT.zip

unzip BlurDatasetGT.zip
unzip BlurDatasetImage.zip
mv gt mask 
