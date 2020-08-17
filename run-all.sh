
cd ..
git clone https://github.com/eldar/pose-tensorflow
cd -

if command -v curl 1>/dev/null; then
    DOWNLOADER="curl -L -O"
else
    DOWNLOADER="wget -c"
fi

$DOWNLOADER https://github.com/italojs/facial-landmarks-recognition-/raw/master/shape_predictor_68_face_landmarks.dat


export CUDA_VISIBLE_DEVICES=0,1

pip install -r requirements.txt


python face_feature_extraction.py
python create_samples.py
python trainer.py

cd models/mpii
./download_models.sh
cd -

TF_CUDNN_USE_AUTOTUNE=0 python classify.py

python dash_app.py
