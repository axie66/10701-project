# Basic (encoder frozen) RNN/LSTM/GRU without attention
# python3 -i train.py \
#     --encoder_type resnet50 \
#     --decoder_type rnn \
#     --lr 0.001 \
#     --batch_size 128 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --decoder_hidden_dim 512

# python3 -i train.py \
#     --encoder_type resnet50 \
#     --decoder_type lstm \
#     --lr 0.001 \
#     --batch_size 128 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --decoder_hidden_dim 512

# python3 -i train.py \
#     --encoder_type resnet50 \
#     --decoder_type gru \
#     --lr 0.001 \
#     --batch_size 128 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --decoder_hidden_dim 512

# python3 -i train.py \
#     --encoder_type resnet50 \
#     --decoder_type gru \
#     --lr 0.001 \
#     --batch_size 128 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --decoder_hidden_dim 512 \
#     --attention

python3 -i train.py \
    --encoder_type resnet50 \
    --decoder_type lstm \
    --lr 0.001 \
    --batch_size 128 \
    --freeze_encoder \
    --pretrained_encoder \
    --epochs 20 \
    --decoder_hidden_dim 512 \
    --attention \
    --ckpt_dir checkpoint/2021-11-27_04:54:49.599401