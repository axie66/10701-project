# GPT2 experiments
# from checkpoint
#python3 train_transformer.py \
#    --encoder_type resnet50 \
#    --batch_size 16 \
#    --freeze_encoder \
#    --pretrained_encoder \
#    --epochs 10 \
#    --first_epoch_warmup 0 \
#    --decoder_type gpt2 \
#    --lr 0.00002 \
#    --ckpt_dir checkpoint/2021-11-27_10:32:50.450736

# python3 train_transformer.py \
#     --encoder_type resnet50 \
#     --batch_size 32 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --first_epoch_warmup 0.5 \
#     --decoder_type gpt2 \
#     --lr 0.00002

# Adafactor
#python3 train_transformer.py \
#    --encoder_type resnet50 \
#    --batch_size 16 \
#    --freeze_encoder \
#    --pretrained_encoder \
#    --epochs 10 \
#    --first_epoch_warmup 0 \
#    --decoder_type gpt2 \
#    --adafactor \
#    --lr 0.001 \
#    --ckpt_dir checkpoint/2021-11-27_12:14:24.311983

python3 train_transformer.py \
    --encoder_type resnet50 \
    --batch_size 16 \
    --freeze_encoder \
    --pretrained_encoder \
    --epochs 10 \
    --first_epoch_warmup 0.5 \
    --decoder_type gpt2 \
    --adafactor \
    --lr 0.001

# T5 experiments
# # Adafactor
# python3 train_transformer.py \
#     --encoder_type resnet50 \
#     --batch_size 32 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --first_epoch_warmup 0.5 \
#     --adafactor \
#     --decoder_type t5 \
#     --lr 0.001

# # Adam w/ lr 1e-5
# python3 train_transformer.py \
#     --encoder_type resnet50 \
#     --batch_size 32 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --first_epoch_warmup 0.5 \
#     --decoder_type t5 \
#     --lr 0.00001

# # Adam w/ lr 2e-5
# python3 train_transformer.py \
#     --encoder_type resnet50 \
#     --batch_size 32 \
#     --freeze_encoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --first_epoch_warmup 0.5 \
#     --decoder_type t5 \
#     --lr 0.00002

# # Freeze T5, only train projection MLP
# python3 train_transformer.py \
#     --encoder_type resnet50 \
#     --batch_size 32 \
#     --freeze_encoder \
#     --freeze_decoder \
#     --pretrained_encoder \
#     --epochs 10 \
#     --decoder_type t5 \
#     --lr 0.0001
