#!/bin/bash
# S2AM3D Interactive Demo Launcher Script

cd "$(dirname "$0")"

DECODER_CONFIG="configs/train.yaml"
DECODER_CKPT="../ckpt/S2AM3D_decoder.pt"

ENCODER_CONFIG="../encoder/configs/final/demo.yaml"
ENCODER_CKPT="../ckpt/Encoder.ckpt"

DATA_DIR="../demo"
DATA_PATH=""

HOST="0.0.0.0"
PORT=8080
DEVICE="cuda:0"

echo "=========================================="
echo "Starting S2AM3D Interactive Demo..."
echo "=========================================="
echo "Decoder config: $DECODER_CONFIG"
echo "Decoder checkpoint: $DECODER_CKPT"
echo ""
echo "Encoder config: $ENCODER_CONFIG"
echo "Encoder checkpoint: $ENCODER_CKPT"
echo ""
if [ -n "$DATA_PATH" ]; then
echo "Data path: $DATA_PATH"
elif [ -n "$DATA_DIR" ]; then
    echo "Data directory: $DATA_DIR"
fi
echo ""
echo "Server: http://$HOST:$PORT"
echo "=========================================="
echo ""

if [ ! -f "$DECODER_CONFIG" ]; then
    echo "❌ Error: Decoder config file not found: $DECODER_CONFIG"
    exit 1
fi

if [ ! -f "$DECODER_CKPT" ]; then
    echo "❌ Error: Decoder checkpoint file not found: $DECODER_CKPT"
    exit 1
fi

if [ ! -f "$ENCODER_CONFIG" ]; then
    echo "⚠️  Warning: Encoder config file not found: $ENCODER_CONFIG"
    ENCODER_CONFIG=""
    ENCODER_CKPT=""
elif [ ! -f "$ENCODER_CKPT" ]; then
    echo "⚠️  Warning: Encoder checkpoint file not found: $ENCODER_CKPT"
    ENCODER_CONFIG=""
    ENCODER_CKPT=""
else
    echo "✅ Encoder config check passed"
fi

if [ -n "$DATA_PATH" ] && [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  Warning: Data file not found: $DATA_PATH"
    DATA_PATH=""
elif [ -n "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then
    echo "⚠️  Warning: Data directory not found: $DATA_DIR"
    DATA_DIR=""
fi

CMD="python interactive_demo.py \
    --config $DECODER_CONFIG \
    --ckpt_path $DECODER_CKPT \
    --host $HOST \
    --port $PORT \
    --device $DEVICE"

if [ -n "$ENCODER_CONFIG" ] && [ -n "$ENCODER_CKPT" ]; then
    CMD="$CMD --encoder_config $ENCODER_CONFIG --encoder_ckpt $ENCODER_CKPT"
fi

if [ -n "$DATA_PATH" ] && [ -f "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
elif [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
    CMD="$CMD --data_dir $DATA_DIR"
fi

echo ""
echo "=========================================="
echo "Command:"
echo "$CMD"
echo "=========================================="
echo ""
echo "Starting server..."
echo "After startup, access: http://$HOST:$PORT"
echo "Press Ctrl+C to stop"
echo ""

eval $CMD
