import argparse

def set_parser():
    parser = argparse.ArgumentParser()

    # Python environment parameters
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--n_gpu', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--date', default='FinalCode', type=str)

    # Data parameters
    parser.add_argument('--region', default='GM', type=str)
    parser.add_argument('--IMAGE_SIZE', default=256, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--train_patient', default=100, type=int)
    parser.add_argument('--test_patient', default=50, type=int)
    parser.add_argument('--rgb', default=False, type=bool)
    parser.add_argument('--raw', default=1, type=int)
    parser.add_argument('--image_size', default=256, type=int)

    # Model parameters
    parser.add_argument('--model', default='Unet', type=str, help = 'Unet or TransUnet')

    parser.add_argument('--content_layers', default=[4,9,16,23,30], type=list)
    parser.add_argument('--style_layers', default=[], type=list)

    # Training parameters
    parser.add_argument('--EPOCHS', default=100, type=int)
    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--LR', default=0.0001, type=float)
    parser.add_argument('--model_save_path', default=f"D:\\ACPC\\", type=str)

    return parser.parse_args()