from config import get_config
from train import fit_and_evaluate
if __name__ == '__main__':
    args = get_config()
    fit_and_evaluate(args)
    print(f'Training completed. Models saved to {args.save_dir}/')