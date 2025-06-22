from utils import plot
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot image from result.')
    parser.add_argument('--fn', type=str, required=True)
    parser.add_argument('--no_detail', action='store_true')
    
    args = parser.parse_args()
    
    plot(args.fn, not args.no_detail)
