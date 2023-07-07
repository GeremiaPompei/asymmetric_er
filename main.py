import argparse

from src.generators import generators_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('generator', help=f'type of generator to use: {list(generators_dict.keys())}')
    parser.add_argument('-f', '--file', help='name of file where save results', default=None)
    args = parser.parse_args()
    generators_dict[args.generator](args.file)


if __name__ == '__main__':
    main()
