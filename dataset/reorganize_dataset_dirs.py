import os, argparse, time
from tqdm import tqdm


def main(args):
    for name in tqdm(sorted(os.listdir(os.path.join(args.root, args.subdir)))):
        dirpath = os.path.join(args.root, args.subdir, name)
        # rename file name to 8 digits
        if not os.path.isdir(dirpath):
            index_name, suffix = os.path.splitext(name)
            if len(index_name) != 8:
                srcpath = os.path.join(args.root, args.subdir, name)
                dstpath = os.path.join(args.root, args.subdir, index_name[:8]+suffix)
                os.rename(srcpath, dstpath)
            continue
        # move file out from their individual folder
        filenames = os.listdir(dirpath)
        if len(filenames) == 0:
            os.rmdir(dirpath)
            continue
        filename = filenames[0]
        suffix = os.path.splitext(filename)[1]
        srcpath = os.path.join(args.root, args.subdir, name, filename)
        dstpath = os.path.join(args.root, args.subdir, name+suffix)
        dirpath = os.path.join(args.root, args.subdir, name)
        os.rename(srcpath, dstpath)
        os.rmdir(dirpath)
    with open('data_processing_log.txt', 'a') as f:
        f.write(f"Reorganized folder {args.subdir} to proper structure    - " + time.ctime() + '\n')
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data",
                        help='dataset root.')
    parser.add_argument('--subdir', type=str, default="step",
                        help='dataset sub-directory to be reorganized.')
    args = parser.parse_args()

    main(args)