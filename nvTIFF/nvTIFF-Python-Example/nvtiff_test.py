import argparse
import time

import cupy
import numpy as np
import matplotlib.pyplot as plt
import tifffile

import nvtiff

parser = argparse.ArgumentParser()

parser.add_argument('tiff_file', type=str, help='tiff file to decode.')
parser.add_argument('-o', '--output_file_prefix', type=str, default=None, help='Output file prefix to save decoded data. Will save one file per image in tiff file.')
parser.add_argument('-s', '--return_single_array', action='store_true', help='Return single array from nvTiff instead of list of arrays')
parser.add_argument('-c', '--check_output', action='store_true', help='Compare nvTiff output to reference CPU result')
parser.add_argument('-p', '--use_pinned_mem', action='store_true', help='Read TIFF data from pinned memory.')
parser.add_argument('-r', '--subfile_range', type=str, default=None, help='comma separated list of starting and ending file indices to decode, inclusive')

args = parser.parse_args()

print("Command line arguments:")
print(f"\ttiff_file: {args.tiff_file}")
print(f"\treturn_single_array: {args.return_single_array}")
print(f"\toutput_file_prefix: {args.output_file_prefix}")
print(f"\tcheck_output: {args.check_output}")
print(f"\tuse_pinned_mem: {args.use_pinned_mem}")
print(f"\tsubfile_range: {args.subfile_range}")
print()

subfile_range = None
if args.subfile_range:
    subfile_range = [int(x) for x in args.subfile_range.split(',')]


# Create cupy array to initialize CUDA)
dummy = cupy.ndarray(1)
del dummy

# Read using tiffile and copy to GPU
cupy.cuda.get_current_stream().synchronize()
t0 = time.time()
ref_imgs = tifffile.imread(args.tiff_file)
t1 = time.time()
ref_imgs_gpu = cupy.asarray(ref_imgs)
cupy.cuda.get_current_stream().synchronize()
t2 = time.time()
print(f"Time for tifffile:")
print(f"\tdecode:   {t1 - t0} s")
print(f"\th2d copy: {t2 - t1} s")
print(f"\ttotal:    {t2 - t0} s")

# Read single nvTiff
cupy.cuda.get_current_stream().synchronize()
t0 = time.time()
f = nvtiff.nvTiffFile(0, args.tiff_file, use_pinned_mem=args.use_pinned_mem)
t1 = time.time()
nvTiff_imgs_gpu = nvtiff.decode(f, subfile_range = subfile_range, return_single_array=args.return_single_array)
cupy.cuda.get_current_stream().synchronize()
t2 = time.time()
print(f"Time for nvTiff:")
print(f"\topen: {t1 - t0} s")
print(f"\tdecode: {t2 - t1} s")
print(f"\ttotal:  {t2 - t0} s")
print()

# Compare results
if args.check_output:
    print(f"Checking output...")
    if f.nsubfiles != 1 and subfile_range:
        ref_imgs = ref_imgs[subfile_range[0]: subfile_range[1]+1,:,:]

    if args.return_single_array:
        nvTiff_imgs = nvTiff_imgs_gpu.get()
        np.testing.assert_equal(ref_imgs, np.squeeze(nvTiff_imgs))
    else:
        nvTiff_imgs = [x.get() for x in nvTiff_imgs_gpu]
        for i in range(len(nvTiff_imgs)):
            if f.nsubfiles == 1:
                np.testing.assert_equal(ref_imgs, np.squeeze(nvTiff_imgs[i]))
            else:
                np.testing.assert_equal(ref_imgs[i,:,:], np.squeeze(nvTiff_imgs[i]))

    print(f"Output matches.")

if args.output_file_prefix:
    print(f"Writing nvTiff outputs to {args.output_file_prefix}_*.png...")
    if args.return_single_array:
        nvTiff_imgs = nvTiff_imgs_gpu.get()
        for i in range(nvTiff_imgs.shape[0]):
            plt.imsave(f"{args.output_file_prefix}_{i}.png", nvTiff_imgs[i,:,:,:])
    else:
        nvTiff_imgs = [x.get() for x in nvTiff_imgs_gpu]
        for i, nvTiff_img in enumerate(nvTiff_imgs):
            plt.imsave(f"{args.output_file_prefix}_{i}.png", nvTiff_img)

