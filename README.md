# GPU Project: Ray Tracer with Reduced Precision
There are 7 branches, each producing the output using different data types:
- datatype_double: double
- datatype_float: float
- datatype_half: half
- datatype_bf16: bf16
- datatype_mixed_3: mixed precision 3
- datatype_mixed_2: mixed precision 2
- datatype_mixed_1: mixed precision 1


## Compiling
1. Select the branch with the desired data type.
2. Make sure `netpbm` is installed if you want to output the image as a jpg.
3. Run `make`.
4. If you want to recompile, make sure to run `make clean` before.


## Running the code
There are two options:
- Option 1: Run `make out.ppm`.<br>This removes the old ppm image and runs `./cudart > out.ppm`.


- Option 2: Run `make out.jpg`.<br>This removes the old jpg image and runs `ppmtojpeg out.ppm > out.jpg`.


## Check the outputs
The output is saved to `out.ppm` or `out.jpg` depending on which command you run.