Uploaded QSv3_simd.pyx v001

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace
To run: python3 run_qs.py -base 1000 -keysize 140 -debug 1 -lin_size 100_000 -quad_size 20

To use the old PoC (that one will easily factor above 200 bit using pypy3):

pypy3 QSv3_050.py -base 6000 -keysize 200

The simd version is a work in progress. Just uploading since I'm taking a break for the day and running a marathon tomorrow (so wont get work done).
This version demonstrates the direction I'm working towards very well.

There's many things still to do:

1. I absolutely need to process each row for smooths while building the 2d interval, so we do not need to keep all rows in memory since it limits the size of the 2d sieve interval too much right now (plus I need to switch away from native python types). For the future I have some ideas on how to use that 2d interval to do some post-processing to find even more smooths... but then I'll have to write it to disk since it gets too big to keep in memory.

2. The create_residue_map function is very slow. Speeding this up will allow for a greater height in the 2d sieve interval. I need to rethink this function and use numpy arrays

3. Make sure to use the numpy c api (cnp) instead of the python numpy api, this removes some extra abstraction. In addition move to numpy and c types wherever I can. We should absolutely not be mixing numpy and native python types, since this ends up an order of magnitude slower then even just using natie python types.

4. Add more simd support and use static typing.

5. One big thing I also need to address is building the iN map... because being able to precompute this quickly is one of the main advantages of my number theory. Right now it still takes too long.

6. etc etc.....

It will be really fast once everything is done. The implementation right now is still kind of sloppy... but anyway, if I die tomorrow from running, you can do it yourself too now.
