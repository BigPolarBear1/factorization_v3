Uploaded QSv3_simd.pyx v001

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
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

6. Change how parallezation is done. The way I'm doing now eats away too much ram and is extremely suboptimal. I.e things such as the partial smooth relation hashmap arn't shared amongst workers.

7. I really really really need to implement lifting aswell. Because if we cap our quadratic coefficient to a range from, for example, 1 to 10_000, then working within that limit we can fairly quickly calculate p-adic linear coefficinet solutions. Which in turn also allows us to work efficiently with a much smaller factor base... which then also speed up other areas of the algorithm such as testing for smooths. I think this may be a key ingredient... but let me address some of the more urgent items on this list first.

8. etc etc.....

It will be really fast once everything is done. The implementation right now is still kind of sloppy... but anyway, if I die tomorrow from running, you can do it yourself too now.

Note: If you have a quadratic coefficient and a linear coefficient in mod m. You can exactly calculate the amount of bits your smooth will be after dividing by the modulus. Hence small coefficients and a large modulus are more likely to yield smooths.
There may be a way to pull these from our precalculated datastructure efficiently... but I will investigate this idea some more in v4. I think for v3, this will be good for now.
