Uploaded QSv3_056_2d_sieving_WIP.py

To use: pypy3 QSv3_056_2d_sieving_WIP.py -base 500 -keysize 100

TO use the old PoC (that one will easily factor above 200 bit):

pypy3 QSv3_050.py -base 6000 -keysize 200

Made some more improvements to 2d sieving. The biggest bottleneck right now is constructing the 2d sieve interval.
So it is time now to optimize that. In theory instead of constructing row by row, we should be able to fill out the interval in 2d dimension. Hence having to execute a whole less code.
Additionally this would then be perfect to be further optimized with SIMD I guess.
Let me start optimizing that code.

Because each process is now responsible for a range of quadratic coefficients, the process at index 0 (which has the smallest quadratic coefficient range) will yield the most smooth. Hence I think it is better to move parallelization to the sieving process itself instead.
There is still a lot of work to do with this 2d sieving. But I am feeling extremely optimistic about this approach (it may not seem like it yet, but just wait... you will see soon! I know what I'm doing now). I should also fix the paper soon, so that once the PoC is ready I can make some noise about it. It's definitely wildly different from default SIQS.

Anyway, I'll go for a run. Figure out the math after that to optimize constructing that 2d sieving interval and then tomorrow implement it in code... and then we should slowly start seeing the true strength of this approach. This really should be many times faster then normal SIQS and not even talking about its potential to be optimized further with SIMD.
