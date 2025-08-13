
Update: Those things at the bottom of chapter 6 in my paper about 2d sieving are wrong. Was making assumptions in my head without running the numbers. But I know this approach of sieving in bulk is the way to go. Because we can pre-calculate an awful lot of information really fast... there's no point in just a single sieve array like SIQS/MPQS does... once we figure out a modulus we want to use, we should do as much sieving as possible..... let me try to figure out the exact math for this... give me a few days max. 

Ok wait actually it is not that hard.

So for example mod 7 we have the following discrimant formulas that generate a multiple of 7:

N=4387</br>
Linear co: 1,6 and quad co: 1</br>
1^2+4\*4387\*1</br>
6^2+4\*4387\*1</br></br>

Linear co: 2,5 and quad co: 4</br>
2^2+4\*4387\*4</br>
5^2+4\*4387\*4</br></br>

etc, do this for all linear coefficients... (this is all information we can pull from the hashmap without addiitonal calculations).
Now we can create a 2d sieve interval where the increments in the width are mod 7. And increments in the heights represent the quadratic coefficient, and we save a mapping to remember which linear  coefficient that is coupled with.
Now we can do a lot of sieving at once.... let me get that code done.

