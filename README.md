To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br></br>
To run: (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter and currently worker support is also not working)

140 bit: python3 run_qs.py -keysize 140 -base 1000 -debug 1 -lin_size 100_000  -quad_size 1 (2 seconds)    </br>
160 bit: python3 run_qs.py -keysize 160 -base 1000 -debug 1 -lin_size 100_000  -quad_size 1 (8 seconds)    </br>
180 bit: python3 run_qs.py -keysize 180 -base 2000 -debug 1 -lin_size 100_000  -quad_size 1 (16 seconds)   </br>
200 bit: python3 run_qs.py -keysize 200 -base 6000 -debug 1 -lin_size 100_000  -quad_size 1 (100 seconds) </br>

Note: I am now finding myself doing functionally the same as SIQS with just more overhead to calculate the factor_base. Which is not good. I will explore the 2d sieving part again and also lifting, but I need to move away to v4 asap, bc I'm burning too much time here.

I'm so depressed. 

To get any strength out of this number theory compared to normal SIQS, you need to achieve 2d sieving. Without that it is pointless, then you just end up doing slower SIQS. 
Looking at this stack overflow PoC, seeing this trick they do to enumerate root values for multiple linear coefficients within the same quadratic coefficient... I think I can do the same by enumerating quadratic coefficients within a modulus.  I think some approach like that is how you succesfully do 2d sieving. Let me bash my head against that today.

I'm so awfully depressed. Today is really bad. Having a lot of suicidal thoughts. 
