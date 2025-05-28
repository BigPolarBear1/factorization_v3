# factorization_v3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Useage: python3 QSv3.py -keysize 40

To do: Will only factor below 50 bits for now, THIS IS AS EXPECTED.
This is a first draft. I will be uploading improvements frequently now.
The approach here is to bypass smooth finding and directly take the GCD on quadratic coefficients.
Right now, it will bruteforce every coefficient and root combination, that is why its so slow.
I know the math behind it and will upload improvements soon.

In addition I also need to fix coefficient lifting to higher powers. That code is broken atm.

Once I finish with the main improvements I'll also upload an updated paper. I've written large sections of it already.
