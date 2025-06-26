Just uploaded QSv3_010.py

Useage: python3 QSv3_010.py -keysize 40

It is very slow still. It is as intended. Just uploading my work in progress for the day.
However this demonstrates indexing a hashmap by relative i-value. So that any coefficient pairings we find are garantueed to be at the correct i-value.

TO DO: I will finish the paper first tomorrow. 
I've already added a bit about how this i-value is actually just the quadratic coefficient.
Since these quadratic coefficients dictate how many times N the quadratic should produce... we can easily verify if a given linear coefficient pairing is at the correct quadratic coefficients.
And if not, we can adjust the quadratic coefficient until it is. I'll add this to the paper tomorrow...and then finish the code. I think I'm almost at the end of this now.... atleast I am seeing a mechanism now that allows me to finish it.
Something that can tell me if a linear coefficient pairing is right or wrong in the integers.
