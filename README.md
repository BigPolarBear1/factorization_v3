Note: That last chapter in the paper is becoming a bit of a frankenstein chapter. I will first finish the PoC, then redo that entire chapter. Most of the math is already in there... its just not pretty looking and all over the place right now.

Just uploaded v3_005:

useage: python QSv3_005.py -keysize 30

Just uploading my work in progress. Bit slower then the previous version. We basically started over again, but this time using the correct simplified math.
We start with a small coefficient, alpha... make sure its a quadratic residue of N... if it is, that means, it must be found as a coefficient solution for every modulo prime. So for each prime, we check at which iN value this coefficient exists... and then finding the other square basically comes down to finding another iN value where everything maps to a solution. 
Next we are going to get rid of that loop, this code has only one loop (excluding the loop for alpha..that one matters less) and we're going to get rid of it... watch what happens in the coming days :).

Update: Almost there... proper PoC should drop any day now... 

I think tomorrow.... I know how to solve it now. If alpha is a small coefficient... I'm already doing it in the code but breaking after retrieving the first iN value... but it will actually have multiple ones (i.e in mod 13, the coefficient "1" will appear at multiple iN values)... and all of this is rather fortunate bc it lets us do something..... you will see. Big day tomorrow. I know I've been saying this all week now.... but that's bc I'm an inch away from breakthrough now.... just took a few days longer then I had hoped.. 
