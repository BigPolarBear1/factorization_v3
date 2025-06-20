I GOT IT!

Use: QSv3_009.py -keysize 50

Added v3_009 ... just fixes a small bug where the modulus for y<sub>0</sub> wasn't calculated correctly... next version will include transfering the result to a primefield..

Update: I am still missing something big time. Let me spent the weekend reading this paper on number field sieve and dissecting it properly. I know there is a connection, but it's a little bit more complicated then what I am imagining in my head. I'll get there very soon...

Update: Oh ok, I think I see now what went wrong. Give me a few days to correct it.... was 99.99% there. Just this last thing now. I get it now.
I had already figured out the coefficients are derivatives of the quadratic from the other side of the congruence... so that was the key that I missed... that's how you adjust those results.
