Just uploaded v3_002:

useage: python QSv3_002.py -keysize 20

Only use 20 bit keys for this version. I've set all the parameters for 20 bit.
Its a proof of idea. Literally nothing more. To demonstrate the math from the paper.

Next I'm going to refactor the entire thing. Get rid of all the loops. 
There is a much better way to go about finding similar small coefficients mod p<sub>i</sub> .. but it is again a big refactoring project. Somewhere in the weekend it will get released.

Update: Instead of indexing that xN datastructure (or I<sub>i</sub>N in the paper) by means of xN, it is much easier to group small coefficients together from any xN mod p<sub>i</sub> and THEN calculate the corresponding xN value. 
Something like that.. hmm. I have a lot of ideas right now, but I am trying to come up with the MOST optimal one.. since I don't want to spent weeks doing rounds of refactoring my code. I want to get it right the first time around. I should go for a run in a few hours and think about this.

I'll get to work tomorrow.. just did some updates to the paper today.. I know how to write this code, just too god damn tired today.
