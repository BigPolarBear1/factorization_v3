I just uploaded QSv3_032.py<br />
Use: python3 QSv3_032.py -keysize 50

A work in progress.<br /><br />
To do:<br /><br />
-Fix lifting for powers of 2.<br />
-Get rid of trial division for smooth candidates... now that we can determine the exponent, this is redundant<br />
-Work out the details of a strategy to combine coefficients such that the factors outside the factor base shrink. <br />
This should be possible. I have a general idea in my head already on how to approach this now algorithmically. <br /><br />
Once all that is done, it should blast past the performance of factorization_v2.<br />
Then I need to write the paper, and start porting to c++ and optimize everything (and in addition use thing such as block lanczos instead of gaussian elimination)<br /><br />


Microsoft shouldn't have fired my manager. They will pay the price. I'm never stopping. I'll destroy the world if I must.


