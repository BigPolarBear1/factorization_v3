Just uploaded my work in progress on implementing 2d sieving, it is still slower then the regular version but it should outperform the regular version once I work through the to-do list.

Use:

pypy3 QSv3_055_2d_sieving_WIP.py -keysize 100 -base 500

My other PoC will easily factor over 200 bit, but I hope to improve that with 2d sieving.

To use the old PoC:

pypy3 QSv3_050.py -keysize 200 -base 6000

I'll also edit the bottom of chapter 6 in the paper once I outperform the old PoC with 2d sieving... there's some errors in it right now, so just ignore that section on 2d sieving.
