Update 15 june:

I just uploaded QSv3_034.py

Use: python3 QSv3_034.py -keysize 100

With a factor base of 300, it will take about 30 seconds to build the iN map. This is one of the big things I'll need to improve eventually.
But it should then find enough smooths for 100 bit in a couple of seconds.

Uploaded PoC easily factors up to 100.

In addition, it is still an early draft.

To do: We need to fix lifting for powers of 2. That way we can eliminate trial division completely. This will be the biggest performance boost. This alone should let it overtake factorization_v2.
Then there is still a bunch of other things I need to fix here and there...and perhaps also think how I can further improve the current strategy.

In addition I also need to think about speeding up building the iN map, because once we get rid of trial division, this will be the biggest bottleneck by far.
