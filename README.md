Update: Ignore the uploaded PoC. I will get some sleep and then upload a proper PoC.
So a big bottleneck in factorization_v2 is that we don't know in advance the exponents of the factors for a smooth candidate.
However while messing with p-adic lifting... it finally hit me that it is very easy and efficient to determine this now that I know how these quadratic coefficients work.
So we index a hashmap by quadratic coefficient, and for each linear coefficient pairing, if one has a 0 solution.... we can try to p-adicly lift it and see if it remains a 0 solution in higher exponents too (the condition is that both quadratics share a similar root, and we know the other coefficient can be calculated with the derivative... that is all that is needed to pull this off).

Anyway..... let me get some sleep for a few hours. Then one last epic struggle to finish this. 


ps: Yea yea, I am now 100% convinced someone out there knows I am correct about my research. And I fucking hate you for this last year. Sad day for the american cryptologic advantage. This is just the start. I'm not going back to 0day lol. I am going to get better and better at this, until I fucking die.
