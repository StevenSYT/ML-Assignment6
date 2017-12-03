#!/bin/bash
woof="image1 image2 image3 image4 image5"
for a in $woof
do
        curl "http://www.utdallas.edu/~axn112530/cs6375/unsupervised/images/$a.jpg" > $a.jpg
done
