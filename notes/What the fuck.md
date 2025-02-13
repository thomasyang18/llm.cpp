# Like wtf 

My little script

```
Tensor Query Tool - Enter tensor names to inspect their values.
Enter tensor name (or 'exit' to quit): transformer.wte.weight
Top-left 5x5 values of the matrix:
[[-0.11010301 -0.03926672  0.03310751  0.13382645 -0.04847569]
 [ 0.04034033 -0.04861503  0.04624869 -0.09900099  0.08258584]
 [-0.12746179  0.04793796  0.18410145 -0.08931495  0.0831188 ]
 [-0.09271405 -0.305332    0.21120381 -0.04194934 -0.0738307 ]
 [-0.05063773 -0.11109029  0.1057948  -0.10009561  0.09852351]]
```

Does NOT line up with my program at all. I mean, the first row kind of matches transposed but....

```
First few values: 
-0.110103 0.178836 0.145671 0.0535858 0.0798318 
-0.0392667 0.154726 -0.0447735 0.0620254 0.100041 
0.0331075 0.100448 0.0641229 -0.0625321 -0.016332 
0.133826 0.157237 0.164127 -0.0402038 -0.00680172 
-0.0484757 -0.181487 -0.118167 0.00566566 0.102142 
```

I think my cnpy parsing library just sucks. 

Never mind, eigen just serializes the pointer thing column major by default so u gotta reverse the dimensions then transpose it back oml crazy shit. 