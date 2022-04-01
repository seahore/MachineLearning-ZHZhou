using MLEx;

/*
Ch4.Node root = Ch4.TreeGenerate(Ch4.defaultData2Train, Ch4.defaultAttrSet2, Ch4.PurityAlgorithm.Gini);
foreach (var d in Ch4.defaultData2Test) {
    Ch4.Classify(root, d);
}
*/

Ch5.BPNetwork net = new Ch5.BPNetwork(new int[] { 8, 8, 1 });
net.BP(Ch5.defaultData3, 0.5, 100);

foreach (var s in Ch5.defaultData3) {
    var t = net.Evaluate(s.Input);
}