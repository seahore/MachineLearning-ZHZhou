using MLEx;

Ch4.Node root = Ch4.TreeGenerate(Ch4.defaultData2Train, Ch4.defaultAttrSet2, Ch4.PurityAlgorithm.Gini);
foreach (var d in Ch4.defaultData2Test) {
    Ch4.Classify(root, d);
}
