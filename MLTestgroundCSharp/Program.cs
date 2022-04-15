using MLEx;
using MathNet.Numerics.LinearAlgebra;

/*
Ch4.Node root = Ch4.TreeGenerate(Ch4.defaultData2Train, Ch4.defaultAttrSet2, Ch4.PurityAlgorithm.Gini);
foreach (var d in Ch4.defaultData2Test) {
    Ch4.Classify(root, d);
}
*/

/*
Ch5.BPNetwork net = new Ch5.BPNetwork(new int[] { 8, 4, 1 });
// net.BGD(Ch5.defaultData3, 0.001, 500000);
net.Adam(Ch5.defaultData3, 0.001, 0.9, 0.999, 1000);

foreach (var s in Ch5.defaultData3) {
    var output = net.Evaluate(s.Input);
    Console.Write("[ ");
    for (int i = 0; i < s.Input.Count; i++) {
        Console.Write(s.Input[i] + "\t");
    }
    Console.Write("] => [ ");
    for (int i = 0; i < output.Count; i++) {
        Console.Write(output[i] + "\t");
    }
    Console.WriteLine("]");
}
*/

/*
for (int i = 0; i < Ch7.defaultData3.Count; ++i) {
    var res = Ch7.AODE(Ch7.defaultData3, Ch7.defaultData3[i].SubVector(0, Ch7.defaultData3[i].Count - 1), Ch7.discreteCount3);
    Console.WriteLine($"Result {i+1}: {res}");
}
*/

var c = Ch9.KMeans(Ch9.defaultData4, 3, 0.001);
