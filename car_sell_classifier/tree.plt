digraph Tree {
node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
0 [label="price <= 59976.24\ngini = 0.487\nsamples = 8000\nvalue = [3360, 4640]\nclass = yes", fillcolor="#c8e4f8"] ;
1 [label="price <= 40070.154\ngini = 0.202\nsamples = 3483\nvalue = [398, 3085]\nclass = yes", fillcolor="#53aae8"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
2 [label="gini = 0.0\nsamples = 1545\nvalue = [0, 1545]\nclass = yes", fillcolor="#399de5"] ;
1 -> 2 ;
3 [label="gini = 0.326\nsamples = 1938\nvalue = [398, 1540]\nclass = yes", fillcolor="#6cb6ec"] ;
1 -> 3 ;
4 [label="km_per_year <= 24124.006\ngini = 0.451\nsamples = 4517\nvalue = [2962, 1555]\nclass = no", fillcolor="#f3c3a1"] ;
0 -> 4 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
5 [label="gini = 0.498\nsamples = 2806\nvalue = [1498, 1308]\nclass = no", fillcolor="#fcefe6"] ;
4 -> 5 ;
6 [label="gini = 0.247\nsamples = 1711\nvalue = [1464, 247]\nclass = no", fillcolor="#e9965a"] ;
4 -> 6 ;
}