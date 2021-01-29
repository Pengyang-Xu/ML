
ofd1 = open('D:\\MyGit\\ML\Data\\x_y_seqindex.csv','r')
ofd2 = open('D:\\MyGit\\ML\Data\\OrderSeq.csv','r')
nfd = open('D:\\MyGit\\ML\Data\\OrderClassificationCheck.txt','w')
orders = []

for i in ofd2:
    t = i.strip().split(',')
    order,design,classification,seq = t
    orders.append(t)

dic = {}
dic[0]='Complex'
dic[1]='Normal'
dic[2]='Simple'
for i in ofd1:
    t = i.strip().split()
    index,p1,p2,p3 = t
    
    p = [p1,p2,p3]

    index_t = p.index(max(p) )
    
    nfd.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % 
              (orders[int(index)][0], orders[int(index)][2], dic[index_t] ,str(p1),str(p2),str(p3), orders[int(index)][3]))