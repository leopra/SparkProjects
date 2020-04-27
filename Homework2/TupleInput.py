def readTuplesSeq(inputfile):
	points = list()
	f = open(inputfile, "r")
	if f.mode == "r":
		for i in f:
			line = i.split(",")
			t = tuple(float(dim) for dim in line)
			points.append(t)
	f.close()
	return points	